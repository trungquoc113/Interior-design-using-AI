"""Train a LoRA on Stable Diffusion 1.5 from style-labelled images (folder name = style, e.g. data/processed/train/Minimal/).

The base model stays frozen; only small LoRA matrices in the attention layers are trained (about a few MB).
Caption of every image = "<style> style interior design photograph".
Uses data/processed/train only, so validation/test images are never seen (no leakage).
To use residential-room data later, point --data to a folder with the same layout.

Quick test:   python -m src.design.train_lora --max-train-steps 50
Real run:     python -m src.design.train_lora --max-train-steps 2000
Out of GPU memory? add --resolution 448 (or 384). Output: outputs/lora/<name>/
"""
import argparse, csv, itertools, random
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from src import config as C


def caption_for(cls):
    return f"{cls.lower()} style interior design photograph"


class StyleImages(Dataset):
    def __init__(self, root, size):
        self.items = [(p, p.parent.name) for p in sorted(Path(root).rglob("*.jpg"))]
        if not self.items:
            raise SystemExit(f"No .jpg images found under {root}")
        self.classes = sorted({c for _, c in self.items})
        self.tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(size),
                             T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), self.classes.index(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.PROCESSED / "train"))
    ap.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--name", default="style_lora")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-train-steps", type=int, default=2000, help="number of images processed (micro-steps)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for LoRA training (use Google Colab / Kaggle if you have none).")

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    random.seed(a.seed); torch.manual_seed(a.seed)
    dev, fp16 = "cuda", torch.float16
    ds = StyleImages(a.data, a.resolution)
    print(f"{len(ds)} images, classes: {ds.classes}")
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)

    tok = CLIPTokenizer.from_pretrained(a.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(a.base_model, subfolder="text_encoder").to(dev)
    ids = tok([caption_for(c) for c in ds.classes], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        text_emb = text_encoder(ids)[0].to(fp16)  # one embedding per class, computed once
    del text_encoder
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(a.base_model, subfolder="vae").to(dev, dtype=fp16).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(a.base_model, subfolder="unet").to(dev, dtype=fp16)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # keep the small trainable part in fp32
    unet.enable_gradient_checkpointing()
    unet.train()
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")

    sched = DDPMScheduler.from_pretrained(a.base_model, subfolder="scheduler")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    out = C.ROOT / "outputs" / "lora" / a.name
    out.mkdir(parents=True, exist_ok=True)

    def save(tag=""):
        StableDiffusionPipeline.save_lora_weights(save_directory=str(out / tag) if tag else str(out),
                                                  unet_lora_layers=get_peft_model_state_dict(unet))

    log, running = [], []
    batches = itertools.cycle(dl)
    for step in tqdm(range(1, a.max_train_steps + 1), desc="LoRA"):
        imgs, labels = next(batches)
        with torch.no_grad():
            lat = vae.encode(imgs.to(dev, dtype=fp16)).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=fp16):
            pred = unet(noisy, t, text_emb[labels.to(dev)]).sample
        loss = F.mse_loss(pred.float(), noise.float())
        scaler.scale(loss / a.grad_accum).backward()
        running.append(loss.item())
        if step % a.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            log.append({"step": step, "loss": sum(running) / len(running)}); running = []
            tqdm.write(f"step {step} loss {log[-1]['loss']:.4f}")
        if a.save_every and step % a.save_every == 0 and step < a.max_train_steps:
            save(f"checkpoint-{step}")
    save()
    with open(out / "loss_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "loss"]); w.writeheader(); w.writerows(log)
    print("Saved LoRA to", out)


if __name__ == "__main__":
    main()
