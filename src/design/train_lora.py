"""Train a LoRA on Stable Diffusion 1.5 from images grouped in folders (folder name = class).

The base model stays frozen; only small LoRA matrices in the attention layers are trained (a few MB).
Caption of every image comes from --caption ({cls} = folder name in lower case).

Style LoRA (default caption, folder = style, e.g. data/processed/train/Minimal/):
    python -m src.design.train_lora --max-train-steps 2000
Furniture LoRA (folders bed/, chair/, table/ ... inside data/furniture):
    python -m src.design.train_lora --data data/furniture --name furniture_lora --max-per-class 300 \
        --caption "a photo of a {cls}, product photo, white background" --max-train-steps 2000
Quick test: add --max-train-steps 50.  Out of GPU memory? add --resolution 448 (or 384).
Output: outputs/lora/<name>/
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

DEFAULT_CAPTION = "{cls} style interior design photograph"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def caption_for(cls, template=DEFAULT_CAPTION):
    return template.format(cls=cls.lower())


class StyleImages(Dataset):
    def __init__(self, root, size, max_per_class=0, seed=0):
        files = [p for p in sorted(Path(root).rglob("*")) if p.suffix.lower() in EXTS]
        by_class = {}
        for p in files:
            by_class.setdefault(p.parent.name, []).append(p)
        rng = random.Random(seed)
        self.items = []
        for c, ps in sorted(by_class.items()):
            if max_per_class and len(ps) > max_per_class:
                ps = rng.sample(ps, max_per_class)
            self.items += [(p, c) for p in ps]
        if not self.items:
            raise SystemExit(f"No images found under {root}")
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
    ap.add_argument("--caption", default=DEFAULT_CAPTION,
                    help='caption template, {cls} = folder name, e.g. "a photo of a {cls}, product photo, white background"')
    ap.add_argument("--max-per-class", type=int, default=0, help="use at most N images per folder (0 = all)")
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
    ds = StyleImages(a.data, a.resolution, a.max_per_class, a.seed)
    print(f"{len(ds)} images, classes: {ds.classes}")
    print("example caption:", caption_for(ds.classes[0], a.caption))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)

    tok = CLIPTokenizer.from_pretrained(a.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(a.base_model, subfolder="text_encoder").to(dev)
    ids = tok([caption_for(c, a.caption) for c in ds.classes], padding="max_length", max_length=tok.model_max_length,
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
