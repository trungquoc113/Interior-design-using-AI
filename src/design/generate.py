"""Design a furnished room from a photo of an EMPTY room.

Method: Stable Diffusion 1.5 + ControlNet (MLSD, straight-line map). The line map keeps the structure of the
room (walls, doors, windows, ceiling); the text prompt sets the interior style. An optional LoRA can be applied.

Put photos of empty rooms in data/empty_rooms/, then:
    python -m src.design.generate --n 2
    python -m src.design.generate --n 3 --low-vram        # if you run out of GPU memory
    python -m src.design.generate --lora outputs/lora/style_lora --lora-scale 0.8
    python -m src.design.generate --lora outputs/lora/furniture_lora --no-trigger
Outputs: outputs/designs/<room>/<style>_<k>.png (folder gets the suffix _lora with --lora),
         outputs/designs/manifest.csv (manifest_lora.csv with --lora), reports/figures/design_<room>.png
"""
import argparse, csv
from pathlib import Path
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C

STYLES = {
    "Darkmodern": "dark modern interior, charcoal walls, black furniture, warm accent lighting",
    "Luxuryclassic": "luxury classic interior, ornate furniture, chandelier, marble floor, gold accents",
    "Minimal": "minimalist interior, white walls, sparse furniture, clean lines, natural light",
    "Popretro": "pop retro interior, bold red and teal colours, 1970s furniture, playful decor",
    "Woodnatural": "natural wood interior, wooden furniture, warm tones, indoor plants, wood panelling",
}
NEGATIVE = "empty room, blurry, distorted, deformed furniture, text, watermark, low quality"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def fit_size(img, max_side=512):
    w, h = img.size
    s = max_side / max(w, h)
    return img.convert("RGB").resize((max(8, int(round(w * s / 8)) * 8), max(8, int(round(h * s / 8)) * 8)), Image.LANCZOS)


def load_mlsd():
    from controlnet_aux import MLSDdetector
    err = None
    for repo in ("lllyasviel/Annotators", "lllyasviel/ControlNet"):  # name depends on the controlnet_aux version
        try:
            return MLSDdetector.from_pretrained(repo)
        except Exception as e:
            err = e
    raise err


def load_pipe(base_model, controlnet_id, low_vram=False):
    """Load Stable Diffusion 1.5 + ControlNet once. Shared by generate.py and app/app.py."""
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if cuda:
        pipe.enable_model_cpu_offload() if low_vram else pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def apply_lora(pipe, lora_dir, scale):
    pipe.load_lora_weights(lora_dir, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[scale])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(C.ROOT / "data" / "empty_rooms"))
    ap.add_argument("--room-type", default="living room")
    ap.add_argument("--styles", nargs="*", default=list(STYLES))
    ap.add_argument("--n", type=int, default=2, help="images per style per room")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5",
                    help="if this download fails, search Hugging Face for a Stable Diffusion 1.5 mirror")
    ap.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_mlsd")
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--lora", default=None, help="folder with trained LoRA weights, e.g. outputs/lora/style_lora")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--no-trigger", action="store_true", help="do not add the style trigger phrase (use for a furniture LoRA)")
    a = ap.parse_args()

    pipe = load_pipe(a.base_model, a.controlnet, a.low_vram)
    if a.lora:
        apply_lora(pipe, a.lora, a.lora_scale)
    mlsd = load_mlsd()

    out_root = C.ROOT / "outputs" / "designs"
    out_root.mkdir(parents=True, exist_ok=True)
    rooms = sorted(p for p in Path(a.input).iterdir() if p.suffix.lower() in EXTS)
    if not rooms:
        raise SystemExit(f"No images found in {a.input}")
    rows = []
    for room_path in rooms:
        room = fit_size(Image.open(room_path))
        control = mlsd(room).convert("RGB").resize(room.size)
        out_dir = out_root / (room_path.stem + ("_lora" if a.lora else ""))
        out_dir.mkdir(exist_ok=True)
        control.save(out_dir / "control_lines.png")
        firsts = {}
        for style in a.styles:
            trigger = f"{style.lower()} style interior design photograph, " if (a.lora and not a.no_trigger) else ""
            prompt = f"{trigger}{STYLES[style]}, {a.room_type}, interior design photography, high quality"
            for k in range(a.n):
                seed = a.seed + k
                gen = torch.Generator(device="cpu").manual_seed(seed)
                img = pipe(prompt, negative_prompt=NEGATIVE, image=control, num_inference_steps=a.steps,
                           guidance_scale=a.guidance, controlnet_conditioning_scale=a.control_scale,
                           generator=gen).images[0]
                f = out_dir / f"{style}_{k}.png"
                img.save(f)
                firsts.setdefault(style, img)
                rows.append({"image": str(f), "room": room_path.stem, "style": style, "seed": seed,
                             "prompt": prompt, "style_match_1to5": "", "realism_1to5": "", "layout_kept_1to5": ""})
                print("saved", f)
        fig, axs = plt.subplots(1, 1 + len(firsts), figsize=(3.6 * (1 + len(firsts)), 3.8))
        for ax, (title, im) in zip(axs, [("Empty room", room)] + list(firsts.items())):
            ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis("off")
        plt.tight_layout()
        C.FIGS.mkdir(parents=True, exist_ok=True)
        plt.savefig(C.FIGS / f"design_{room_path.stem}{'_lora' if a.lora else ''}.png", dpi=110); plt.close(fig)

    manifest = out_root / ("manifest_lora.csv" if a.lora else "manifest.csv")
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("Manifest:", manifest, "(each rater fills the three *_1to5 columns)")


if __name__ == "__main__":
    main()
