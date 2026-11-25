"""Web app: upload a photo of an EMPTY room, type what you want, press Send, get furnished designs.

Run from the repository root:   python -m app.app
Then open http://127.0.0.1:7860 in the browser. Models load once at start-up (the first run downloads them).

The prompt can be Vietnamese (with accents) or English. Vietnamese is translated with Helsinki-NLP/opus-mt-vi-en (CPU).
The prompt actually sent to the generator is shown under "Chi tiết" so you can check the translation.
"""
import argparse
import gradio as gr
import torch
from src.design.generate import STYLES, NEGATIVE, fit_size, load_mlsd, load_pipe, apply_lora

NO_STYLE = "(Không dùng phong cách mẫu)"
ROOMS = [("Theo mô tả của tôi", ""), ("Phòng khách", "living room"), ("Phòng ngủ", "bedroom"),
         ("Bếp", "kitchen"), ("Phòng ăn", "dining room")]
QUALITY = "photorealistic interior design photography, soft natural lighting, highly detailed, sharp focus"
EXTRA_NEGATIVE = ", cartoon, illustration, painting, oversaturated, distorted perspective, extra furniture legs"
MODELS = {}


class Translator:
    name = "Helsinki-NLP/opus-mt-vi-en"

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(self.name)
        self.model = MarianMTModel.from_pretrained(self.name).eval()

    def __call__(self, text):
        batch = self.tok([text], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.no_grad():
            out = self.model.generate(**batch, num_beams=4, max_new_tokens=160)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def needs_translation(text):
    """Vietnamese with accents contains non-ASCII letters; pure ASCII text is treated as English."""
    return any(ord(c) > 127 for c in text)


def build_prompt(user_prompt, style, room_type, lora_on=False, raw=False):
    user_prompt = user_prompt.strip().rstrip(".")
    if raw:
        return user_prompt
    parts = []
    if style != NO_STYLE:
        parts += ([f"{style.lower()} style interior design photograph"] if lora_on else []) + [STYLES[style]]
    if room_type:
        parts.append(room_type)
    parts += [user_prompt, QUALITY]
    return ", ".join(p for p in parts if p)


def on_send(room_img, user_prompt, style, room_type, n, seed, control_scale, change, steps, guidance, max_side,
            neg_text, lora_scale, raw, use_translation, progress=gr.Progress()):
    if room_img is None:
        raise gr.Error("Hãy tải ảnh phòng trống lên trước.")
    if not (user_prompt or "").strip():
        raise gr.Error("Hãy nhập mô tả thiết kế (ví dụ: phòng ngủ tối giản, tường trắng, giường gỗ sáng).")
    text = user_prompt.strip()
    if use_translation and needs_translation(text):
        progress(0, desc="Đang dịch mô tả sang tiếng Anh")
        text = MODELS["translate"](text)
    lora_on = bool(MODELS.get("lora")) and float(lora_scale) > 0 and not MODELS.get("no_trigger")
    if MODELS.get("lora"):
        MODELS["pipe"].set_adapters(["style"], adapter_weights=[float(lora_scale)])
    prompt = build_prompt(text, style, room_type, lora_on, raw)

    room = fit_size(room_img, int(max_side))
    control = MODELS["mlsd"](room).convert("RGB").resize(room.size)
    images = []
    for k in range(int(n)):
        progress(k / int(n), desc=f"Đang tạo ảnh {k + 1}/{int(n)}")
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + k)
        common = dict(negative_prompt=neg_text or NEGATIVE, guidance_scale=float(guidance),
                      controlnet_conditioning_scale=float(control_scale), generator=gen)
        if MODELS.get("i2i") and float(change) < 0.99:  # start from the original photo: keeps its colours and light
            images.append(MODELS["i2i"](prompt, image=room, control_image=control, strength=float(change),
                                        num_inference_steps=int(int(steps) / max(float(change), 0.3)), **common).images[0])
        else:
            images.append(MODELS["pipe"](prompt, image=control, num_inference_steps=int(steps), **common).images[0])
    return images, control, prompt


def build_ui(has_lora=False):
    with gr.Blocks(title="Thiết kế nội thất bằng AI") as demo:
        gr.Markdown("## Thiết kế nội thất bằng AI từ ảnh phòng trống\nTải ảnh phòng trống, nhập mô tả, bấm **Gửi**.")
        with gr.Row():
            with gr.Column(scale=1):
                room_img = gr.Image(label="Ảnh phòng trống", type="pil")
                user_prompt = gr.Textbox(label="Mô tả thiết kế", lines=3,
                                         placeholder="Ví dụ: phòng ngủ tối giản, tường trắng, giường gỗ sáng, rèm voan trắng")
                style = gr.Dropdown([NO_STYLE] + list(STYLES), value=NO_STYLE, label="Phong cách mẫu (tuỳ chọn)")
                send = gr.Button("Gửi", variant="primary")
                with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                    room_type = gr.Dropdown(ROOMS, value="", label="Loại phòng")
                    n = gr.Slider(1, 4, value=2, step=1, label="Số ảnh")
                    seed = gr.Number(value=42, precision=0, label="Seed (đổi để ra ảnh khác)")
                    change = gr.Slider(0.5, 1.0, value=0.85, step=0.05,
                                       label="Mức thay đổi so với ảnh gốc (thấp = giữ màu và ánh sáng gốc, 1 = vẽ lại hoàn toàn)")
                    control_scale = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Độ bám cấu trúc phòng")
                    steps = gr.Slider(20, 50, value=35, step=1, label="Số bước (nhiều = nét hơn, chậm hơn)")
                    guidance = gr.Slider(4, 12, value=8, step=0.5, label="Độ bám mô tả (guidance)")
                    max_side = gr.Dropdown([512, 576, 640], value=576, label="Kích thước ảnh (cạnh dài, px)")
                    neg_text = gr.Textbox(value=NEGATIVE + EXTRA_NEGATIVE, label="Điều không muốn xuất hiện", lines=2)
                    lora_scale = gr.Slider(0.0, 1.5, value=0.8, step=0.1, label="Cường độ LoRA (0 = tắt)", visible=has_lora)
                    raw = gr.Checkbox(value=False, label="Chỉ dùng đúng mô tả tôi nhập (không thêm gì)")
                    use_translation = gr.Checkbox(value=True, label="Dịch mô tả tiếng Việt có dấu sang tiếng Anh")
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Kết quả", columns=2, height=480)
                with gr.Accordion("Chi tiết", open=False):
                    prompt_view = gr.Textbox(label="Mô tả đã gửi cho mô hình", lines=5, interactive=False)
                    control_view = gr.Image(label="Đường nét cấu trúc phòng (ControlNet dùng)", interactive=False)
        inputs = [room_img, user_prompt, style, room_type, n, seed, control_scale, change, steps, guidance, max_side,
                  neg_text, lora_scale, raw, use_translation]
        send.click(on_send, inputs, [gallery, control_view, prompt_view])
        user_prompt.submit(on_send, inputs, [gallery, control_view, prompt_view])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_mlsd")
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--lora", default=None, help="folder with trained LoRA weights, e.g. outputs/lora/style_lora")
    ap.add_argument("--no-trigger", action="store_true", help="furniture LoRA: do not add the style trigger phrase")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="temporary public link through gradio, off by default")
    a = ap.parse_args()
    print("Loading models (first run downloads several GB)...")
    MODELS["translate"] = Translator()
    MODELS["mlsd"] = load_mlsd()
    MODELS["pipe"] = load_pipe(a.base_model, a.controlnet, a.low_vram)
    if a.lora:
        apply_lora(MODELS["pipe"], a.lora, 0.8)
        MODELS["lora"] = True
        MODELS["no_trigger"] = a.no_trigger
    try:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        MODELS["i2i"] = StableDiffusionControlNetImg2ImgPipeline.from_pipe(MODELS["pipe"])
    except Exception as e:  # the slider then falls back to the normal pipeline
        print("img2img pipeline not available:", e)
    print("Ready.")
    build_ui(bool(a.lora)).queue().launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
