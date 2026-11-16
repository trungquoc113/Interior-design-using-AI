"""Web app: upload a photo of an EMPTY room, describe the design in a chat (Vietnamese or English), get furnished designs.

Run from the repository root:   python -m app.app
Then open http://127.0.0.1:7860 in the browser. Models load once at start-up (the first run downloads them).

How the chat works: every message is translated to English (Helsinki-NLP/opus-mt-vi-en, runs on the CPU) and added to a
list of requirements. The button "Tao thiet ke" combines style + room type + all requirements into ONE prompt and
generates new images with Stable Diffusion + ControlNet (MLSD). The prompt actually used is shown under the images.
Note: the generator reads descriptions, it does not follow edit commands, so every run redraws the whole room.
"""
import argparse
import gradio as gr
import torch
from PIL import Image
from src.design.generate import STYLES, NEGATIVE, fit_size, load_mlsd, load_pipe, apply_lora

NO_STYLE = "(Không chọn phong cách)"
QUALITY = "photorealistic interior design photography, soft natural lighting, highly detailed, sharp focus"
EXTRA_NEGATIVE = ", cartoon, illustration, painting, oversaturated, distorted perspective, extra furniture legs"
ROOMS = [("Phòng khách", "living room"), ("Phòng ngủ", "bedroom"), ("Bếp", "kitchen"), ("Phòng ăn", "dining room")]
MODELS = {}


class Translator:
    name = "Helsinki-NLP/opus-mt-vi-en"

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(self.name)
        self.model = MarianMTModel.from_pretrained(self.name).eval()

    def __call__(self, text):
        batch = self.tok([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = self.model.generate(**batch, num_beams=4, max_new_tokens=96)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def build_prompt(style, room_type, requirements, lora_on=False, raw=False):
    reqs = [r.strip().rstrip(".") for r in requirements if r.strip()]
    if raw:  # only what the user typed
        return ", ".join(reqs)
    parts = []
    if style != NO_STYLE:
        parts += ([f"{style.lower()} style interior design photograph"] if lora_on else []) + [STYLES[style]]
    parts += [room_type] + reqs + [QUALITY]
    return ", ".join(p for p in parts if p)


def on_message(msg, history, reqs, use_translation):
    msg = (msg or "").strip()
    if not msg:
        return history, reqs, ""
    en = MODELS["translate"](msg) if use_translation else msg
    reqs = list(reqs) + [en]
    reply = f"Đã ghi nhận (gửi cho mô hình): **{en}**\n\nCác yêu cầu hiện tại: " + "; ".join(reqs)
    history = list(history) + [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
    return history, reqs, ""


def on_clear():
    return [], [], ""


def on_generate(room_img, style, room_type, reqs, n, seed, control_scale, lora_scale, raw, steps, guidance, max_side, neg_text, change, progress=gr.Progress()):
    if room_img is None:
        raise gr.Error("Hãy tải ảnh phòng trống lên trước.")
    if (raw or style == NO_STYLE) and not reqs:
        raise gr.Error("Hãy nhắn ít nhất một yêu cầu trong khung chat trước khi tạo.")
    room = fit_size(room_img, int(max_side))
    control = MODELS["mlsd"](room).convert("RGB").resize(room.size)
    lora_on = MODELS.get("lora") and float(lora_scale) > 0
    if MODELS.get("lora"):
        MODELS["pipe"].set_adapters(["style"], adapter_weights=[float(lora_scale)])
    prompt = build_prompt(style, room_type, reqs, lora_on, raw)
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
        gr.Markdown("## Thiết kế nội thất bằng AI từ ảnh phòng trống\n"
                    "1) Tải ảnh phòng trống. 2) Chọn phong cách. 3) Nhắn yêu cầu chi tiết. 4) Bấm **Tạo thiết kế**.")
        reqs = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                room_img = gr.Image(label="Ảnh phòng trống", type="pil")
                style = gr.Dropdown([NO_STYLE] + list(STYLES), value="Minimal", label="Phong cách mẫu")
                room_type = gr.Dropdown(ROOMS, value="living room", label="Loại phòng")
                raw = gr.Checkbox(value=False, label="Chỉ dùng đúng mô tả tôi nhập (bỏ phong cách và loại phòng mẫu)")
                with gr.Accordion("Tuỳ chọn nâng cao", open=False):
                    n = gr.Slider(1, 4, value=2, step=1, label="Số ảnh")
                    seed = gr.Number(value=42, precision=0, label="Seed (đổi để ra ảnh khác)")
                    control_scale = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Độ bám cấu trúc phòng")
                    lora_scale = gr.Slider(0.0, 1.5, value=0.8, step=0.1, label="Cường độ LoRA (0 = tắt)", visible=has_lora)
                    steps = gr.Slider(20, 50, value=35, step=1, label="Số bước (nhiều = nét hơn, chậm hơn)")
                    guidance = gr.Slider(4, 12, value=8, step=0.5, label="Độ bám mô tả (guidance)")
                    max_side = gr.Dropdown([512, 576, 640], value=576, label="Kích thước ảnh (cạnh dài, px)")
                    neg_text = gr.Textbox(value=NEGATIVE + EXTRA_NEGATIVE, label="Điều không muốn xuất hiện (negative prompt)", lines=2)
                    change = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Mức thay đổi so với ảnh gốc (thấp = giữ màu và ánh sáng gốc, 1 = vẽ lại hoàn toàn)")
                    use_translation = gr.Checkbox(value=True, label="Dịch tin nhắn tiếng Việt sang tiếng Anh")
            with gr.Column(scale=1):
                chat_kwargs = dict(label="Mô tả chi tiết (ví dụ: sofa màu xám, rèm trắng, sàn gỗ sáng)", height=340)
                try:  # Gradio 5 needs type="messages"; Gradio 6 removed the parameter
                    chat = gr.Chatbot(type="messages", **chat_kwargs)
                except TypeError:
                    chat = gr.Chatbot(**chat_kwargs)
                txt = gr.Textbox(placeholder="Nhập yêu cầu rồi nhấn Enter", show_label=False)
                with gr.Row():
                    send = gr.Button("Gửi yêu cầu")
                    clear = gr.Button("Xoá yêu cầu")
                go = gr.Button("Tạo thiết kế", variant="primary")
        gallery = gr.Gallery(label="Kết quả", columns=2, height=420)
        with gr.Row():
            control_view = gr.Image(label="Đường nét cấu trúc phòng (ControlNet dùng)", interactive=False)
            prompt_view = gr.Textbox(label="Mô tả đã gửi cho mô hình", lines=6, interactive=False)
        send.click(on_message, [txt, chat, reqs, use_translation], [chat, reqs, txt])
        txt.submit(on_message, [txt, chat, reqs, use_translation], [chat, reqs, txt])
        clear.click(on_clear, None, [chat, reqs, txt])
        go.click(on_generate, [room_img, style, room_type, reqs, n, seed, control_scale, lora_scale, raw, steps, guidance, max_side, neg_text, change], [gallery, control_view, prompt_view])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_mlsd")
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--lora", default=None, help="folder with trained LoRA weights, e.g. outputs/lora/style_lora")
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
    try:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline
        MODELS["i2i"] = StableDiffusionControlNetImg2ImgPipeline.from_pipe(MODELS["pipe"])
    except Exception as e:  # the slider then falls back to the normal pipeline
        print("img2img pipeline not available:", e)
    print("Ready.")
    build_ui(bool(a.lora)).queue().launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
