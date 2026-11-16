# Credits and licences

Everything below was created by others and is used with attribution. Re-check each licence on its own page before submitting.

## Data
- **Interior Style Classification** (Roboflow Universe, workspace `roboflowworkplace`), CC BY 4.0. Retail-store interior photos, 5 style classes.
  https://universe.roboflow.com/roboflowworkplace/interior-style-classification

## Pretrained models (not trained by us)
- **Stable Diffusion 1.5** (base image generator). Licence: check the model page you downloaded it from (CreativeML Open RAIL-M family).
- **ControlNet MLSD v1.1** (`lllyasviel/control_v11p_sd15_mlsd`), licence listed on its page as openrail.
- **controlnet_aux** (MLSD line detector), **diffusers**, **transformers**, **peft**, **gradio**: open-source libraries, see their licences.
- **Helsinki-NLP/opus-mt-vi-en** (Vietnamese to English translation), Apache-2.0.
- **ResNet50 ImageNet weights** from torchvision (starting point of our classifier).

## What the team built
- Data pipeline: cleaning, near-duplicate grouping (pHash), leakage-safe grouped split.
- Style classifier: Small CNN baseline, Random Forest baseline, ResNet50 two-phase fine-tuning, evaluation, Grad-CAM.
- Design pipeline: ControlNet + Stable Diffusion 1.5 for empty-room design, optional LoRA we train, Gradio web app with Vietnamese chat.
- Evaluation: human ratings of generated designs, classifier-based check.

## Use of AI assistance
State here how AI tools (for example a chat assistant) were used in the project, as your course's academic integrity policy requires.
