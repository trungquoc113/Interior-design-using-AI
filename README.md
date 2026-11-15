# Interior Style AI

Recognise interior design styles from interior photos (retail store interiors in the current dataset) using deep learning (PyTorch), with Grad-CAM explanations.
Final project for the Artificial Intelligence course (VNUK).

## Research Questions
1. How much does transfer learning (ResNet50 / EfficientNet) improve over a CNN trained from scratch?
2. How do data augmentation and image quality affect accuracy?
3. Which styles does the model confuse, and which image regions does it rely on (Grad-CAM)?

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Place images in `data/raw/<style_name>/*.jpg` (e.g. `modern`, `scandinavian`, `industrial`).
The dataset is not committed to Git. Record the source, licence and download link in `docs/DATA.md`.

## Running the Pipeline
```bash
python -m src.data.clean          # remove corrupt/small/duplicate images -> data/interim, log: reports/cleaning_log.csv
python -m src.data.split_grouped  # leakage-safe split: copies of one photo stay in the same split -> data/processed
python -m src.leakage_check       # optional: closest train image for each test image (reports/leakage_check.csv)
python -m src.show_pairs          # optional: side-by-side figure of the closest pairs
python -m src.baseline_rf         # secondary baseline: Random Forest on colour histograms
python -m src.train --model small_cnn --epochs 30 --tag small_cnn
python -m src.evaluate --tag small_cnn
# Week 4: transfer learning, phase 1 = train the new head only, phase 2 = fine-tune the last block
python -m src.train --model resnet50 --freeze --epochs 8 --finetune-epochs 8 --unfreeze-blocks 1 --class-weights --label-smoothing 0.1 --tag resnet50_finetuned
python -m src.evaluate --tag resnet50_finetuned
python -m src.sweep --model resnet50 --epochs 5   # optional hyperparameter search -> reports/sweep_results.csv
python -m src.compare                              # comparison table and charts
python -m src.gradcam --tag resnet50_finetuned --image path/to/room.jpg
```
If you do not have a GPU, run the project on Google Colab or Kaggle.

## Repository Structure
```
data/        raw, interim, processed (not committed)
notebooks/   EDA, preprocessing, baseline, models
src/         data/clean.py, data/split.py, data/split_grouped.py, leakage_check.py, show_pairs.py, dataset.py, models.py, train.py,
             evaluate.py, gradcam.py, baseline_rf.py
reports/     figures, cleaning_log.csv, metrics_*.json
app/         demo (Streamlit/Gradio), optional
docs/        DATA.md, notes, report
```

## Team Members and Responsibilities
| Member | Role |
|---|---|
| ... | ... |

## Results (fill in after running the experiments)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest + colour histogram | | |
| SmallCNN | | |
| ResNet50 | | |

## Design module (empty room to furnished room)
Uses pretrained Stable Diffusion 1.5 + ControlNet (MLSD). No generator is trained; the style classifier is used as a secondary checker.
```bash
pip install -r requirements-design.txt
# put 3-5 photos of EMPTY rooms in data/empty_rooms/
python -m src.design.generate --n 2                 # add --low-vram if GPU memory runs out
python -m src.design.classify_designs --tag resnet50_finetuned   # secondary check (out-of-domain)
python -m src.design.summarize_ratings rater1.csv rater2.csv      # human ratings, main evidence
```
