# Interior Style AI

Recognise interior design styles from room photos using deep learning (PyTorch), with Grad-CAM explanations.


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
python -m src.data.split          # stratified 70/15/15 split -> data/processed
python -m src.baseline_rf         # secondary baseline: Random Forest on colour histograms
python -m src.train --model small_cnn --epochs 30 --tag small_cnn
python -m src.evaluate --tag small_cnn
# Week 4: transfer learning
python -m src.train --model resnet50 --epochs 15 --freeze --class-weights --tag resnet50_frozen
python -m src.evaluate --tag resnet50_frozen
python -m src.gradcam --tag resnet50_frozen --image path/to/room.jpg
```
If you do not have a GPU, run the project on Google Colab or Kaggle.

## Repository Structure
```
data/        raw, interim, processed (not committed)
notebooks/   EDA, preprocessing, baseline, models
src/         data/clean.py, data/split.py, dataset.py, models.py, train.py,
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
