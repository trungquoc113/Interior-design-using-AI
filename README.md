# Interior Style AI




## Cài đặt
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dữ liệu
Đặt ảnh vào `data/raw/<ten_phong_cach>/*.jpg` (ví dụ `modern`, `scandinavian`, `industrial`).
Dataset không commit lên git; ghi nguồn, giấy phép và link tải trong `docs/DATA.md`.

## Chạy pipeline
```bash
python -m src.data.clean          # loại ảnh hỏng/nhỏ/trùng -> data/interim, log: reports/cleaning_log.csv
python -m src.data.split          # stratified 70/15/15 -> data/processed
python -m src.baseline_rf         # baseline phụ: Random Forest + histogram màu
python -m src.train --model small_cnn --epochs 30 --tag small_cnn
python -m src.evaluate --tag small_cnn
# Tuần 4: transfer learning
python -m src.train --model resnet50 --epochs 15 --freeze --class-weights --tag resnet50_frozen
python -m src.evaluate --tag resnet50_frozen
python -m src.gradcam --tag resnet50_frozen --image path/to/room.jpg
```
Có thể chạy trên Google Colab / Kaggle nếu máy không có GPU.

## Cấu trúc
```
data/        raw, interim, processed (không commit)
notebooks/   EDA, preprocessing, baseline, models
src/         data/clean.py, data/split.py, dataset.py, models.py, train.py,
             evaluate.py, gradcam.py, baseline_rf.py
reports/     figures, cleaning_log.csv, metrics_*.json
app/         demo (Streamlit/Gradio), tùy chọn
docs/        DATA.md, ghi chú, báo cáo
```


## Kết quả (điền sau khi chạy thật)
| Mô hình | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest + màu | | |
| SmallCNN | | |
| ResNet50 | | |
