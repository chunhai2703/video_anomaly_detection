## Cấu trúc chính

```text
video-anomaly-detection/
├── app.py
├── infer.py
├── train.py
├── outputs/
│   └── model.pth
├── src/
│   ├── models/
│   │   ├── autoencoder.py
│   │   └── predictor.py
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── ui/
│   │   └── dashboard.py
│   └── utils/
│       └── video.py
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy UI

```bash
streamlit run app.py
```

Trong UI:

- `Model path`: mặc định `outputs/model.pth`.
- `Frame stride`: 2 là cân bằng tốt. Với video 3 phút, có thể tăng 3–5 để nhanh hơn.
- `Batch size`: 64 nếu RAM/GPU ổn; giảm xuống 16–32 nếu máy yếu.
- `Threshold`: dùng Auto percentile trước, sau đó chỉnh manual nếu cần.

## Chạy CLI

```bash
python infer.py \
  --input path/to/video.mp4 \
  --model_path outputs/model.pth \
  --output_dir outputs/inference \
  --stride 2 \
  --batch_size 64 \
  --top_k 24 \
  --save_frames
```

Xuất video có overlay:

```bash
python infer.py --input path/to/video.mp4 --model_path outputs/model.pth --save_video --stride 2
```

## Output

- `outputs/inference/scores.csv`: score từng frame đã xử lý.
- `outputs/inference/summary.json`: thống kê tổng quan.
- `outputs/inference/top_frames/`: ảnh anomaly/top-score đã đánh dấu.
- `outputs/inference/annotated_output.mp4`: chỉ có khi bật `--save_video`.