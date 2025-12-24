# Histopathology Gland Segmentation

This project implements a U-Net based segmentation model for identifying gland regions in histopathology images from the GLaS (MICCAI 2015) dataset.

## 🎯 Problem Statement

Perform pixel-level segmentation of gland/tissue regions from histopathology microscopy images using deep learning.

## 🧠 Model Architecture

- **U-Net**: Encoder-decoder architecture optimized for biomedical image segmentation
- **Loss**: Combined BCE + Dice Loss for better segmentation performance
- **Output**: Binary mask (gland vs background)

## 📊 Dataset

- **Source**: [GLaS MICCAI 2015 Gland Segmentation](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation)
- **Task**: Binary segmentation of glands
- **Images**: Histopathology microscopy images with corresponding masks

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch
- CUDA (optional)

### Installation

```bash
git clone <your-repo-url>
cd HistopathologySegmentation
pip install -r requirements.txt
```

### Training

```bash
cd src
python train.py
```

### Evaluation

```python
from src.evaluate import evaluate_model
# Load test data and evaluate
```

### Deployment

#### API (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### UI (Streamlit)

```bash
streamlit run ui/app.py
```

#### Docker

```bash
docker-compose up --build
```

## 📈 Metrics

- **Dice Coefficient**: Primary segmentation metric
- **IoU (Jaccard)**: Overlap quality measure
- **Pixel Accuracy**: Overall accuracy

## 📂 Project Structure

```
HistopathologySegmentation/
├── data/
│   └── GLaS/
├── models/
│   └── unet.pth
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License


This project is open-source.

## Sample Output 
Original Image:
<img width="901" height="778" alt="image" src="https://github.com/user-attachments/assets/88273483-8801-43c6-808f-b422c9c268ee" />

Segmentated Output:
<img width="848" height="452" alt="image" src="https://github.com/user-attachments/assets/6003d3fb-717e-4ff0-bb81-48048ace1a6f" />

[Histopathology Gland Segmentation Output.pdf](https://github.com/user-attachments/files/24323975/Histopathology.Gland.Segmentation.Output.pdf)
