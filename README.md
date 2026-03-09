# 👗 Fashion MNIST Augmented — Deep Learning Competition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Images](https://img.shields.io/badge/Train-180%2C000%20images-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> Clothing image classification on an augmented dataset using rotations —
> larger and more challenging than the original Fashion MNIST.

---

## 🎯 Goal

Build a **deep learning model** capable of correctly classifying
**30,000 clothing images** (including versions rotated at 45°, 90° and 270°)
into **10 categories**.

---

## 📦 Download the Data

| File | Size | Link |
|---|---|---|
| `train_images.zip` | ~122 MB | [⬇ Download](https://github.com/BwiraC/competition_fashion_mnist_augmented/releases/download/V1.0/train_images.zip) |
| `test_images.zip` | ~24 MB | [⬇ Download](https://github.com/BwiraC/competition_fashion_mnist_augmented/releases/download/V1.0/test_images.zip) |
| `train_labels_PUBLIC.csv` | ~6 MB | [⬇ Download](train_labels_PUBLIC.csv) |
| `sample_submission.csv` | ~60 KB | [⬇ Download](sample_submission.csv) |

> 💡 Replace the `#` links with your actual Kaggle / Hugging Face / Google Drive URLs.

---

## 📁 File Structure

```
competition/
│
├── train_images/               ← 180,000 PNG images (28×28, grayscale)
│   ├── train_000000_orig.png   ← original image
│   ├── train_000001_aug1.png   ← random rotation
│   ├── train_000002_aug2.png   ← random rotation
│   └── ...
│
├── test_images/                ← 30,000 PNG images to classify
│   ├── test_000000_orig.png
│   ├── test_000001_aug1.png
│   └── ...
│
├── train_labels_PUBLIC.csv     ← labels for training images
└── sample_submission.csv       ← expected submission format
```

---

## 🏷️ The 10 Classes

| Code | Class | Code | Class |
|:---:|---|:---:|---|
| 0 | T-shirt / top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

---

## 📊 Dataset Description

| Feature | Value |
|---|---|
| Training images | **180,000** |
| Test images | **30,000** |
| Format | PNG 28×28 pixels, grayscale |
| Classes | 10 |
| Augmentation | Rotations at 45°, 90°, 270° (randomly selected) |
| Original source | Fashion MNIST (60,000 train / 10,000 test) |

Each original image comes with **2 rotated versions** randomly chosen
from {45°, 90°, 270°}. The label is identical for all 3 versions.

---

## 📋 CSV File Formats

### `train_labels_PUBLIC.csv`
```
image_id, filename,              label, label_name
0,        train_000000_orig.png, 7,     Sneaker
1,        train_000001_aug1.png, 7,     Sneaker
2,        train_000002_aug2.png, 7,     Sneaker
3,        train_000003_orig.png, 3,     Dress
...
```

### `sample_submission.csv`
```
image_id, filename,             label
0,        test_000000_orig.png, 3
1,        test_000001_aug1.png, 7
2,        test_000002_aug2.png, 0
...
```

> ⚠️ Your submission must contain **exactly** the columns `image_id` and `label`.
> The `filename` column is optional but recommended.

---

## 📤 How to Submit

1. Train your model on the 180,000 images in `train_images/`
2. Predict the label (0–9) for each of the 30,000 images in `test_images/`
3. Create a CSV file in the following format:

```python
import pandas as pd

# Example with your predictions
sample = pd.read_csv('sample_submission.csv')
sample['label'] = your_predictions   # list or array of 30,000 integers (0-9)
sample.to_csv('my_submission.csv', index=False)
```

4. Submit `my_submission.csv`

---

## 📏 Evaluation Metric

Submissions are evaluated using **accuracy** (overall correctness):

```
score = number of correct labels / 30,000
```

A random guessing model would score around **10%**.
The reference score on the original Fashion MNIST is around **92%** —
the rotations make this dataset **more challenging**.

---

## 🚀 Baseline Example (PyTorch)

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

class FashionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(f"{self.img_dir}/{row['filename']}").convert('L')
        img   = np.array(img, dtype=np.float32) / 255.0
        img   = torch.tensor(img).unsqueeze(0)   # (1, 28, 28)
        label = int(row['label'])
        return img, label

# Loading
train_dataset = FashionDataset('train_labels_PUBLIC.csv', 'train_images')
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Simple CNN model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training (1 epoch example)
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss    = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

## 🔍 What Makes This Dataset Different from the Original

| Feature | Original Fashion MNIST | This Competition |
|---|---|---|
| Train set | 60,000 | **180,000** |
| Test set | 10,000 | **30,000** |
| Format | CSV pixel values | **PNG images** |
| Rotations | None | **45°, 90°, 270°** |
| Difficulty | Known score ~92% | **Unknown distribution** |

---

## 🛠️ Reproduce the Dataset

The full generation script is available in this repo:

```bash
git clone https://github.com/your-username/fashion-mnist-augmented.git
cd fashion-mnist-augmented
pip install tensorflow scipy pillow tqdm pandas numpy
python fashion_mnist_competition.py
```

---

## 📜 License

This project is based on [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
published by Zalando Research under the MIT license.
The augmentations and competition structure are original.

---

## ✉️ Contact

Questions or issues? Open an **Issue** on this GitHub repo.
