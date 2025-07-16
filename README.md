# CityScape Semantic Segmentation

This repository contains a complete semantic segmentation pipeline for the [Cityscapes Dataset](https://www.cityscapes-dataset.com/), using two popular deep learning architectures: **FCN8s** and **UNet**. The project includes dataset exploration, training with multiple loss functions, and final evaluation.

## 📂 Project Structure

The project is organized as four sequential Jupyter Notebooks:

| Notebook | Description |
|----------|-------------|
| `001 Dataset EDA.ipynb` | Perform exploratory data analysis (EDA) and generate a cleaned dataset file list. |
| `002 Train FCN8s.ipynb` | Train the FCN8s model using three different loss functions: Dice Loss, Cross Entropy Loss, and a Combined Loss. |
| `003 Train UNET.ipynb` | Train the UNet model with the same three loss functions as FCN8s. |
| `004 Test Inference.ipynb` | Run inference and evaluate the trained models on validation/test data. |

> ⚠️ **Note:** Please run the notebooks in sequence from `001` to `004`.

---

## 🏗️ Model Architectures

- **FCN8s**: Fully Convolutional Network with skip connections.
- **UNet**: Classic U-shaped architecture with encoder-decoder structure.

Each model is trained using:
- **Dice Loss**
- **Cross Entropy Loss**
- **Combined Loss** (Dice + Cross Entropy)

---

## 🧪 Inference

The final notebook (`004 Test Inference.ipynb`) loads the best models and performs evaluation on the test dataset. Metrics such as IoU and Dice Score are used to compare the models' performance.

---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt


