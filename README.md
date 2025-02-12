# 🌴 Machine Learning Model for Detecting the Ripeness Level of Fresh Palm Fruit Bunches Using YOLOv8

## 📌 Overview
This project presents a machine learning model designed to detect the ripeness level of fresh palm fruit bunches using the YOLOv8 method. The model has been developed and tested using Google Colab, leveraging deep learning techniques for object detection.

🔗 **Published Journal Paper**: [Read the full research](https://ejournal.gunadarma.ac.id/index.php/jpp/article/view/11848)

## 🏋️️ Methodology
This study follows the **CRISP-DM** (Cross Industry Standard Process for Data Mining) methodology to ensure a structured and effective approach to machine learning model development.

## 📷 Sample Prediction
> Below is a comparison of images before and after being processed by the YOLOv8 model.

| **Before Detection** | **After Detection** |
|----------------------|----------------------|
| ![image](https://github.com/user-attachments/assets/dc912ff0-402e-4be7-9e74-0815515984bd) | ![image](https://github.com/user-attachments/assets/bbf9d74f-7691-4622-83da-2f816ddd91e6) |
| ![image](https://github.com/user-attachments/assets/e160d25e-0499-4d8d-8a7d-4a9bf0d830c1) | ![image](https://github.com/user-attachments/assets/3341eb6a-1a2d-4db8-a975-9eac0a05e4cc) |
| ![image](https://github.com/user-attachments/assets/4d844e1e-726d-488b-bc90-dc01af09fe31) | ![image](https://github.com/user-attachments/assets/b36c6d0a-1f68-4b6d-9680-c42839eaa38b) |

## 🛠 Technologies Used
- **YOLOv8** (You Only Look Once - Object Detection)
- **Google Colab** (Model Training and Testing)
- **Python** (Model Implementation)
- **OpenCV, NumPy, Pandas** (Image Processing & Data Handling)
- **Roboflow** (Dataset Preparation)

## 🚀 Getting Started
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Open Google Colab
Upload the notebook or use the provided Google Colab link to run the model.

### 3️⃣ Install Required Libraries
```python
!pip install ultralytics opencv-python numpy pandas roboflow
```

### 4️⃣ Run YOLOv8 Model
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load pre-trained model
results = model('sample_image.jpg')  # Perform inference
```

## 📊 Results & Performance
| Training Time | Precision | Recall | mAP |
|--------------|------------|--------|-----|
| 3.107 Hours  | 0.945      | 0.947  | 0.98 |

## 🌟 Citation
If you use this model in your research or projects, please cite our publication:
```
Your Name et al., "Machine Learning Model for Detecting the Ripeness Level of Fresh Palm Fruit Bunches Using YOLOv8," Journal Name, Year.
```
