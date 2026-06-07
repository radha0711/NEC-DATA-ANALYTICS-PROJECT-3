# Gender & Face Detection System
This project is a complete pipeline for detecting the **Name** and **Gender** of a person in an image or real-time camera feed. It uses OpenCV, Python's `face_recognition` library, and `scikit-learn` to extract features and train classification models.
---
## Directory Structure
```text
gender_face_detection/
├── dataset/                    # Stores training images (minimum 50, maximum 150)
│   ├── Barack Obama_Male/
│   ├── Taylor Swift_Female/
│   └── ...
├── models/                     # Trained SVM classifiers and label encoders
├── data/                       # Pickle file containing extracted face embeddings
├── src/
│   ├── __init__.py
│   ├── data_prep.py            # Step 1: Extract 128D face encodings & save them
│   ├── train.py                # Step 2: Train SVM classifiers for Name and Gender
│   ├── predict.py              # Step 3: Classify a static test image
│   └── webcam_predict.py       # Output Step: Run live webcam prediction
├── capture_dataset.py          # Helper tool to capture your own face via webcam
├── generate_sample_dataset.py  # Helper tool to download and augment celebrity images
└── requirements.txt            # Package dependencies
```
---
## Prerequisites & Installation
Ensure you have Python 3.8+ installed. You can install all dependencies by running:
```bash
pip install -r requirements.txt
```
*Note: The `face_recognition` package requires a C++ compiler and `cmake` to compile `dlib` on Windows. If you run into issues installing `face_recognition`, make sure you have Visual Studio C++ build tools installed.*
---
## How to Run the Project
### Phase 1: Setup Dataset (50 - 150 images)
You need training images organized in directories under `dataset/` named as `Name_Gender` (e.g., `dataset/John Doe_Male/`).
You have two options to easily create this dataset:
#### Option A: Run automated script (Recommended)
This script downloads 4 celebrity portraits and applies data augmentation (rotation, translation, brightness scaling) to generate a balanced dataset of 100 images (25 images per person):
```bash
python generate_sample_dataset.py
```
#### Option B: Capture your own face via Webcam
This script opens your webcam and captures up to 30 frames of your own face, saving and cropping them to `dataset/Your Name_Your Gender/` folder:
```bash
python capture_dataset.py
```
---
### Phase 2: Feature Extraction (Step 1)
Run the following script to scan the `dataset/` folder, compute the 128-dimensional face encoding vector for each image, and save the embeddings to `data/processed_data.pkl`:
```bash
python src/data_prep.py
```
---
### Phase 3: Train Machine Learning Models (Step 2)
Run the script to load the embeddings and train Support Vector Machine (SVM) classifiers for name and gender classification:
```bash
python src/train.py
```
This script will split the data, evaluate model performance, print the accuracy metrics, and save the models to the `models/` directory.
---
### Phase 4: Make Predictions
#### Predict on a Single Test Image (Step 3)
To run prediction on any specific local image, pass its path to the prediction script:
```bash
python src/predict.py --image path/to/your/image.jpg
```
#### Real-Time Camera Inference (Output Step)
To run prediction on a live camera stream, use the following output command:
```bash
python src/webcam_predict.py
```
* Bounding boxes will outline detected faces.
* Above/below the bounding boxes, the predicted **Name (Gender)** and **confidence level** will be drawn.
* Press `q` while focused on the camera window to close the stream.