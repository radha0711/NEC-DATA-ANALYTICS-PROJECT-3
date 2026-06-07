import os
import argparse
import pickle
import face_recognition
import numpy as np
# Directory setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "src":
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Paths to models
NAME_MODEL_PATH = os.path.join(MODELS_DIR, "name_model.pkl")
NAME_ENCODER_PATH = os.path.join(MODELS_DIR, "name_encoder.pkl")
GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "gender_model.pkl")
GENDER_ENCODER_PATH = os.path.join(MODELS_DIR, "gender_encoder.pkl")
def load_models():
    # Verify all model files exist
    paths = [NAME_MODEL_PATH, NAME_ENCODER_PATH, GENDER_MODEL_PATH, GENDER_ENCODER_PATH]
    for path in paths:
        if not os.path.exists(path):
            print(f"Error: Required model file '{path}' not found.")
            print("Please ensure models are trained by running 'train.py' first.")
            return None
            
    # Load name classifier and encoder
    with open(NAME_MODEL_PATH, "rb") as f:
        name_classifier = pickle.load(f)
    with open(NAME_ENCODER_PATH, "rb") as f:
        name_encoder = pickle.load(f)
        
    # Load gender classifier and encoder
    with open(GENDER_MODEL_PATH, "rb") as f:
        gender_classifier = pickle.load(f)
    with open(GENDER_ENCODER_PATH, "rb") as f:
        gender_encoder = pickle.load(f)
        
    return name_classifier, name_encoder, gender_classifier, gender_encoder
def predict_image(image_path, confidence_threshold=0.5):
    models = load_models()
    if models is None:
        return
        
    name_classifier, name_encoder, gender_classifier, gender_encoder = models
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return
        
    print(f"\nProcessing image: {image_path}")
    try:
        # Load image
        img = face_recognition.load_image_file(image_path)
        
        # Get face locations and encodings
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        if not face_encodings:
            print("No faces detected in the image.")
            return
            
        print(f"Detected {len(face_encodings)} face(s). Running prediction...")
        
        for idx, (face_loc, face_enc) in enumerate(zip(face_locations, face_encodings)):
            face_enc = face_enc.reshape(1, -1)
            
            # Predict Name
            name_probs = name_classifier.predict_proba(face_enc)[0]
            best_name_idx = np.argmax(name_probs)
            name_conf = name_probs[best_name_idx]
            
            if name_conf >= confidence_threshold:
                pred_name = name_encoder.inverse_transform([best_name_idx])[0]
            else:
                pred_name = "Unknown"
                
            # Predict Gender
            gender_probs = gender_classifier.predict_proba(face_enc)[0]
            best_gender_idx = np.argmax(gender_probs)
            gender_conf = gender_probs[best_gender_idx]
            pred_gender = gender_encoder.inverse_transform([best_gender_idx])[0]
            
            # Get face bounding box coordinates
            top, right, bottom, left = face_loc
            
            print(f"\nFace #{idx + 1} at bounding box (Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}):")
            print(f"  - Predicted Name:   {pred_name} (Confidence: {name_conf * 100:.2f}%)")
            print(f"  - Predicted Gender: {pred_gender} (Confidence: {gender_conf * 100:.2f}%)")
            
    except Exception as e:
        print(f"Error processing image: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Name and Gender from a face image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold to identify a person. (Default: 0.5)")
    args = parser.parse_args()
    
    predict_image(args.image, args.threshold)