import os
import cv2
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
            
    # Load models
    with open(NAME_MODEL_PATH, "rb") as f:
        name_classifier = pickle.load(f)
    with open(NAME_ENCODER_PATH, "rb") as f:
        name_encoder = pickle.load(f)
    with open(GENDER_MODEL_PATH, "rb") as f:
        gender_classifier = pickle.load(f)
    with open(GENDER_ENCODER_PATH, "rb") as f:
        gender_encoder = pickle.load(f)
        
    return name_classifier, name_encoder, gender_classifier, gender_encoder
def main():
    print("--- Output Step: Real-Time Webcam Inference ---")
    
    models = load_models()
    if models is None:
        return
        
    name_classifier, name_encoder, gender_classifier, gender_encoder = models
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam. Make sure your camera is connected.")
        return
        
    print("Webcam started. Press 'q' inside the webcam window to exit.")
    
    # Hyperparameters
    confidence_threshold = 0.55
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
            
        # Resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB (required by face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for face_loc, face_enc in zip(face_locations, face_encodings):
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
            
            # Scale coordinates back up (since we processed 1/4 size image)
            top, right, bottom, left = face_loc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Choose border color based on predicted gender
            # Soft blue for male, soft pink/purple for female, grey for unknown
            if pred_name == "Unknown":
                color = (128, 128, 128) # Grey
            elif pred_gender == "Female":
                color = (180, 105, 255) # Pink/Purple
            else:
                color = (255, 144, 30) # Blue
                
            # Draw bounding box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Label string formatting
            label = f"{pred_name} ({pred_gender})"
            if pred_name != "Unknown":
                label += f" {name_conf * 100:.0f}%"
                
            # Draw label box below/above the face
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, 
                label, 
                (left + 6, bottom - 8), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.55, 
                (255, 255, 255), 
                1
            )
            
        # Display the result frame
        cv2.imshow("Gender & Face Detection - Press Q to Quit", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference session closed successfully.")
if __name__ == "__main__":
    main()