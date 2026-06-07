import os
import pickle
import face_recognition
import numpy as np
# Directory setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "src":
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
def main():
    print("--- Step 1: Feature Extraction (Data Preparation) ---")
    
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print(f"Error: Dataset directory '{DATASET_DIR}' is empty or does not exist.")
        print("Please run 'generate_sample_dataset.py' or 'capture_dataset.py' to generate some training images first.")
        return
        
    known_embeddings = []
    known_names = []
    known_genders = []
    
    # Iterate through each folder in dataset directory
    # Folders should be named as "Name_Gender" (e.g. "Barack Obama_Male")
    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    if not folders:
        print("No folders found in dataset directory. Ensure they are structured as Name_Gender.")
        return
        
    total_images_processed = 0
    total_faces_saved = 0
    
    for folder in folders:
        # Split folder name to get name and gender
        parts = folder.rsplit("_", 1)
        if len(parts) != 2:
            print(f"Warning: Skipping folder '{folder}' because it is not in the format 'Name_Gender'.")
            continue
            
        name, gender = parts
        gender = gender.capitalize()
        
        if gender not in ["Male", "Female"]:
            print(f"Warning: Invalid gender '{gender}' for folder '{folder}'. Must be Male or Female.")
            continue
            
        folder_path = os.path.join(DATASET_DIR, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nProcessing person: {name} | Gender: {gender} ({len(image_files)} images)")
        
        for img_name in image_files:
            total_images_processed += 1
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Load image
                image = face_recognition.load_image_file(img_path)
                
                # Get face encodings
                # Since images are already cropped to faces, hog model is extremely fast and accurate
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    # Store the first detected face embedding
                    known_embeddings.append(encodings[0])
                    known_names.append(name)
                    known_genders.append(gender)
                    total_faces_saved += 1
                    print(f"  [SUCCESS] {img_name}: Face embedding extracted.")
                else:
                    print(f"  [WARNING] {img_name}: No face detected. Skipping.")
            except Exception as e:
                print(f"  [ERROR] {img_name}: Failed to process. Error: {e}")
                
    # Save the extracted embeddings and labels
    if total_faces_saved > 0:
        data = {
            "embeddings": np.array(known_embeddings),
            "names": known_names,
            "genders": known_genders
        }
        
        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(data, f)
            
        print(f"\nSuccessfully stored {total_faces_saved} embeddings in '{PROCESSED_DATA_PATH}'")
        print(f"Total images attempted: {total_images_processed}")
    else:
        print("\nError: No face embeddings could be extracted. Please check your training images.")
if __name__ == "__main__":
    main()