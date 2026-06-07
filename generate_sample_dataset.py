import os
import urllib.request
import cv2
import numpy as np
import face_recognition
import random
# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
# Notable public figures with high-resolution portraits
CELEBRITY_DATA = [
    {
        "name": "Barack Obama",
        "gender": "Male",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
    },
    {
        "name": "Donald Trump",
        "gender": "Male",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg"
    },
    {
        "name": "Scarlett Johansson",
        "gender": "Female",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/26/Scarlett_Johansson_by_Gage_Skidmore_2019.jpg"
    },
    {
        "name": "Taylor Swift",
        "gender": "Female",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/76/Taylor_Swift_2022_infobox.jpg"
    }
]
AUGMENT_COUNT_PER_PERSON = 25
def download_image(url, save_path):
    print(f"Downloading from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        print(f"Successfully downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download image: {e}")
        return False
def crop_face(image_path):
    # Load using face_recognition
    img = face_recognition.load_image_file(image_path)
    # Convert RGB to BGR for OpenCV processing
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Detect face location
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        print(f"No face detected in {image_path}. Using entire image.")
        return cv2.resize(bgr_img, (250, 250))
    
    # Get coordinates of first face
    top, right, bottom, left = face_locations[0]
    h, w, _ = bgr_img.shape
    
    # Add 20% padding
    pad_h = int((bottom - top) * 0.2)
    pad_w = int((right - left) * 0.2)
    
    y1 = max(0, top - pad_h)
    y2 = min(h, bottom + pad_h)
    x1 = max(0, left - pad_w)
    x2 = min(w, right + pad_w)
    
    cropped = bgr_img[y1:y2, x1:x2]
    return cv2.resize(cropped, (250, 250))
def augment_image(face_img):
    h, w = face_img.shape[:2]
    
    # 1. Random rotation (-12 to +12 degrees)
    angle = random.uniform(-12, 12)
    # 2. Random scale (0.9 to 1.1)
    scale = random.uniform(0.9, 1.1)
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 3. Random translation (-10 to +10 pixels)
    tx = random.uniform(-10, 10)
    ty = random.uniform(-10, 10)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply transformation
    aug_img = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 4. Random brightness (0.75 to 1.25)
    brightness = random.uniform(0.75, 1.25)
    aug_img = np.clip(aug_img * brightness, 0, 255).astype(np.uint8)
    
    # 5. Random horizontal flip (50% chance)
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        
    return aug_img
def main():
    print("--- Celebrity Face Dataset Generator ---")
    
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    total_images_generated = 0
    
    for person in CELEBRITY_DATA:
        name = person["name"]
        gender = person["gender"]
        url = person["url"]
        
        # Subfolder in dataset: Name_Gender (e.g. Barack Obama_Male)
        folder_name = f"{name}_{gender}"
        person_dir = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Download face
        temp_img_path = os.path.join(temp_dir, f"{name.lower().replace(' ', '_')}.jpg")
        success = download_image(url, temp_img_path)
        
        if not success:
            print(f"Skipping dataset generation for {name} due to download error.")
            continue
            
        try:
            # Crop face region
            face_img = crop_face(temp_img_path)
            
            # Save original cropped face as first image
            orig_save_path = os.path.join(person_dir, "original.jpg")
            cv2.imwrite(orig_save_path, face_img)
            total_images_generated += 1
            
            # Generate augmented versions
            for i in range(1, AUGMENT_COUNT_PER_PERSON):
                aug_img = augment_image(face_img)
                aug_save_path = os.path.join(person_dir, f"augmented_{i:02d}.jpg")
                cv2.imwrite(aug_save_path, aug_img)
                total_images_generated += 1
                
            print(f"Generated {AUGMENT_COUNT_PER_PERSON} images for {name} ({gender}) in dataset/{folder_name}")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    # Cleanup temp
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        
    print(f"\nDataset generation complete! Total training images generated: {total_images_generated}")
    if total_images_generated < 50:
        print("WARNING: Dataset size is less than 50. Please check errors or capture custom faces.")
    elif total_images_generated > 150:
        print("WARNING: Dataset size exceeds 150.")
    else:
        print("Dataset size is within the required 50-150 range.")
if __name__ == "__main__":
    main()