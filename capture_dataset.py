import os
import cv2
import face_recognition
def main():
    print("--- Webcam Face Dataset Collector ---")
    print("This script will help you capture face photos using your webcam to add yourself or someone else to the dataset.")
    
    # Input Name
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
        
    # Input Gender
    gender = ""
    while gender not in ["Male", "Female"]:
        g_input = input("Enter gender (Male/Female): ").strip().capitalize()
        if g_input in ["Male", "Female"]:
            gender = g_input
        else:
            print("Invalid input. Please enter 'Male' or 'Female'.")
            
    # Create folder: dataset/Name_Gender/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"{name}_{gender}"
    save_dir = os.path.join(base_dir, "dataset", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Check how many images already exist
    existing_imgs = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    count = len(existing_imgs)
    print(f"Current images in dataset/{folder_name}: {count}")
    
    max_images = 30
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("\n--- Controls ---")
    print(" Press 'c' to capture a single face image")
    print(" Press 's' to start automatic capture (captures 30 frames rapidly)")
    print(" Press 'q' to quit")
    
    auto_capture = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # For display, copy frame and detect faces
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face location
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Draw bounding boxes
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
        # Display instructions on screen
        status_text = f"Captured: {count}/{max_images}"
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, "C: Capture | S: Auto-Capture | Q: Done", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Capture Dataset - Press Q to Quit", display_frame)
        
        # Determine if we should capture
        should_save = False
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            should_save = True
        elif key == ord('s'):
            auto_capture = True
            print("Starting automatic capture. Please look at the camera and change expressions/angles slightly...")
            
        if auto_capture and face_locations:
            should_save = True
            cv2.waitKey(200) # Wait 200ms between captures to give time for motion/expression changes
            
        if should_save:
            if not face_locations:
                print("No face detected! Adjust lighting/position.")
                if auto_capture:
                    auto_capture = False # Pause auto-capture if no face is found
            else:
                # Crop and save face
                top, right, bottom, left = face_locations[0]
                
                # Add 20% margin around face
                pad_h = int((bottom - top) * 0.2)
                pad_w = int((right - left) * 0.2)
                
                y1 = max(0, top - pad_h)
                y2 = min(h, bottom + pad_h)
                x1 = max(0, left - pad_w)
                x2 = min(w, right + pad_w)
                
                cropped_face = frame[y1:y2, x1:x2]
                
                if cropped_face.size > 0:
                    cropped_face = cv2.resize(cropped_face, (250, 250))
                    filename = f"face_{count:03d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, cropped_face)
                    count += 1
                    print(f"Saved: {filepath}")
                
                if count >= max_images:
                    print(f"Reached target of {max_images} images.")
                    auto_capture = False
                    break
                    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing. Total images in dataset/{folder_name}: {count}")
if __name__ == "__main__":
    main()