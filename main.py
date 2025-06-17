import cv2
import numpy as np
import os
import pickle
import time
import json
from datetime import datetime
import threading

class FaceRecognitionLock:
    def __init__(self, confidence_threshold=0.6, lock_timeout=5):
        """
        Initialize the face recognition lock system
        
        Args:
            confidence_threshold: Minimum confidence for face recognition (0-1) - now set to 0.6 (60%)
            lock_timeout: Time in seconds to keep lock open after recognition
        """
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Face recognition with optimized parameters
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,      # Default radius
            neighbors=8,   # Default neighbors  
            grid_x=8,      # Default grid_x
            grid_y=8,      # Default grid_y
            threshold=80.0 # Lower threshold for better recognition
        )
        
        # System parameters (set to 60% confidence threshold)
        self.confidence_threshold = confidence_threshold
        self.lock_timeout = lock_timeout
        
        # Data storage
        self.users_db_path = 'users_database.pkl'
        self.faces_data_path = 'faces_data.pkl'
        self.log_file = 'access_log.json'
        
        # User database
        self.users = {}  # {user_id: {'name': str, 'is_admin': bool, 'created': datetime}}
        self.faces_data = {'faces': [], 'labels': [], 'user_ids': []}
        
        # Lock state
        self.is_locked = True
        self.lock_timer = None
        self.current_user = None
        
        # Load existing data
        self.load_data()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("Face Recognition Lock System Initialized")
        print(f"Confidence threshold set to: {confidence_threshold*100}%")
        print(f"Registered users: {len(self.users)}")
    
    def load_data(self):
        """Load user database and face data from files"""
        try:
            if os.path.exists(self.users_db_path):
                with open(self.users_db_path, 'rb') as f:
                    self.users = pickle.load(f)
            
            if os.path.exists(self.faces_data_path):
                with open(self.faces_data_path, 'rb') as f:
                    self.faces_data = pickle.load(f)
                
                # Train recognizer if we have face data
                if len(self.faces_data['faces']) > 0:
                    faces_array = np.array(self.faces_data['faces'])
                    labels_array = np.array(self.faces_data['labels'])
                    self.face_recognizer.train(faces_array, labels_array)
                    print(f"Loaded {len(faces_array)} face samples for training")
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_data(self):
        """Save user database and face data to files"""
        try:
            with open(self.users_db_path, 'wb') as f:
                pickle.dump(self.users, f)
            
            with open(self.faces_data_path, 'wb') as f:
                pickle.dump(self.faces_data, f)
            
            print("Data saved successfully")
        
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def log_access(self, user_id, success, confidence=None):
        """Log access attempts"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': self.users.get(user_id, {}).get('name', 'Unknown'),
            'success': success,
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.1f}%" if confidence else None
        }
        
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        
        except Exception as e:
            print(f"Error logging access: {e}")
    
    def add_user(self, name, is_admin=False):
        """Add a new user to the system"""
        user_id = len(self.users) + 1
        self.users[user_id] = {
            'name': name,
            'is_admin': is_admin,
            'created': datetime.now().isoformat()
        }
        self.save_data()
        print(f"User '{name}' added with ID: {user_id}")
        return user_id
    
    def capture_user_faces(self, user_id, num_samples=100):
        """Capture face samples for a specific user with improved quality"""
        if user_id not in self.users:
            print(f"User ID {user_id} not found")
            return False
        
        user_name = self.users[user_id]['name']
        print(f"Capturing face samples for {user_name}")
        print("IMPORTANT INSTRUCTIONS:")
        print("1. Look directly at the camera")
        print("2. Keep your face well-lit and clearly visible")
        print("3. Slowly move your head left/right and up/down")
        print("4. Try different expressions (smile, neutral, etc.)")
        print("5. Ensure good lighting from front")
        print("Press SPACE to capture samples, 'q' to finish early")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        samples_captured = 0
        capture_mode = False
        
        while samples_captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better face detection
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(120, 120),  # Larger minimum size
                maxSize=(300, 300)   # Maximum size to avoid very large faces
            )
            
            for (x, y, w, h) in faces:
                # Only capture if in capture mode (SPACE pressed)
                if capture_mode and len(faces) == 1:  # Only one face visible
                    # Extract and preprocess face
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to consistent size
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Apply Gaussian blur to reduce noise
                    face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
                    
                    # Normalize lighting
                    face_roi = cv2.equalizeHist(face_roi)
                    
                    # Store face data
                    self.faces_data['faces'].append(face_roi)
                    self.faces_data['labels'].append(user_id)
                    self.faces_data['user_ids'].append(user_id)
                    
                    samples_captured += 1
                    capture_mode = False  # Reset capture mode
                    
                    print(f"Sample {samples_captured} captured!")
                
                # Visual feedback
                color = (0, 255, 0) if len(faces) == 1 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Face quality indicators
                face_roi = gray[y:y+h, x:x+w]
                brightness = np.mean(face_roi)
                cv2.putText(frame, f'Brightness: {brightness:.0f}', (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Instructions and status
            cv2.putText(frame, f'Capturing for: {user_name}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Samples: {samples_captured}/{num_samples}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status messages
            if len(faces) == 0:
                cv2.putText(frame, 'No face detected', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(frame, 'Multiple faces - ensure only you are visible', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'Press SPACE to capture sample', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Face Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to capture
                capture_mode = True
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Retrain the recognizer with better parameters
        if len(self.faces_data['faces']) > 0:
            faces_array = np.array(self.faces_data['faces'])
            labels_array = np.array(self.faces_data['labels'])
            
            print("Training face recognition model...")
            self.face_recognizer.train(faces_array, labels_array)
            print("Training completed!")
        
        self.save_data()
        print(f"Successfully captured {samples_captured} face samples for {user_name}")
        return True
    
    def recognize_face(self, face_roi):
        """Recognize a face and return user_id and confidence with improved preprocessing"""
        if len(self.faces_data['faces']) == 0:
            return None, 0
        
        try:
            # Preprocess face the same way as during training
            face_roi = cv2.resize(face_roi, (100, 100))
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
            face_roi = cv2.equalizeHist(face_roi)
            
            user_id, confidence = self.face_recognizer.predict(face_roi)
            
            # LBPH confidence: lower is better, convert to similarity percentage
            # Typical range: 0-100, where 0 is perfect match
            if confidence < 100:
                similarity = max(0, (100 - confidence) / 100.0)
            else:
                similarity = 0
            
            # Debug output
            print(f"Recognition result - User ID: {user_id}, Raw confidence: {confidence:.2f}, Similarity: {similarity:.3f} ({similarity*100:.1f}%)")
            
            return user_id, similarity
        
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, 0
    
    def unlock_door(self, user_id, confidence):
        """Unlock the door for a specific user"""
        if user_id in self.users:
            self.is_locked = False
            self.current_user = user_id
            user_name = self.users[user_id]['name']
            
            print(f"🔓 DOOR UNLOCKED for {user_name} (Confidence: {confidence*100:.1f}%)")
            
            # Log successful access
            self.log_access(user_id, True, confidence)
            
            # Set timer to automatically lock again
            if self.lock_timer:
                self.lock_timer.cancel()
            
            self.lock_timer = threading.Timer(self.lock_timeout, self.lock_door)
            self.lock_timer.start()
            
            return True
        return False
    
    def lock_door(self):
        """Lock the door"""
        self.is_locked = True
        self.current_user = None
        print("🔒 DOOR LOCKED")
    
    def manual_lock(self):
        """Manually lock the door"""
        if self.lock_timer:
            self.lock_timer.cancel()
        self.lock_door()
    
    def calculate_fps(self):
        """Calculate FPS for performance monitoring"""
        self.fps_counter += 1
        if self.fps_counter >= 10:
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_lock_interface(self, frame, faces):
        """Draw the lock interface overlay with proper colors"""
        height, width = frame.shape[:2]
        
        # Lock status with proper colors
        if self.is_locked:
            lock_color = (0, 0, 255)  # Red for locked
            lock_text = "🔒 LOCKED"
        else:
            lock_color = (0, 255, 0)  # Green for unlocked
            lock_text = "🔓 UNLOCKED"
        
        # Draw status box with proper colors
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (10, 10), (350, 120), lock_color, 3)  # Colored border
        
        cv2.putText(frame, lock_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, lock_color, 2)
        
        # Show confidence threshold
        cv2.putText(frame, f"Threshold: {self.confidence_threshold*100:.0f}%", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show current user if unlocked
        if not self.is_locked and self.current_user:
            user_name = self.users[self.current_user]['name']
            cv2.putText(frame, f"Welcome, {user_name}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # System info
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Users: {len(self.users)}', (width - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Faces: {len(faces)}', (width - 150, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'l' to lock manually, 'q' to quit", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_faces(self, frame, faces):
        """Process detected faces for recognition with improved logic"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve lighting consistency
        
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = 10
            y1 = max(0, y - padding)
            y2 = min(gray.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(gray.shape[1], x + w + padding)
            
            face_roi = gray[y1:y2, x1:x2]
            
            # Skip if face is too small or too large
            if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                continue
            
            # Recognize face
            user_id, confidence = self.recognize_face(face_roi)
            
            # Check face quality (brightness)
            brightness = np.mean(face_roi)
            
            # Determine if this is a valid recognition
            is_recognized = (user_id is not None and 
                           user_id in self.users and 
                           confidence >= self.confidence_threshold)
            
            if is_recognized:
                # Recognized user with sufficient confidence
                user_name = self.users[user_id]['name']
                color = (0, 255, 0)  # Green for recognized
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f'{user_name}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f'Conf: {confidence*100:.1f}%', (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, f'AUTHORIZED', (x, y+h+35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Unlock if door is locked
                if self.is_locked:
                    self.unlock_door(user_id, confidence)
            
            else:
                # Unknown face or confidence too low
                color = (0, 0, 255)  # Red for unknown/low confidence
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                if user_id and user_id in self.users:
                    user_name = self.users[user_id]['name']
                    cv2.putText(frame, f'{user_name}?', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Low: {confidence*100:.1f}%', (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, f'Need ≥{self.confidence_threshold*100:.0f}%', (x, y+h+35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Log failed attempt for known user
                    self.log_access(user_id, False, confidence)
                else:
                    cv2.putText(frame, 'Unknown', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Conf: {confidence*100:.1f}% if confidence else 0:.1f %', (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, 'ACCESS DENIED', (x, y+h+35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def run_lock_system(self):
        """Run the main face recognition lock system"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Face Recognition Lock System Active")
        print(f"Confidence threshold: {self.confidence_threshold*100}%")
        print("Press 'l' to lock manually, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 1.3, 5, minSize=(100, 100)
            )
            
            # Process faces for recognition
            frame = self.process_faces(frame, faces)
            
            # Draw interface
            frame = self.draw_lock_interface(frame, faces)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display frame
            cv2.imshow('Face Recognition Lock', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.manual_lock()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.lock_timer:
            self.lock_timer.cancel()

def setup_system():
    """Setup the face recognition lock system"""
    lock_system = FaceRecognitionLock(
        confidence_threshold=0.3,  # Set to 30% confidence threshold
        lock_timeout=10           # Seconds to keep door unlocked
    )
    
    while True:
        print("\n" + "="*50)
        print("FACE RECOGNITION LOCK SYSTEM - 60% CONFIDENCE")
        print("="*50)
        print("1. Add new user")
        print("2. Capture face samples for user")
        print("3. Start lock system")
        print("4. View users")
        print("5. View access logs")
        print("6. Delete user")
        print("7. Exit")
        
        choice = input("Select option (1-7): ")
        
        if choice == '1':
            name = input("Enter user name: ")
            is_admin = input("Is admin user? (y/n): ").lower() == 'y'
            user_id = lock_system.add_user(name, is_admin)
            print(f"User added with ID: {user_id}")
        
        elif choice == '2':
            lock_system.save_data()
            if not lock_system.users:
                print("No users found. Add a user first.")
                continue
            
            print("Available users:")
            for uid, user in lock_system.users.items():
                print(f"  {uid}: {user['name']}")
            
            try:
                user_id = int(input("Enter user ID to capture faces: "))
                if user_id in lock_system.users:
                    num_samples = int(input("Number of samples to capture (default 50): ") or "50")
                    lock_system.capture_user_faces(user_id, num_samples)
                else:
                    print("Invalid user ID")
            except ValueError:
                print("Invalid input")
        
        elif choice == '3':
            if not lock_system.users:
                print("No users registered. Add users first.")
                continue
            if len(lock_system.faces_data['faces']) == 0:
                print("No face samples captured. Capture faces first.")
                continue
            
            lock_system.run_lock_system()
        
        elif choice == '4':
            if lock_system.users:
                print("\nRegistered Users:")
                for uid, user in lock_system.users.items():
                    admin_status = " (Admin)" if user.get('is_admin', False) else ""
                    print(f"  {uid}: {user['name']}{admin_status}")
                    
                    # Count face samples
                    sample_count = lock_system.faces_data['user_ids'].count(uid)
                    print(f"      Face samples: {sample_count}")
            else:
                print("No users registered")
        
        elif choice == '5':
            try:
                if os.path.exists(lock_system.log_file):
                    with open(lock_system.log_file, 'r') as f:
                        logs = json.load(f)
                    
                    print(f"\nLast 10 access attempts:")
                    for log in logs[-10:]:
                        status = "✓" if log['success'] else "✗"
                        conf_display = log.get('confidence_percentage', 'N/A')
                        print(f"  {status} {log['timestamp']}: {log['user_name']} (confidence: {conf_display})")
                else:
                    print("No access logs found")
            except Exception as e:
                print(f"Error reading logs: {e}")
        
        elif choice == '6':
            if not lock_system.users:
                print("No users to delete")
                continue
            
            print("Users:")
            for uid, user in lock_system.users.items():
                print(f"  {uid}: {user['name']}")
            
            try:
                user_id = int(input("Enter user ID to delete: "))
                if user_id in lock_system.users:
                    user_name = lock_system.users[user_id]['name']
                    del lock_system.users[user_id]
                    
                    # Remove face data
                    indices_to_remove = [i for i, uid in enumerate(lock_system.faces_data['user_ids']) if uid == user_id]
                    for i in reversed(indices_to_remove):
                        del lock_system.faces_data['faces'][i]
                        del lock_system.faces_data['labels'][i]
                        del lock_system.faces_data['user_ids'][i]
                    
                    lock_system.save_data()
                    print(f"User '{user_name}' deleted successfully")
                else:
                    print("Invalid user ID")
            except ValueError:
                print("Invalid input")
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    # Install required package if not present
    try:
        import cv2
        # Check if face recognition module is available
        cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Error: opencv-contrib-python is required for face recognition")
        print("Install it with: pip install opencv-contrib-python")
        exit(1)
    except ImportError:
        print("Error: OpenCV is not installed")
        print("Install it with: pip install opencv-contrib-python")
        exit(1)
    
    setup_system()