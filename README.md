# Face-Recognition

## Overview

This project implements a basic face recognition lock system using OpenCV and the LBPHFaceRecognizer algorithm. The system aims to unlock a door automatically when a registered user's face is detected. It also includes features for adding users, capturing face samples, managing users, accessing logs, and locking/unlocking the door manually.

## Features

- **Face Recognition:** Uses LBPHFaceRecognizer to identify registered users based on their facial features.
- **User Management:** Allows adding new users with names and optional admin privileges.
- **Face Capture:** Enables capturing face samples for each user. Includes instructions for optimal capture conditions.
- **Lock/Unlock:** Automatically unlocks the door upon successful face recognition of a registered user. Manual lock/unlock options are available.
- **Access Logging:** Records access attempts (success/failure, confidence level) in a JSON log file.
- **Confidence Threshold:** Configurable confidence threshold for face recognition.
- **Lock Timeout:** Configurable timeout duration after successful unlock.
- **User Interface:** Provides a command-line interface for managing the system.
- **Error Handling:** Includes basic error handling for file operations, user input, and face recognition.

## Requirements

- **Python:** 3.6 or higher
- **OpenCV:** `pip install opencv-contrib-python` (Important! The `cv2.face` module requires the contrib version)
- **NumPy:** `pip install numpy`
- **pickle:** (Built-in, no installation needed)
- **json:** (Built-in, no installation needed)

## Installation

1.  **Install Dependencies:** Run the following command in your terminal:
    ```bash
    pip install opencv-contrib-python numpy
    ```
2.  **Save the Code:** Save the Python code as a `.py` file (e.g., `face_lock.py`).

## Usage

1.  **Run the Script:** Execute the script from your terminal:
    ```bash
    python face_lock.py
    ```
2.  **Follow the Menu:** The script will present a menu with the following options:
    - `1. Add new user`: Adds a new user to the system.
    - `2. Capture face samples for user`: Captures face samples for a registered user.
    - `3. Start lock system`: Starts the face recognition lock system (requires users and face data).
    - `4. View users`: Lists the registered users and the number of face samples captured for each.
    - `5. View access logs`: Displays the recent access attempts from the log file.
    - `6. Delete user`: Removes a registered user.
    - `7. Exit`: Exits the program.

## Configuration

- **`confidence_threshold`:** This value (default: 0.3) represents the minimum confidence level required for a face recognition match to be considered valid. Adjusting this value can impact the system's sensitivity and false positive rate. Lower values are more lenient, higher values are stricter.
- **`lock_timeout`:** This value (default: 10 seconds) specifies the duration the door remains unlocked after a successful face recognition.

These values are set within the `FaceRecognitionLock` class constructor. You can modify them in the script to customize the system's behavior.

## Important Notes

- **Camera Access:** Ensure the script has access to your camera. You might need to grant camera permissions in your operating system settings.
- **Lighting Conditions:** Good lighting conditions are crucial for accurate face recognition. Ensure the user's face is well-lit and clearly visible.
- **Facial Position:** Users should position their faces directly towards the camera and avoid excessive head movements.
- **Data Storage:** User data and face samples are stored in pickle files (`users_database.pkl` and `faces_data.pkl` respectively). Keep these files secure.
- **Security Considerations:** This is a basic example and is not intended for high-security applications. Consider adding more robust security measures for production environments.
- **Error Reporting:** The script provides basic error reporting. For production deployments, you should implement more comprehensive logging and error handling.

## Author

- **Abdul Basit Tonmoy**

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
