# Face Recognition based Lock System

## Overview

This project implements a face recognition-based door lock system using Python and OpenCV. It allows authorized users to unlock a door by being recognized by the system. The system includes features for user management, face data capture, access logging, and system statistics.

## Features

- **Face Recognition:** Utilizes Local Binary Patterns Histograms (LBPH) for face recognition.
- **User Management:** Add, delete, and view registered users.
- **Face Data Capture:** Capture face samples for each user to train the recognition model.
- **Access Logging:** Logs all access attempts (successful and failed) with timestamps and user information.
- **System Statistics:** Provides insights into the number of users, face samples, and other system metrics.
- **Manual Lock/Unlock:** Allows manual control of the door lock.
- **Data Persistence:** Stores user data and face samples in pickle files for persistence.
- **Log Export:** Exports access logs to a CSV file for analysis.
- **Graceful Shutdown:** Saves data on program exit or interruption.

## Prerequisites

- **Python 3.6+**
- **OpenCV (cv2):** `pip install opencv-python`
- **OpenCV Contrib (cv2.face):** `pip install opencv-contrib-python` (required for face recognition algorithms)
- **NumPy:** `pip install numpy`

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd Face-Recognition
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt  # Create a requirements.txt file with the above dependencies
    ```

## Directory Structure

```
Face-Recognition/
├── main.py
├── operations/
│ ├── face_recognition.py
│ ├── data_management.py
│ └── lock_system.py
├── data/ # Auto-created
│ ├── users_database.pkl # User data
│ └── faces_data.pkl # Face recognition data
├── log_files/ # Auto-created
│ ├── access_log.json # Access logs
│ └── access_log_export.csv # Exported logs (optional)
└── README.md
```

## Usage

1.  **Run `main.py`:**

    ```bash
    python main.py
    ```

2.  **Follow the on-screen menu:**

    - **Add new user:** Enter the user's name and specify if they are an administrator.
    - **Capture face samples for user:** Select a user ID and follow the instructions to capture face samples. Ensure good lighting and clear visibility of the face.
    - **Start lock system:** The system will activate the camera and attempt to recognize faces.
    - **View users:** Display a list of registered users.
    - **View access logs:** Display the recent access attempts.
    - **Delete user:** Remove a user and their associated face data.
    - **Save data manually:** Save the current user and face data to files.
    - **View system statistics:** Display system statistics.
    - **Export logs to CSV:** Export access logs to a CSV file.
    - **Clean up orphaned data:** Remove face data for users who no longer exist.
    - **Exit:** Terminate the program.

## Configuration

- **Confidence Threshold:** The `confidence_threshold` parameter in `LockSystem` controls the minimum confidence level required for a face to be recognized. A lower threshold increases the risk of false positives, while a higher threshold increases the risk of false negatives. The default is 0.3 (30%).
- **Lock Timeout:** The `lock_timeout` parameter in `LockSystem` specifies the number of seconds the door remains unlocked after a successful recognition. The default is 10 seconds.
- **Face Sample Count:** The `num_samples` parameter in `capture_user_faces` determines the number of face samples to capture for each user. More samples generally improve recognition accuracy.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Author

**Abdul Basit Tonmoy**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
