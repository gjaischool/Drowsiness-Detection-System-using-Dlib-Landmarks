
# Drowsiness Detection Project Setup

## Files to Include

### 1. `README.md`

#### Content for README.md

```markdown
# Drowsiness Detection System using Dlib Landmarks

This project is a preliminary test implementation for detecting drowsiness using eye aspect ratio (EAR) values derived from facial landmarks. The implementation is part of a larger effort to develop a mobile app for real-time drowsiness detection. 

## Overview

The system uses a webcam to capture real-time video and processes the frames using `dlib`'s 68-point facial landmark predictor to detect eyes and calculate the eye aspect ratio (EAR). The EAR is used to determine if the user is drowsy by comparing it against a predefined threshold.

**Technologies Used:**
- Python
- OpenCV
- Dlib
- Matplotlib

## How It Works
- Detects the user's face using `dlib`'s frontal face detector.
- Extracts facial landmarks, particularly the eyes, using a pre-trained `shape_predictor_68_face_landmarks.dat` model.
- Computes the eye aspect ratio (EAR) to determine whether the eyes are closed for a prolonged period, indicating drowsiness.
- Displays real-time video using `Matplotlib` with overlaid markers to indicate drowsiness.

## Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. You'll need the following libraries:

- `dlib` - for facial landmark detection
- `opencv-python` (`cv2`) - for video stream handling
- `scipy` - for spatial distance calculation
- `matplotlib` - for displaying video in real time

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd drowsiness-detection-system
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's Model Zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the root of the project folder.

### Usage

Run the following command to start the detection:

```bash
python drowsiness_detection.py
```

Press `q` to quit the webcam feed.

### Note

- Make sure your webcam is connected and accessible.
- The default threshold values are set based on standard averages; you may need to tweak them for better results depending on individual differences.

## Future Work

This project serves as a test implementation and will be integrated with an Android mobile application to improve driver safety by alerting users about their drowsiness state.
```
