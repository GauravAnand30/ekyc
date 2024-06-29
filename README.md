# ekyc

# eKYC Project

## Overview
The eKYC project is designed to streamline the Know Your Customer (KYC) process by leveraging various technologies for identity verification. This project includes features such as camera photo detection, Aadhar card upload and extraction, OTP validation, facial recognition, and hand gesture verification. It is built using Python and Streamlit.

## Features
- **Homepage**: Provides an overview of all the steps involved in the eKYC process.
- **Camera Photo Detection**: Allows users to capture their photo using a webcam.
- **Aadhar Card Upload and Extraction**: Enables users to upload their Aadhar card, and extracts necessary information using OCR (Optical Character Recognition).
- **OTP Validation**: Validates the Aadhar card through OTP verification.
- **Facial Recognition and Hand Gesture Verification**: Uses facial recognition and hand gestures for additional verification steps.

## Libraries
The project utilizes the following libraries:
- **Streamlit**: For building the web application.
- **OpenCV**: For camera and image processing.
- **Pillow**: For image processing.
- **Pytesseract**: For extracting text from images (OCR).
- **SQLite**: For database operations.
- **UADAI API**: For Aadhar verification.
- **Requests**: For handling HTTP requests.
- **Mediapipe**: For facial recognition and hand gesture detection.



**HOMEPAGE STEPS OF VERIFICATION**
![](https://github.com/GauravAnand30/ekyc/blob/main/screenshots/Screenshot%202024-06-29%20202015.png)
![](https://github.com/GauravAnand30/ekyc/blob/main/screenshots/Screenshot%202024-06-29%20202139.png)

**CAMERA PHOTO CAPTURE**
![](https://github.com/GauravAnand30/ekyc/blob/main/screenshots/Screenshot%202024-06-29%20202324.png)

**AADHAR CARD VALIDATION**
![](https://github.com/GauravAnand30/ekyc/blob/main/screenshots/Screenshot%202024-06-29%20202414.png)
**GESTURE VERIFICATION**
![](https://github.com/GauravAnand30/ekyc/blob/main/screenshots/Screenshot%202024-06-29%20202535.png)

## Installation
To get started with the eKYC project, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/GauravAnand30/ekyc.git
   cd ekyc-project

2. **Create the environment**
   python -m venv myenv
   myenv\Scripts\activate
   pip install streamlit

3. **Install all the dependicies**
    run : streamlit run main.py
