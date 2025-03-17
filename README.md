# Lane Detection System

## Overview
This project implements a real-time lane detection system using **OpenCV** and **computer vision techniques**. The system processes video frames, applies various image processing techniques, and extracts lane boundaries. The communication between the video source and processing script is handled via **socket communication**.

## Features
- **Video Frame Processing**: Reads frames from a video stream.
- **Rescaling**: Reduces frame size for faster processing.
- **Grayscale Conversion**: Converts frames to grayscale.
- **Region of Interest Selection**: Masks the road area to focus processing.
- **Top-Down Transformation**: Converts perspective to a bird's-eye view.
- **Edge Detection**: Uses the Sobel filter for lane marking detection.
- **Binarization**: Converts images to binary for easier analysis.
- **Lane Line Detection**: Fits lane boundary lines using polynomial regression.
- **Final Visualization**: Draws detected lanes onto the original frame.
- **Socket Communication**: Uses a custom socket-based system to send and receive frames.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install opencv-python numpy object_socket
```

## Usage
### Running the Lane Detection
1. **Start the sender script** to transmit video frames:
```bash
python producer.py
```
2. **Start the receiver (lane detection) script**:
```bash
python consumer.py
```
3. The program will continuously process video frames and display the results.
4. Press `q` to exit.

### Expected Output
The script will display multiple stages of image processing:
- **Small Frame**: Rescaled version of the input frame.
- **Grayscale**: Black and white representation of the frame.
- **Road Mask**: Highlights the selected road region.
- **Top-Down View**: A transformed perspective for better processing.
- **Edge Detection**: Sobel filter applied to extract edges.
- **Lane Detection**: Overlay of detected lane lines on the frame.

## Socket Communication
The project includes a **custom socket-based framework** to handle frame transmission:
- `sender.py` reads frames from a video file and sends them to a receiver over TCP.
- `lane_detection.py` receives frames and processes them for lane detection.
- `object_socket.py` contains the `ObjectSenderSocket` and `ObjectReceiverSocket` classes for efficient object transmission.

### Configuring the Socket Connection
Modify the `ObjectSenderSocket` and `ObjectReceiverSocket` parameters to match your network setup:
```python
s = object_socket.ObjectSenderSocket('127.0.0.1', 5000, print_when_awaiting_receiver=True, print_when_sending_object=True)
```
```python
s = object_socket.ObjectReceiverSocket('127.0.0.1', 5000, print_when_connecting_to_sender=True, print_when_receiving_object=True)
```
Ensure both scripts are using the **same IP and port** for communication.

## Notes
- Ensure your video source is compatible with OpenCV.
- The lane detection accuracy depends on road conditions and lighting.
- You can fine-tune the **threshold values** and **Sobel filter parameters** for better results.

## Author
**Octavian Cojocariu**

