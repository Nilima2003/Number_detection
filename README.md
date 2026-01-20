# RTSP OCR Snapshot Monitor

A Python application that monitors RTSP camera streams, detects motion, extracts numbers using OCR, and logs detections.

## Features
- Real-time motion detection from multiple RTSP cameras
- OCR number extraction using Doctr
- Automatic snapshot capture on motion detection
- Gallery view for browsing captured images
- Logging system with search and export functionality

## Installation

### Method 1: Using pip
```bash
# Clone the repository
git clone <your-repo-url>
cd rtsp-ocr-monitor

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
