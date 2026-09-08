# VirtualWeel 🚗

VirtualWeel is a Python computer vision application that converts hand gestures into driving controls (`W`, `A`, `S`, `D`) for games using OpenCV and MediaPipe.

## 🌟 Features
- **Gesture Steering**: Tilt your hands to turn left (`A`) and right (`D`) with a real-time steering wheel PNG overlay on your camera feed.
- **Gas & Brake Control**:
  - **Right Thumb Up** = `W` (Gas)
  - **Left Thumb Up** = `S` (Brake)
- **Camera Device Selector**: Supports built-in Mac webcams and mobile cameras (iPhone Continuity Camera).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python steering.py
```

## 🎮 Controls & Shortcuts
- **`ESC`**: Quit
- **`C`**: Switch camera device
- **`0`** / **`M`**: Select Mobile Camera
- **`1`**: Select Mac Camera

## 📜 License
MIT
