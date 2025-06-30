
# 🤖 Gesture-Controlled 3D Robot Simulation

This is a group project for the Computer Graphics course (KE-7112), focused on integrating computer vision with real-time 3D graphics. The system uses advanced hand tracking to control a virtual robot in a custom-rendered environment.

## 🧠 Project Overview

Our application leverages **MediaPipe** for real-time hand gesture tracking and **PyOpenGL** for rendering 3D objects. The current implementation allows intuitive human-computer interaction by mapping gestures to 3D object transformations. This serves as a foundation for controlling a humanoid robot in future iterations.

## 👥 Team & Roles

- **Htoo Thet Naung** – Project Lead, Vision Integration, System Architecture
- *(Add remaining team members and roles here)*

## ✨ Features

- 🎥 Real-time hand tracking using MediaPipe
- 🧊 3D object manipulation using PyOpenGL
- 🔧 Multiple interaction modes:
  - **Position Control** – via single-hand pinch
  - **Rotation Control** – via three-finger pinch
  - **Scale Control** – via two-handed pinch
- 💡 Designed for extensibility toward full humanoid robot simulation

## 🖥️ System Requirements

- Python 3.12+
- Webcam
- Compatible with macOS/Linux/Windows

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vision_control_graphic.git
   cd vision_control_graphic
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

To launch the application:
```bash
python main.py
```

## 🎮 Gesture Controls

| Gesture                              | Action            |
|--------------------------------------|-------------------|
| Single hand index finger pinch       |      |
| Three-finger pinch                   |        |
| Two-handed index finger pinch        |         |
| ESC key                              | Exit application  |

## 🔭 Roadmap

- [ ] Replace cube with custom humanoid robot model
- [ ] Expand gesture library (e.g., wave, swipe)
- [ ] Implement environment interaction (e.g., object picking)

## 📄 License

MIT License — see `LICENSE` file for details.