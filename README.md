# Semantic Equirectangular Visual Tracking in Lightweight 3D Building Reconstructions

A visual tracking system designed to align real-world **360° equirectangular images** with **synthetic views** rendered from **lightweight 3D building models**. It enables robust camera pose estimation and robot velocity control by combining **semantic segmentation**, **Gaussian Mixture** representations, and a **Virtual Visual Servoing** framework, with support for **frequency-domain acceleration** and **seamless equirectangular handling**.

This repository implements the system described in the paper:  
📄 _"Semantic Equirectangular Visual Tracking in Lightweight 3D Building Reconstructions"_ (ICRA 2026)

---

## 📚 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)

---

## ✨ Features

- 360° panoramic image-based localization using equirectangular projections
- Synthetic-to-real semantic alignment via Gaussian Mixture matching
- Fast Gaussian mixture computation using **Fourier transforms**
- Seamless Gaussian mixture computation for **wrap-around continuity**
- Robust to frame skipping and dynamic occlusions
- Robot-agnostic velocity commands in the **camera frame**
- ROS-integrated pipeline with C++/Python nodes

---

## 🛠️ Installation

### 1. Clone and build the workspace

```bash
git clone <your-repo-url> ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 2. Install dependencies
```bash
rosdep install --from-paths src --ignore-src -r -y
```
- System requirements (install separately if not handled by rosdep):
- ROS (Noetic or Melodic)
- OpenCV
- ViSP (Visual Servoing Platform)
- PyTorch (for GM computation and FFT)
- OneFormer (for semantic segmentation)
  
---

## 🚀 Usage

### 1. Launch your 360° camera and robot controller
Ensure:

- The 360° camera driver is publishing images
- Your robot controller is subscribing to velocity commands

### 2. Launch SEGMVVS-360:

```bash
roslaunch SEGMVVS360 SEGMVVS360.launch
```

### 3. (Optional) Launch Gaussian Mixture services:
```bash
roslaunch gaussian_mixture gaussian_mixture.launch
```

---

## 📁 Repository Structure

```bash

SEGMVVS-360/
├── SEGMVVS360/
│   ├── launch/
│   │   └── SEGMVVS360.launch
│   ├── nodes/
│   │   ├── SEGMVVS360.cpp
│   │   ├── real_image_publisher.py
│   │   └── building_segmentation.py
│   ├── srv/
│   │   └── building_segmentation.srv
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── README.md
│
└── gaussian_mixture/
    ├── launch/
    │   └── gaussian_mixture.launch
    ├── nodes/
    │   └── gaussian_mixture_server
    ├── srv/
    │   ├── ComputeGaussianMixture.srv
    │   ├── ComputeOrientedGaussianMixture.srv
    │   └── ComputePhotometricEllipticalGaussianMixture.srv
    ├── CMakeLists.txt
    └── package.xml

```






