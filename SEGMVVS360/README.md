# EPVS: Equirectangular Photometric Visual Servoing

Author: Nathan Crombez

### 1. Build
```bash
sudo apt install ros-noetic-visp ros-noetic-visp-bridge
cd ~/catkin_ws/src/
git clone https://github.com/NathanCrombez/DHPVS
cd ..
catkin_make
```

### 2. Instructions
* Remap the expected topics in the launch file for: 
  * Your 360 camera's images topic
  * Your 360 camera's info topic (only to get the image width and height)
  * Your robot arm topic to set velocities¹
  * Your robot arm topic to get pose¹
* Adapt the parameters:
  * gain: control law gain
  * imSizeFactor: image resizing factor
  * depth: unknown points depth used to compute the interaction matrix

¹ EPVS provides velocities expressed in the camera frame. The transformation to your robot's flange is up to you.


### 3. Usage
When all is set, run your 360 camera's driver and robot's controller (bridge), then
simply launch: 
```bash
roslaunch EPVS EPVS.launch
```
Follow the instructions that appear in the terminal.