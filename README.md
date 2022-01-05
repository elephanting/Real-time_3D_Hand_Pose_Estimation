# Real-time 3D Hand Pose Estimation
<img src="/assets/asset.gif" width=50% height=50%>
This is a demo project of my master thesis. A system estimates hand poses from single images in real-time (30FPS on GTX1060).

## Installation
### Install dependencies
```
pip install -r requirements.txt
```
```
cd manopth && pip install . && cd ..
```
```
cd yolov3 && pip install . && cd ..
```
### Prepare network models
Download models from [here](https://drive.google.com/file/d/1wLbBuZoCJGrXbwoPijAZOBG1A23xrVnk/view?usp=sharing) and unzip it in checkpoints folder. The folder structure should look like this:
```
checkpoints/
    detnet_demo.pth
    iknet_demo.pth
```

### Prepare MANO model
Download MANO model from [here](https://mano.is.tue.mpg.de/) and unzip it in mano folder. The folder structure should look like this:
```
mano/
    models/
      ...
    webuser/
      ...
```
## Demo
Demo for webcam input:
```
python demo.py
```
Demo for video input:
```
python demo.py --video video_path
```

## Acknowledgement
- Part of the system was modified from [minimal-hand](https://github.com/CalciferZh/minimal-hand).
- Code of Open3d rendering was adapted from [Minimal-Hand](https://github.com/lingtengqiu/Minimal-Hand).
