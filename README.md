# Autonomous Driving Experiment

![Image of Autonomous Driving Experiment](https://github.com/m-dayani/robo-ad/blob/main/results/participants.jpg?raw=true)

An experimental autonomous driving system written using (mostly) the Python programming language.

- Robot can navigate and move between street lanes autonomously (using the `DBoW` place recognition method)
- On-board detection of several objects, including a toy person, a car, street signs, and four states of a traffic light, using `SSD-Mobilenet-V2`
- Two operational modes: **Race** and **Urban**
- **Race**: Robot navigates between the lanes around the arena and stops if detects one of several objects (using visual detection) or other external obstacles (e.g. a foot, using an Ultrasonic sensor).
- **Urban**: Robot moves in a straight line and stops behind the pedestrian crossing if the red light is detected. It moves again with no red light until it reaches a certain distance from the stop sign at the end of the road and stops again.


## Dependencies (Requirements)

- Tested with Python 3.9 and 3.12
- numpy==1.26.4
- opencv-python==4.6.0.66
- scipy
- Pillow
- tflite-runtime==2.11.0
- picamera2
- pyserial
- pyyaml
- pygame
- pybind11
- matplotlib
- apriltag
- [DBoW2](https://github.com/dorian3d/DBoW2)


## Usage

### Server Robot (e.g. Raspberry Pi)

Run the autonomous driving system:

```bash
(py_env) ~/robo-ad/python/auto_driving $ python main.py
```

Run the recording pipeline (used for collecting datasets for object detection and DBoW VPR)

```bash
(py_env) ~/robo-ad/python/auto_driving $ python recording.py --video_cap_mode video
```

Create a DBoW2 database, vocabulary, and commands table

```bash
(py_env) ~/robo-ad/python/slam/vpr_dbow $ python create_voc_db_adv.py ../../config/AUTO_DRIVING.yaml /path/to/root/dbow/datasets
```

### Client Controller (Desktop)

```bash
(py_env) ~/robo-ad/python/man_ctrl $ python flight_ctrl_client.py --host_ip <Server IP>
```


