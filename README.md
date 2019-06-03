# Tracking application

This program tracks a region of interest in a video input stream

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install opencv-python
pip install python-vlc
pip install numpy
```

The application makes use of [Yolo](https://pjreddie.com/darknet/yolo/) along with the [config-file](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg), the networks [weights](https://pjreddie.com/media/files/yolov3.weights) and the [COCO dataset](http://cocodataset.org/#home)

### Running the application

Start the program with a video file __detection_input.avi__ and an mp3-sound __siren.mp3__.
```bash
python ROI.py
```

### Usage
Use the mouse to draw a box on the video frame, press enter and the application will run the detection process in this selected area.
It will play an alarm sound when specific objects are recognized
+ The sound can be stopped with 'Q'
+ The application can be closed with 'ESC'

### Licenses / Copyrights
Motorbike video origin: [Pixabay](https://pixabay.com/en/videos/scooters-traffic-street-motorcycle-5638/)<br>
Alarm sound origin: [Freesound](https://freesound.org/people/israra/sounds/434055/)
