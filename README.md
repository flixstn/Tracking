# Tracking application

This program tracks a region of interest in a video input stream

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Installation

Use `requirements.txt` and `get_models.sh` to install dependencies.

```bash
pip3 install -r requirements.txt
./get_models.sh
```

The application makes use of [Yolo](https://pjreddie.com/darknet/yolo/) along with the [config-file](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg), the networks [weights](https://pjreddie.com/media/files/yolov3.weights) and the [COCO dataset](http://cocodataset.org/#home)

### Running the application

Start the program and provide a video and mp3 file.
```bash
python ROI.py -i input_file -s sound_file
```

### Usage
Use the mouse to draw a box on the video frame, press enter and the application will run the detection process in this selected area.
It will play an alarm sound when specific objects are recognized
* The sound can be stopped with 'Q'
* The application can be closed with 'ESC'

### Possible test files
* Video: [Pixabay](https://pixabay.com/en/videos/scooters-traffic-street-motorcycle-5638/)
* Sound: [Freesound](https://freesound.org/people/israra/sounds/434055/)
