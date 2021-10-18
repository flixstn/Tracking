# Import library
import numpy as np
import cv2
import vlc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input for analyzing')
parser.add_argument('-s', '--sound', required=True, help='set alarm sound')
args = vars(parser.parse_args())

# Initialize video reading and the media player for the alarm sound
cap_inp = cv2.VideoCapture(args.get('input'))
sound_file = vlc.MediaPlayer(args.get('sound'))

# Initialize the parameters for the neural network and processing
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4   # Non-maximum suppression threshold
inp_width = 416       # Width of network's input image
inp_height = 416      # Height of network's input image

# Initialize the parameters for the area that will be tracked
# Changing these parameters will change the area that will be tracked by the YOLO network
_, pre_frame = cap_inp.read()
region_frame = cv2.selectROI('Region of interest', pre_frame, showCrosshair=False)
roi_x1 = region_frame[0]
roi_y1 = region_frame[1]
roi_x2 = roi_x1 + region_frame[2]
roi_y2 = roi_y1 + region_frame[3]
cv2.destroyWindow('Region of interest')

# Give the configuration and weight files for the model and load the network with them.
yolo_config = "yolov3.cfg"
yolo_weights = "yolov3.weights"

detection = "detection.txt"
with open(detection, 'rt') as f:
    detection = f.read().rstrip('\n').split('\n')

neural_net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load names of classes
classes = "coco.names"
with open(classes, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Get the names of the output layers
def getOutputsNames(neural_net):
    """
    Get the names of the output layers
    :type name: cv2.dnn.readNetFromDarknet Object
    :param name: Name of the neural network
    """
    
    # Get the names of all the layers in the network
    layers_names = neural_net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in neural_net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence score using non-maxima suppression
def postprocess(frame, outs):
    """
    Postprocessing with non-maxima suppression 
    :type name: cv2.VideoCapture Object
    :param name: Captured frame
    
    :type name: cv2.dnn.readNetFromDarknet Forward Method
    :param name: Output from the output layer
    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])


    # Performs non maximum suppression given boxes and corresponding scores
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw the bounding box
        drawPrediction(class_ids[i], confidences[i], left, top, left + width, top + height)


# Draw the predicted bounding box
def drawPrediction(class_id, conf, left, top, right, bottom):
    """
    :type name: integer
    :param name: Class ID 
    
    :type name:float
    :param name: Confidence score  
    
    :type name:float
    :param name: left fix point for bounding box
    
    :type name:float
    :param name: top fix point for bounding box
    
    :type name:float
    :param name: top fix point for bounding box
    
    :type name:float
    :param name: bottom fix point for bounding box
    """

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)
        
    # Draw the bounding box according to the documentation:
    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(small_frame, (left, top), (right, bottom), (255, 0, 0), 2)

    # Display the label at the top left of the bounding box
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.putText(small_frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    
    # Play a pre defined sound when certain objects are detected
    if classes[class_id] in detection:
        sound_file.play()

# Processing the video in a loop
while cap_inp.isOpened():

    # Read each frame of the video object
    ret_val, small_frame = cap_inp.read()
    _, roi = cap_inp.read()

    if ret_val:
        # Draw a rectangle on the video frame, only this area will be tracked
        cv2.rectangle(roi, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        small_frame = small_frame[roi_y1:roi_y2, roi_x1:roi_x2]


        # Create a 4D blob from the frame object.
        blob = cv2.dnn.blobFromImage(small_frame, 1 / 255, (inp_width, inp_height), [0, 0, 0], 1, crop=True)

        # Sets the input to the network
        neural_net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = neural_net.forward(getOutputsNames(neural_net))
        # Remove the bounding boxes with low confidence
        postprocess(small_frame, outs)

        # Show the processed and tracked frame
        cv2.imshow('ROI', roi)
        cv2.imshow('Video', small_frame)
        
        # Exit the loop when 'ESC' is pressed
        # Stop the sound when 'Q' is pressed
        k = cv2.waitKey(1)
        
        if k == ord('q'):
            sound_file.stop()
            
        elif k == 27:
            break
        
    else:
        print("Error while processing")
        break

# Release input stream after processing
cap_inp.release()
cv2.destroyAllWindows()
