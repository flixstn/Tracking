# Import library
import numpy as np
import cv2
import vlc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input for analyzing')
parser.add_argument('-s', '--sound', help='set alarm sound')

args = vars(parser.parse_args())

# Initialize video reading and the media player for the alarm sound
capInp = cv2.VideoCapture(args.get('input'))
sound_file = vlc.MediaPlayer(args.get('sound'))

# For writing the video to a file, opt-in the VideoWriter-object as well as
# 'capOut.write(frame)' and 'capOut.release()' below
'''
output_file = "detection_output.avi"
capOut = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
'''

# Initialize the parameters for the neural network and processing
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Initialize the parameters for the area that will be tracked
# Changing these parameters will change the area that will be tracked by the YOLO network
_, pre_frame = capInp.read()
region_frame = cv2.selectROI('Region of interest', pre_frame, showCrosshair=False)
roi_x1 = region_frame[0]
roi_y1 = region_frame[1]
roi_x2 = roi_x1 + region_frame[2]
roi_y2 = roi_y1 + region_frame[3]
cv2.destroyWindow('Region of interest')

# Give the configuration and weight files for the model and load the network with them.
yoloConfig = "yolov3.cfg"
yoloWeights = "yolov3.weights"

neuralNet = cv2.dnn.readNetFromDarknet(yoloConfig, yoloWeights)
neuralNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neuralNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Opt-Out
# Load names of classes
classes = "coco.names"
with open(classes, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Get the names of the output layers
def getOutputsNames(neuralNet):
    """
    Get the names of the output layers
    :type name: cv2.dnn.readNetFromDarknet Object
    :param name: Name of the neural network
    """
    
    # Get the names of all the layers in the network
    layersNames = neuralNet.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in neuralNet.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence score using non-maxima suppression
def postprocess(frame, outs):
    """
    Postprocessing with non-maxima suppression 
    :type name: cv2.VideoCapture Object
    :param name: Captured frame
    
    :type name: cv2.dnn.readNetFromDarknet Forward Method
    :param name: Output from the output layer
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])


    # Performs non maximum suppression given boxes and corresponding scores
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw the bounding box
        drawPrediction(classIds[i], confidences[i], left, top, left + width, top + height)


# Draw the predicted bounding box
def drawPrediction(classId, conf, left, top, right, bottom):
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
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        
    # Draw the bounding box according to the documentation:
    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(small_frame, (left, top), (right, bottom), (255, 0, 0), 2)

    # Display the label at the top left of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(small_frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    
    # Play a pre defined sound when certain objects are detected
    if classes[classId] == "person" or classes[classId] == "car" or classes[classId] == "motorbike":
        sound_file.play()

# Processing the video in a loop
while capInp.isOpened():

    # Read each frame of the video object
    frameBool, small_frame = capInp.read()
    _, roi = capInp.read()

    if frameBool:
        # Draw a rectangle on the video frame, only this area will be tracked
        cv2.rectangle(roi, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        small_frame = small_frame[roi_y1:roi_y2, roi_x1:roi_x2]


        # Create a 4D blob from the frame object.
        blob = cv2.dnn.blobFromImage(small_frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=True)

        # Sets the input to the network
        neuralNet.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = neuralNet.forward(getOutputsNames(neuralNet))
        # Remove the bounding boxes with low confidence
        postprocess(small_frame, outs)

        # Show the processed and tracked frame
        # capOut.write(frame)

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
capInp.release()
# capOut.release()
cv2.destroyAllWindows()
