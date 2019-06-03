## ROI Documentation

__getOutputNames__(*neuralNet*) <br>
&nbsp;&nbsp;&nbsp;&nbsp;Get the names of the output layer<br>

&nbsp;&nbsp;&nbsp;&nbsp;__Parameters:__ __neuralNet__(cv2.dnn.readFromDarknet)

__postprocess__(*frame,outs*) <br>
&nbsp;&nbsp;&nbsp;&nbsp; Postprocessing with non-maxima suppression <br>

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;__Parameters:__ 
+ __frame__(cv2.VideoCapture)
+ __outs__(cv2.dnn.readNetFromDarknet.forward)

__drawPrediction__(*classId, conf, left, top, right, bottom*)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Draw the predicted bounding box and play alarm sound <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;__Parameters:__ <br>
+ __classId__(integer)
+ __conf__(float)
+ __left__(float)
+ __top__(float)
+ __right__(float)
+ __bottom__(float)

