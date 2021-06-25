import numpy as np
import cv2
import tensorflow as tf
import time



def preprocessing(img):
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    input_data = (np.float32(input_data) - input_mean) / input_std
    return input_data

def predict(img):
    input_data = preprocessing(img)
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    return boxes,classes,scores

def draw_bndbox(imgOrignal,boxes,classes,scores,labels,imW,imH):
    for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(imgOrignal, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(imgOrignal, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(imgOrignal, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                # Draw circle in center
                xcenter = xmin + (int(round((xmax - xmin) / 2)))
                ycenter = ymin + (int(round((ymax - ymin) / 2)))
                #cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)
                # Print info
                print('Object ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')


def main():
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        new_frame_time = time.time()
        _ , imgOrignal=cap.read()
        boxes,classes,scores = predict(imgOrignal)
        draw_bndbox(imgOrignal,boxes,classes,scores,labels,imW,imH)
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        print (fps)
        cv2.putText(imgOrignal, fps+" fps", (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)       
        cv2.imshow("Result",imgOrignal)
        key=cv2.waitKey(1)
        if key==27:           ## PRESS Esc to triminate the program
            break
    cap.release()
    cv2.destroyAllWindows()



labels= ['Green','Person','Red','Limit 25','Limit 40','Stop']



## Preparing Camera and Loading Haarcascade

min_conf_threshold=0.6
cap=cv2.VideoCapture(1)


input_mean = 127.5
input_std = 127.5



cap.set(3, 640)
cap.set(4, 480)
imW , imH = 640, 480
font=cv2.FONT_HERSHEY_COMPLEX


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter('detect.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]





if __name__ == '__main__':
    main()