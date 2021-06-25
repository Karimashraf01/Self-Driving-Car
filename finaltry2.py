import cv2
import numpy as np
import houghmodule as hm
import time
import from_saved_model as pd
import websocket



#intialize web socket
ws = websocket.WebSocket()
ip="192.168.1.8"


def connect_ws(ip):
    try:
        ws.connect("ws://"+ip) # كان في اتنين واحد اسمه محمد و واحد اسمه عيدو محمد مات,يتبقى مين
        print ("...............")
    except:
        connect_ws(ip) # عيد




def getImg(display=False, size=[480, 240]):
    _, img = cap.read()
    img = cv2.resize(img, (size[0], size[1]))
    if display:
        cv2.imshow('IMG', img)
    return img


connect_ws(ip)

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(1)
prev_prediction=''


if __name__ == '__main__':
    ws.send('170')
    while True:
        new_frame_time = time.time()
        img = getImg(False,[640,480])
        imgod = img.copy()
        steering_angle , heading_image = hm.predict(img)
        deviation = steering_angle - 90
        error = abs(deviation)
        deviation = - deviation
        if deviation < 5 and deviation > -5:
            deviation = 0
            error = 0
        print(deviation)
        if deviation > 45:
            deviation = 45
        if deviation < -45:
            deviation = -45
        if deviation > 5 and deviation < 45:
            # print(deviation)
            ws.send(str(deviation+90))
        if deviation < -5 and deviation > -45:
            # print(deviation)
            ws.send(str(deviation+90))

    
    
        #prediction
        pred_bbox,_,classes,_=pd.predict(imgod) 
        image = pd.draw_bbox(imgod, pred_bbox)
        #get predicted Class index
        # print(classes)
        # print(index)
        # #send action
        if classes.size != 0:
            for c in classes:
                if c == 5 or c == 1 or c == 2 :
                    if prev_prediction==str(1):
                        pass
                    else:
                        ws.send(str(1)) # Stop
                        prev_prediction = str(1)
                elif c == 0 or c == 4:
                    if prev_prediction==str(2):
                        pass
                    else:
                        ws.send(str(2)) # full Speed
                        prev_prediction = str(2)
                elif c==3 :
                    if prev_prediction==str(3):
                        pass
                    else:
                        ws.send(str(3)) # half Speed
                        prev_prediction = str(3)
        else:
            if prev_prediction == str('2'):
                pass
            else:
                ws.send(str(2)) # full Speed
                prev_prediction = str(2)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        
        cv2.putText(image, fps + " fps", (7, 35), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('Resutlt',image)
        # cv2.imshow('Curve',imgResult)
        Stacked = np.hstack((image,heading_image))
        cv2.imshow('stacked',Stacked)
        # print('FPS: '+ fps)
        #print(curve)
        key=cv2.waitKey(1)
        if key == 27:
            ws.send(str(1))
            break
    ws.send(str(1))
    ws.close()
    cv2.destroyAllWindows()
    cap.release()
    

