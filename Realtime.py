from keras.models import  load_model
from keras import datasets, Sequential
import cv2 
import numpy as np 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import  image
import time
model = Sequential()
#url = "http://192.168.137.249:4747/video"
video = cv2.VideoCapture(0)
label = ['NguyenCongPhuong', 'HaAnhTuan', 'HuongGiang', 'DongNhi', 'PhanVanDuc', 'DamVinhHung', 'TrongHoang', 'SonTungMTP', 'DenVau', 'DoanVanHau', 'AnhDuc']
model = load_model("Model.h5")

dem = 0 
while True:
    #Doc IP tu Cam 
    ret, frame = video.read() #ret la ket qua doc anh , frame la anh doc ve 
    #Neu doc thanh cong thi hien thi
    if ret == True:
        # Resize
        img = frame.copy()
        img = cv2.resize(img,(128,128))
        img = img_to_array(img)
        img = img.reshape(1,128,128,3) 
        img = img.astype('float32') 
        img = img/255 
        predict = model.predict(img)
        # Predict
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2
        dem = dem + 1 
        if dem == 5:
            k = str(np.max(predict)*100)
            text =  ('Ti le giong voi dien vien '+ label[np.argmax(predict)] + 'la' +  k  + '%')
            cv2.putText(frame, text ,org, font,fontScale/3, color, thickness, cv2.LINE_AA)
            dem =0
        cv2.imshow("IPCam",frame)
        
    #Dem q de thoat
    if cv2.waitKey(1) == ord("q"):
        break
video.release()
cv2.destroyAllWindows()