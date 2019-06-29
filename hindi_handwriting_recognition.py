import numpy as np
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Conv2D,Flatten
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.utils import np_utils, print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as k
import os
os.chdir('C:\\Users\\user\\Documents')

data=pd.read_csv('project.csv')

dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,0:1024]
Y=Y[:,1024]
X_train=X[0:70000,:]
X_train=X_train/255
X_test=X[70000:72001,:]
X_test=X_test/255

Y=Y.reshape(Y.shape[0],1)
Y_train=Y[0:70000,:]
Y_train=Y_train.T
Y_test=Y[70000:72001,:]
Y_test=Y_test.T


image_x=32
image_y=32
train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y=train_y.reshape(train_y.shape[1],train_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
X_train=X_train.reshape(X_train.shape[0],image_x,image_y,1)
X_test=X_test.reshape(X_test.shape[0],image_x,image_y,1)


def keras(image_x,image_y):
    num_of_classes=37
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(image_x,image_y,1),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Conv2D(64,(5,5),activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(5,5),padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    filepath='devanagiri.h5'
    checkpoint1=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list=[checkpoint1]
    return model,callbacks_list

model,callbacks_list=keras(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=1,batch_size=64,callbacks=callbacks_list)
model.save('devanagiri.h5')

import cv2
from keras.models import load_model
from collections import deque

model1=load_model('devanagiri.h5')
print(model1)

letter_count={0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna', 6: 'character_06_cha',
                    7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna',
                    11: '11_taamatar',
                    12: '12_thaa', 13: '13_daa', 14: '14_dhaa', 15: '15_adna', 16: '16_tabala', 17: '17_tha',
                    18: '18_da',
                    19: '19_dha', 20: '20_na', 21: '21_pa', 22: '22_pha',
                    23: '23_ba',
                    24: '24_bha', 25: '25_ma', 26: '26_yaw', 27: '27_ra', 28: '28_la', 29: '29_waw', 30: '30_motosaw',
                    31: '31_petchiryakha',32: '32_patalosaw', 33: '33_ha',
                    34: '34_chhya', 35: '35_tra', 36: '36_gya', 37: 'CHECK'}

def keras_predict(model,image):
    processed=keras_process_image(image)
    print('processed:',processed.shape)
    pred_probab=model.predict(processed)[0]
    pred_class=list(pred_probab).index(max(pred_probab))
    return max(pred_probab),pred_class


def keras_process_image(img):
    image_x=32
    image_y=32
    img=cv2.resize(img,(image_x,image_y))
    img=np.array(img)
    img=np.reshape(img,(-1,image_x,image_y,1))
    return img


cap=cv2.VideoCapture(0)
Lower_blue=np.array([10,50,50])
Upper_blue=np.array([130,255,255])
pred_class=0
pts=deque(maxlen=512)
blackboard=np.zeros((480,640,3),dtype=np.uint8)
digit=np.zeros((200,200,3),dtype=np.uint8)
while(cap.isOpened()):
    ret,img=cap.read()
    img=cv2.flip(img,1)
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV,Lower_blue,Upper_blue)
    blur=cv2.medianBlur(mask,15)
    blur=cv2.GaussianBlur(blur,(5,5),0)
    thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
    center=None
    if len(cnts)>=1:
        contour=max(cnts,key=cv2.contourArea)
        if cv2.contourArea(contour)>500:
            ((x,y),radius)=cv2.minEnclosingCircle(contour)
            cv2.circle(img,(int(x),int(y)),int(radius),(0,255,0),2)
            cv2.circle(img,center,5,(0,0,255),-1)
            M=cv2.moments(contour)
            center=(int(M['m10'] / M['m00']),int(M['m01'] / M['m00'] ))
            pts.appendleft(center)
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard,pts[i-1],pts[i],(255,255,255),10)
                cv2.line(img,pts[i-1],pts[i],(255,0,0),5)
    elif len(cnts)==0:
        if len(pts) != []:
            blackboard_gray=cv2.cvtColor(blackboard,cv2.COLOR_BGR2GRAY)
            blur1=cv2.medianBlur(blackboard_gray,15)
            blur1=cv2.GaussianBlur(blur1,(5,5),0)
            thresh1=cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            blackboard_cnts=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts) >= 1:
                cnt=max(blackboard_cnts,key=cv2.contourArea)
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt)>2000:
                    x,y,w,h=cv2.boundingRect(cnt)
                    digit=blackboard_gray[y:y+h,x:x+w]
                    pred_probab,pred_class=keras_predict(model1,digit)
                    print(pred_class,pred_probab)
            pts=deque(maxlen=512)
            blackboard=np.zeros((480,640,3),dtype=np.uint8)
    cv2.putText(img,'Conv Network:'+str(letter_count[pred_class]),(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow('Frames',img)
    cv2.imshow('Contours',thresh)
    k=cv2.waitKey(10)
    if k==27:
        break
