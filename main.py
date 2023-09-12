import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor=SelfiSegmentation(1)
fpsReader=cvzone.FPS()
imgBg=cv2.imread("1.jpeg")
list=os.listdir()
print(list)
list=[]
for imgPath in list:
    img=cv2.imread(f'Images/{imgPath}')
    list.append(img)
print(len(list))

while True:
    success,img=cap.read()
    imgOut=segmentor.removeBG(img,imgBg,threshold=0.8)

    imgStack=cvzone.stackImages([img,imgOut],2,1)
    _,imgStack=fpsReader.update(imgStack,color=(0,0,255))
    cv2.imshow("Image",imgStack)

    cv2.waitKey(1)