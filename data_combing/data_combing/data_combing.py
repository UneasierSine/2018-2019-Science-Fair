import cv2
import numpy as np
from tkinter.filedialog import askdirectory
import os

counter = 0
iteration = 0
data = np.zeros(shape=(10,14))
keypoints = []

width = 1280
height = 720
GRID_SIZE = 10

rangeCols = 128
rangeRows = 72

rangeWidth = 10
rangeHeight = 10

def click_box(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global keypoints
        global counter
        counter += 1
        horBlock = int(x / 10)
        verBlock = int(y / 10)
        blockNum = (horBlock + 1 ) + (verBlock + 1) * 128
        keypoints.append(blockNum)
        print("Keypoint ",counter," [",x,",",y,"] [",horBlock,",",verBlock,"] ",blockNum,"\n")
        cv2.rectangle(img,(horBlock * 10, verBlock * 10), ((horBlock+1) * 10, (verBlock+1) * 10), (0,255,0), 1)

dir = askdirectory()

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    img = cv2.imread(dir+"/"+filename, 1)
    print(dir)

    for x in range(rangeCols):
        cv2.line(img, (rangeWidth * x, 0), (rangeWidth * x, 719),(255,0,0),1)

    for y in range(rangeRows):
        cv2.line(img, (0, rangeHeight * y), (1279, rangeHeight * y),(255,0,0),1)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_box)

    while(1):
        cv2.imshow("Image", img)
        #if cv2.waitKey(20) & 0xFF == 27:
        if cv2.waitKey(30) == ord('x'):
            counter += 1
            print("Keypoint ",counter,"\n")
            keypoints.append(0)
        if cv2.waitKey(30) == ord('a'):
            print(keypoints)
            break
        if counter > 13:
            print(keypoints)
            data[iteration] = keypoints
            iteration += 1
            keypoints = []
            counter = 0
            break

    if iteration > 10:
        print(data)
        np.savetxt("keypoints.csv",data,delimiter=",")
        break

cv2.destroyAllWindows()

