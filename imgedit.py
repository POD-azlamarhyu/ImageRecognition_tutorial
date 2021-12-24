import glob
import os
import math
from PIL import Image
import cv2


category=["dryer","ketlle","officechair","scissors","outlet"]
datadir="./img/"


newdir="./objdata2/"
train="train/"
test="test/"
size=224

if os.path.exists(newdir)==False:
    os.mkdir(newdir)
else:
    pass

if os.path.exists(newdir+train)==False:
    os.mkdir(newdir+train)

if os.path.exists(newdir+test)==False:
    os.mkdir(newdir+test)

def editImg(file,num,th,cat):
    img=cv2.imread(file)

    imgRe=cv2.resize(img,(size,size))

    print("file name : " , file)

    if num < th:
        cv2.imwrite(newdir+train+cat+"/edit"+str(num)+".png",imgRe)
    elif num >= th:
        cv2.imwrite(newdir+test+cat+"/edit"+str(num)+".png",imgRe)

for i,cat in enumerate(category):
    print("image category:{} executing ".format(cat))
    files=glob.glob(datadir+cat+"/*.png")
    th=math.floor(len(files)*0.8)
    print("image numbers:{}".format(len(files)))

    if os.path.exists(newdir+train+cat) == False:
        os.mkdir(newdir+train+cat)

    if os.path.exists(newdir+test+cat) == False:
        os.mkdir(newdir+test+cat)

    for j,f in enumerate(files):
        print("editnumber "+str(j))
        print("filename : {} num:{}".format(f,j))

        if j < 100:
            editImg(f,j,th,cat)
        else:
            pass