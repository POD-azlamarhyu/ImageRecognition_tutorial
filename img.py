import cv2
import numpy as np
import glob
import os

class Img:

    def __ini__(self):
        self.moviedata=[]
        

    def forimg(self,path,dir,step,form,num,category):
        mve = cv2.VideoCapture(path)
        Fs=int(mve.get(cv2.CAP_PROP_FRAME_COUNT))
        train="traindata/"
        test="test/"
        pathhead=dir+'/'
        pathfoot='/out_'+'img'+str(num)
        index=np.arange(0,Fs,step)

        

        for i in range(Fs-1):
            flag,frame=mve.read()
            check=i==index

            if flag==True:

                if True in check:
                    if i <10 :
                        path_out=pathhead+pathfoot+'0000'+str(i)+form
                    elif i < 100:
                        path_out=pathhead+pathfoot+'000'+str(i)+form
                    elif i < 1000:
                        path_out=pathhead+pathfoot+'00'+str(i)+form
                    elif i < 10000:
                        path_out=pathhead+pathfoot+'0'+str(i)+form
                    else:
                        pass

                    cv2.imwrite(path_out,frame)

                else:
                    pass
            else:
                pass
        return

    def main(self):
        
        self.moviedata=glob.glob("./data/*")
        img = "./objdata"
        if os.path.exists(img) == False:
            os.mkdir(img)
        else:
            pass
        # print(len(self.moviedata))
        category=["dryer","knife","pc","scissors","outlet"]
        for i in range(len(self.moviedata)):
            print("{} is executed ..... ".format(self.moviedata[i]))
            os.mkdir(img+"/traindata/"+category[i])
            os.mkdir(img+"/testdata/"+category[i])
            self.forimg(self.moviedata[i],img,1,'.png',i,category[i])

if __name__=="__main__":
    execute = Img()
    execute.main()

