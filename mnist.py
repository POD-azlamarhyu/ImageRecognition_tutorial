try:
    import urllib.request
except ImportError:
    raise ImportError('you should use python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import tqdm

class Mnist:
    def __init__(self):
        self.url='http://yann.lecun.com/exdb/mnist/'
        self.key={
                'train_img':'train-images-idx3-ubyte.gz',
                'train_label':'train-labels-idx1-ubyte.gz',
                'test_img':'t10k-images-idx3-ubyte.gz',
                'test_label':'t10k-labels-idx1-ubyte.gz'
        }
        self.datasetDir=os.path.dirname(os.path.abspath(__file__))
        self.saveFile=self.datasetDir+"/mnist.pkl"
        self.trainNum=60000
        self.testNum=1000
        self.imgDim=(1,28,28)
        self.imgSize=784


    def download(self,filename):
        filePath="{}/{}".format(self.datasetDir,filename)
        print("Downloding" + filename+".....")
        urllib.request.urlretrieve(self.url+filename,filePath)
        print("has done.")

    def downloadMnist(self):
        for v in tqdm(self.key.values):
            self.download(v)

    def loadLabel(self,filename):
        filePath=self.datasetDir+"/"+filename
        print("Converting "+filename+" to Numpy arrays .....")
        with gzip.open(filePath,'rb') as f:
            labels=np.frombuffer(f.read(),np.uint8,offset=8)
        print("has Done.")

        return labels

    def loadImg(self,filename):
        filePath="{}/{}".format(self.datasetDir,filename)
        print("Converting "+filename+" to Numpy arrays .....")
        with gzip.open(filePath,'rb') as f:
            data=np.frombuffer(f.read(),np.uint8,offset=16)
        data=data.reshape(-1,self.imgSize)
        print("has Done")

        return data

    def convertNp(self):
        dataset = {}
        dataset['train_img'] =  self.loadImg(self.key['train_img'])
        dataset['train_label'] = self.loadLabel(self.key['train_label'])
        dataset['test_img'] = self.loadImg(self.key['test_img'])
        dataset['test_label'] = self.loadLabel(self.key['test_label'])

        return dataset

    def initMnist(self):
        self.downloadMnist()
        dataset=self.convertNp()
        print("Creating pickle file ....")
        with open(self.saveFile,'wb') as f:
            pickle.dump(dataset,f,-1)
        print("has done.")

    def changeOneHotLabel(self,ary):
        x=np.zeros((ary.size,10))
        for i,r in enumerate(x):
            r[x[i]]=1
        
        return x

    def loadMnist(self,normalize=True,flatten=True,oneHotLabel=False):
        if os.path.exists(self.saveFile) == False :
            self.initMnist()
        else:
            pass

        with open(self.saveFile,'rb') as f:
            dataset=pickle.load(f)


        if normalize:
            for key in ('train_img','test_img'):
                dataset[key]=dataset[key].astype(np.float32)
                dataset[key]/= 255.0

        if oneHotLabel:
            dataset['train_label']=self.changeOneHotLabel(dataset['train_label'])
            dataset['test_label']=self.changeOneHotLabel(dataset['test_label'])

        if flatten==False:
            for key in ('train_img','test_img'):
                dataset[key]=dataset[key].reshape(-1,1,28,28)

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

if __name__ == "__main__":
    mnist = Mnist()
    mnist.initMnist()