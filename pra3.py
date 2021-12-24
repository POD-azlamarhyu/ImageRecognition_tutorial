import sys
import os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist as lmt
from PIL import Image
import pickle
import tqdm

def get_data():
    (x_train,t_train),(x_test,t_test)=lmt(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_newwork():
    with open("mnist.pkl",'rb') as f:
        network=pickle.load(f)
    
    return network

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

def predict(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=softmax(a3)

    return y

x,t=get_data()
network=init_newwork()
accuracy_count=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_count+=1

print("Accuracy : {}".format(float(accuracy_count)/len(x)))

