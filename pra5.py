import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mnist import Mnist
from convent import SimpleConvNet
from trainer import Trainer

class train_cnn:
    def __init__(self):
        self.mnist=Mnist()
        (self.x_train,self.t_train),(self.x_test,self.t_test)=self.mnist.loadMnist(flatten=False)
        self.network=SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
        self.maxEpochs=20
        self.trainer=Trainer(self.network,self.x_train,self.t_train,self.x_test,self.t_test,epochs=self.maxEpochs,mini_batch_size=100,optimizer='Adam',optimizer_param={'lr':0.001},evaluate_sample_num_per_epoch=1000)

    def main(self):
        self.trainer.train()
        self.network.save_params("params.pkl")
        print("Saved network parameters.")

        merkers={'train':'o','test':'s'}
        
        x=np.arange(self.maxEpochs)
        plt.plot(x, self.trainer.train_acc_list, marker='o', label='train', markevery=2)
        plt.plot(x, self.trainer.test_acc_list, marker='s', label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

if __name__=="__main__":
    imgCog=train_cnn()
    imgCog.main()