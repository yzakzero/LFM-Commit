__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''
'''
数据下载地址：https://grouplens.org/datasets/movielens/
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle,sys
import os

import matplotlib.pyplot as plt
import matplotlib;
matplotlib.use('TkAgg')

def splitTrainSetTestSet(odatas,frac):
    testset = odatas.sample(frac=frac, axis=0)
    trainset = odatas.drop(index=testset.index.values.tolist(), axis=0)
    return trainset,testset

def readDatas():
    path = 'ml-latest-small/ratingsNew.csv'
    odatas = pd.read_csv(path,usecols=[0,1,2])
    return odatas

class LFM():
    def __init__(self,dataset,factors, epochs, lr, lamda):
        self.dataset = dataset

        self.userList, self.itemList = self.__getListMap()
        self.factors=factors
        self.epochs=epochs
        self.lr=lr
        self.lamda=lamda

        self.p = pd.DataFrame(np.random.randn(len(self.userList), factors), index=self.userList)
        self.q = pd.DataFrame(np.random.randn(len(self.itemList), factors), index=self.itemList)
        self.bu = pd.DataFrame(np.random.randn(len(self.userList)), index=self.userList)
        self.bi = pd.DataFrame(np.random.randn(len(self.itemList)), index=self.itemList)

    def __prediction(self,pu, qi, bu, bi):
        return (np.dot(pu, qi.T) + bu + bi)[0]

    def __getError(self,r, pu, qi, bu, bi):
        return r - self.__prediction(pu, qi, bu, bi)

    def __getListMap(self):
        userSet, itemSet = set(), set()
        for d in self.dataset.values:
            userSet.add(int(d[0]))
            itemSet.add(int(d[1]))
        userList = list(userSet)
        itemList = list(itemSet)
        return userList, itemList

    def fit(self):
        for e in tqdm(range(self.epochs)):
            for d in self.dataset.values:
                u, i, r = d[0], d[1], d[2]
                error = self.__getError(r, self.p.loc[u], self.q.loc[i], self.bu.loc[u], self.bi.loc[i])
                self.p.loc[u] += self.lr * (error * self.q.loc[i] - self.lamda * self.p.loc[u])
                self.q.loc[i] += self.lr * (error * self.p.loc[u] - self.lamda * self.q.loc[i])
                self.bu.loc[u] += self.lr * (error - self.lamda * self.bu.loc[u])
                self.bi.loc[i] += self.lr * (error - self.lamda * self.bi.loc[i])

    def __RMSE(self,a, b):
        #print(a)
        return(np.average((np.array(a) - np.array(b)) ** 2)) ** 0.5

    def testRMSE(self,testSet):
       # print(testSet)

        y_true, y_hat = [], []
        for d in tqdm(testSet.values):
            user = int(d[0])
            item = int(d[1])

            # print(user)
            # print(item)


            if user in self.userList and item in self.itemList:
                hat=self.__prediction(self.p.loc[user], self.q.loc[item], self.bu.loc[user], self.bi.loc[item])
                y_hat.append(hat)
                y_true.append(d[2])


        rmse = self.__RMSE(y_true,y_hat)



        return rmse





    def save(self,path):
        with open(path,'wb+') as f:
            pickle.dump(self,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            return pickle.load(f)

def play(factors):
    factors=factors #隐因子数量
    epochs=10 #迭代次数
    lr=0.01 #学习率
    lamda=0.1 #正则项系数

    model_path='model/lfm.model'

    trainset, testSet = splitTrainSetTestSet(readDatas(),0.2)
    # print(trainset)
    # print(testSet)

    #lfm=LFM.load(model_path)

    lfm=LFM(trainset,factors, epochs, lr, lamda)
    lfm.fit()
    lfm.save(model_path)

    rmse_test = lfm.testRMSE(testSet)

    # print(testSet)
    # sys.exit()





    rmse_train = lfm.testRMSE(trainset)

    print(factors)
    print('rmse_train:'+str(rmse_train))
    print('rmse_test:'+str(rmse_test))
    return rmse_train

def Save(l='list',n='docname'):
    # 创建文件夹，存放输出结果
    if not os.path.exists("result_variable"):
        os.mkdir("result_variable")
    # 保存数据
    np.savez('result_variable/'+n,l)


def load(n='docname'):
    train_acc = np.load('result_variable/'+n+'.npz')
    print(train_acc['arr_0'])  # 查看各个数组名称

if __name__ == '__main__':

   Count=[]#影响因子量
   Rmse=[]#Rmse

   for i in  range(1,10):
        Count.append(i)
        Rmse.append(play(i))


   #print(s,t)

   #
   # Save(Count,'Count')
   # Save(Rmse,'Rmse')





   plt.plot(Count, Rmse, color='red')
  # 10 1.0854367966564888

   plt.title('train Rmse')
   plt.xlabel("Factors")
   plt.ylabel("Rmse")
   plt.show()

   # train_acc = np.load('result_variable/Rmse.npz')
   # print(train_acc['arr_0'])  # 查看各个数组名称