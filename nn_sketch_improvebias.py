# cookie, donut, or neither?

import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_dim = 784     # images are 28*28 = 784 total pixels
hidden_dim = 700    # hidden units
output_dim = 3      # outputs either 'donut', 'cookie', or 'neither'


random.seed(1000)

class Image:
    def __init__(self, bitmap, identifier):
        self.data = bitmap
        self.tag = identifier

class NeuralNet:
    def __init__(self):
        # NN structure variables
        # layer 0 (input layer)
        self.l0 = np.ones((1,input_dim))
        self.l0_bias = self.l0
        # layer 1 (hidden layer #1)
        self.l1 = np.random.random((1,hidden_dim))
        self.l1_bias = self.l1
        # layer 2 (output layer)
        self.l2 = np.random.random((1,output_dim))
        # synapse 0 (weight connecting input layer to hidden layer)
        self.syn0 = 2 * np.random.random((input_dim+1,hidden_dim)) - 1
        # synapse 1 (weight connecting hidden layer to output layer)
        self.syn1 = 2 * np.random.random((hidden_dim+1,output_dim)) - 1
        # synapse H (connecting previous layers to current layer)
        self.deltaLog = np.empty((0, 0))

        # Backprop and gradient descent variables
        # learning rate
        self.alpha = 0.01
        self.l1_error = np.random.random((1,hidden_dim))
        self.l1_delta = np.random.random((1,hidden_dim))

        self.l1_error = np.zeros((1,hidden_dim))
        self.l2_error = np.zeros((1,output_dim))
        self.l2_delta = np.zeros((1,output_dim))


        self.syn0_update = np.zeros_like(self.syn0)
        self.syn1_update = np.zeros_like(self.syn1)

        self.cost = 0;

        # Keep track of guesses
        self.guessLog = np.empty((0,0))
        self.costLog = np.empty((0,0))
        self.changeLog = list()
    def sigmoidFunction(self, x, derivative=False):
        #definition of sigmoid function
        func = 1 / (1.0 + np.exp(-x))
        if (derivative == False):
            return func
        # derivative of sigmoid function
        else:
            return func * (1.0 - func)

    def relu(self, x, derivative = False):# relu
        if (derivative == False):
            return x * (x > 0)
        elif (derivative == True):
            return 1. * (x > 0)

    def costFunction(self, x, y, derivative = False):
        if (derivative == False):
            #return (-(np.sum(np.mean((y * np.log(x) + (1-y)*np.log((1-x)))))))
            #logprobs = -(np.multiply(y, np.log(x)) + np.multiply((1.0-y),np.log((1.0-x))))
            logprobs = (y-x)**2
            cost = np.mean(np.sum(logprobs))
            return cost
    def backprop(self, h, ans):     # h is an int, ans is an Image ob# ject

        #self.l1 = self.sigmoidFunction(np.dot(self.l0_bias,self.syn0))

        #self.l2 = self.sigmoidFunction(np.dot(self.l1_bias,self.syn1))

        ansMatrix = np.zeros(self.l2.shape)

        ansMatrix[0, ans.tag] = 1
        self.l2_error = (ansMatrix - self.l2)
        # print("other cost " + str(np.mean(np.sum(self.l2_error))))
        # #   print(np.mean(self.l2_delta))
        self.cost = self.costFunction(self.l2,ansMatrix)
        #print("cost " + str(self.cost))
        # self.l2_error = self.costFunction(self.l2, ansMatrix)
        # print(self.l2_error)

        self.costLog = np.append(self.costLog, self.cost)
        self.l2_delta = self.l2_error * self.sigmoidFunction(self.l2, derivative=True)
        self.changeLog = np.append(self.changeLog, np.mean(self.l2_delta))
        self.l1_error = self.l2_delta.dot(self.syn1.T)
        self.l1_delta = self.l1_error * self.sigmoidFunction(self.l1_bias,derivative=True)

        # # print(self.l0_bias.T.shape)
        # # print(self.l1_delta.shape)
        # # # print("synapse is " + str(self.syn1_update))
        # after looping through all layers, add updated weights


        # gradient checking


        #print(np.mean(self.alpha * self.l1_bias.T.dot(self.l2_delta)))
        self.syn1 += self.alpha * self.l1_bias.T.dot(self.l2_delta)
        self.syn0 += self.alpha * self.l0_bias.T.dot(self.l1_delta[:,:-1])
        #print("syn0: " + str(np.mean(self.alpha * self.l1_bias.T.dot(self.l2_delta))))
        # self.l1 = self.l1_bias[:, -1]
        # self.l0 = self.l0_bias[:, -1]
        ## # print(self.syn2)
        # also reset synapse updates
        #self.l2_error *= 0

    def forwardFeed(self, image):
        # convert l1 and l2 into matrices (currently they are vectors)
        self.l0 = image.data
        self.l0 = self.l0.reshape(self.l0.shape[0], -1).T
        print(self.l0)
        # normalize input
        self.l0 = (self.l0-np.mean(self.l0))/np.std(self.l0)

        # # print("l0 at fwd " + str(self.l0.shape))
        self.l0_bias = np.append(self.l0, np.ones((1,1)), axis=1)

        # # print(self.l1.shape)
        self.l2 = self.l2.reshape(self.l2.shape[0], -1)
        self.l1 = self.sigmoidFunction(self.l0_bias.dot(self.syn0))

        # # print("l1 shape si " + str(self.l1.shape))
        self.l1_bias = np.append(self.l1,np.ones((1,1)), axis=1)
        # # print(self.l1.shape)
        # # print("l1 shape is : " + str(self.l1.shape))
        # # print("syn1.shape is : " + str(self.syn1.shape))

        self.l2 = self.sigmoidFunction(self.l1_bias.dot(self.syn1))

        #self.l2 = self.l2.reshape(self.l2.shape[0], -1).T
        # # print("l2 is : " + str(self.l2))
        # # print(self.l2.shape)
        h = np.argmax(self.l2)

        ## # print("new guess is " + str(h))


        return h


#**** Data loading and Processing ****#

# sketches of donuts (output [0])
donutData = np.load('sketch/donut.npy')
# sketches of cookies (output [1])
cookieData = np.load('sketch/cookie.npy')
# # print(cookieData.shape[0])
# # print(donutData.shape[0])
# # sketches of neither (output [2])
neitherData = np.load('sketch/clock.npy')
neitherData = np.append(neitherData, np.load('sketch/apple.npy'), axis = 0)
neitherData = np.append(neitherData, np.load('sketch/dog.npy'), axis = 0)
print(neitherData.shape)

numExamples = donutData.shape[0]+cookieData.shape[0]+neitherData.shape[0]
print("numex is : " + str(numExamples))

total= list()
train = list()
test = list()

for i in range(donutData.shape[0]):
    total.append(Image(donutData[i,:], 0))      # 0 represents "donut"
for i in range(cookieData.shape[0]):
     total.append(Image(cookieData[i,:], 1))     # 1 represents "cookie"
for i in range(neitherData.shape[0]):
     total.append(Image(neitherData[i,:], 2))    # 2 represents "neither"

# randomize list
random.shuffle(total)
# # print(int(numExamples/(10/7)))
# split total examples into training and test examples (70-30 split)
train = total[:int(numExamples/(10/9))]
test = total[int(numExamples/(10/9)):]
# # print(len(train))
# # print(len(test))
# # print(len(train)+len(test))

# # print(total[25262].tag)

net = NeuralNet()

# training loop
numCorrect = 0
cost_log = list()
acc_log = list()
guessList = list()
trueList = list()
epoch = 6
for j in range(epoch):
    for i in range(len(train)):
        h = net.forwardFeed(train[i])
        if h == train[i].tag:
            numCorrect += 1
        guessList.append(h)
        trueList.append(train[i].tag)
        # # print(net.cost)
        net.backprop(h, train[i])
        cost_log.append(net.cost)
        acc_log.append(numCorrect/(i+1))
        #print("Iteration: " + str(i) + " " + "Guess is : " + str(h) + " . True is : " + str(train[i].tag) + ". Accuracy is : " + str(numCorrect / (i + 1)) + " Cost is " + str(net.cost))

        # realtime plotting
        if (i % int((len(train)/500)) == 0):
            print("Epoch: " + str(j) + "    " + "Iteration: " + str(i) + " " + "Guess is : " + str(h) + " . True is : " + str(train[i].tag) + ". Accuracy is : " + str(numCorrect/(i+1)) + " Cost is " + str(net.cost))

            # plt.figure(1000)
            # plt.title("cost")
            # plt.plot(np.arange(0,len(acc_log)), cost_log)
            #
            # plt.figure(1001)
            # plt.title("accuracy")
            # plt.scatter(np.arange(0,len(acc_log)), acc_log, color=(0.0, 0.5, 0.0), s=1, marker='o')

            # plt.figure(1002)
            # plt.title("change")
            # plt.scatter(np.arange(0,len(net.changeLog)), net.changeLog, color=(0.0, 0.5, 0.0), s=1, marker='o')
            # # plt.scatter(np.arange(0,len(guessList)), guessList, color=(0.0, 0.5, 0.0), s=1, marker='o')
            # # plt.scatter(np.arange(0,len(trueList)), trueList, color=(0.0, 0, 0.0), s=0.5, marker='o')
            plt.show(block=False)
            plt.pause(0.1)
    acc_log.clear()
    numCorrect = 0

    # shuffle after each epoch
    random.shuffle(train)


np.save(file="syn0.npy", arr=net.syn0)
np.save(file="syn1.npy", arr=net.syn1)
np.save(file="l0.npy", arr=net.l0)
np.save(file="l0_bias.npy", arr=net.l0_bias)
np.save(file="l1.npy", arr=net.l1)
np.save(file="l1_bias.npy", arr=net.l1_bias)
np.save(file="l2.npy", arr=net.l2)

np.savetxt(fname="syn0.txt", X=net.syn0)
np.savetxt(fname="syn1.txt", X=net.syn1)

np.savetxt(fname="l0.txt", X=net.l0)
np.savetxt(fname="l0_bias.txt", X=net.l0_bias)
np.savetxt(fname="l1.txt", X=net.l1)
np.savetxt(fname="l1_bias.txt", X=net.l1_bias)
np.savetxt(fname="l2.txt", X=net.l2)
# test loop
numCorrect = 0
for i in range(len(test)):
    h = net.forwardFeed(test[i])
    if h == test[i].tag:
        numCorrect += 1
    print("Accuracy: " + str(numCorrect/len(test)))
