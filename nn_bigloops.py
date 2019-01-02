# teaching a computer to binary search

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.nan)
# neural network structure
np.random.seed(1000)
# X, vector with elements 1, value of current guess, and the signal (too high or too low)
X = np.array((1, 3))

l0 = np.ones((3,1))

l0_big = np.random.random((100, 3))

l1 = np.random.random((100,4))
l2 = np.random.random((1,100))
# syn0 is the hidden layer with 4 hidden units
syn0 = 2*np.random.random((3, 4))-1
syn1 = 2*np.random.random((4, 100))-1
# out is the output layer, with 100 possible output values (1-100) inclusive
out = np.empty((1, 100))
alpha = 0.0000005
#alpha = 0.0000009

def sigmoidFunction(x, derivative=False):
    # definition of sigmoid function
    func = 1 / (1.0 + np.exp(-x))
    if (derivative == False):
        return func
    # derivative of sigmoid function
    else:
        return func * (1.0 - func)


def newGuess(h, highOrLow, k0, k1, k2):
    # forward feed
    # normalize h
    x = ((h - 50.00) / 100.00)
    z0 = np.array([1, x, highOrLow], dtype='float64') #################### get this to work!
   # np.put(k0, numGuess % 100, (z0))                   # it seems that the first input is screwing stuff over
    k0[numGuess%100] = z0
    #print(k0)
    #k0 = k0.reshape(k0.shape[0],-1).T

    k1 = sigmoidFunction(k0.dot(syn0))
 #   print(syn1.shape)
    k2 = sigmoidFunction(k1.dot(syn1))

    k2 = k2.reshape(k2.shape[0],-1)
    #print("l2 is : " + str(l2))

    h = ((np.argmax(l2,axis=0)))[0] + 1
    #print(h)
    return h, k0, k1, k2


# log of all guesses made. This is compared to Y to compute total cost.
X_log = np.zeros((100, 1))
cost_log = np.empty((0, 0))
numGuess_log = np.empty((0, 0))
syn0_log = np.empty((0,0))
syn1_log = np.empty((0,0))
i_log = np.empty((0,0))
i = 0
j = 0
numSessions = 500
numIterations = 100

numCorrect = 0
# randomly initialize hypothesis
#test = np.random.randint(1,100,(numIterations,1))

#n = random.randint(1,100)
h = random.randint(1, 100)

while j < numSessions:

    # Master loop with numSessions amount of test numbers to be guessed.
    # We hope that performance gradually improves w/ each increase in j.
    n = random.randint(1, 100)
    h = random.randint(1, 100)
    test = np.full((numIterations, 1), n)
    i = 0
    while i < numIterations:  #### One loop through this means a set of 100 guesses.
                                 # We want the machine to guess much below the max numIterations.
        # guess this number
        #
        numGuess = 0
        #h, l0_big, l1, l2 = newGuess(h, 0, l0_big, l1, l2)#random.randint(1, 100)
    # play the game
        while (numGuess < 100):

            highLow = 0

           # print("Guess is:" + str(h) + ". " + "TRUE is: " + str(test[0]) + "." + " We're on game " + str(i))
            np.put(X_log, (numGuess), h)

            if (h == test[i]):
                numCorrect += 1
                print(str(h) + " is correct! Guessed in " + str(numGuess+i*100) + " guesses. Accuracy: " + str(numCorrect/(j+1)))
                i = numIterations-1
                break
                
            if (h > test[i]):
                #  print("Too high")
                highLow = 1
                h, l0_big, l1, l2 = newGuess(h, 1, l0_big, l1, l2)
            elif (h < test[i]):
                # print("Too low")
                highLow = -1
                h, l0_big, l1, l2 = newGuess(h, -1, l0_big, l1, l2)
            # print(numGuess)
            #print("Guess at " + str(numGuess) + "and " + str(i) + ": " + str(h) + "." + " TRUE is " + str(["TRUE", test[i]))

            numGuess = numGuess + 1

            # print("Guessed in: " + str(numGuess) + " tries")
            # print("Correct guess was: " + str(n))
       # Y = np.full(X_log.shape, n)

        # print(X_log.shape)
        # print(X_log)
        # print(Y)
        # print(Y.shape)
        # compute cost


        numGuess_log = np.append(numGuess_log, numGuess)
       # print("true is : " + str(n))
        # print(i)
        # print(cost)
        # backprop implementation
        # after playing one game, we analyze the results and make the machine improve
        # reshape layers into vectors

        l0_big = l0_big.reshape(l0_big.shape[0],-1)
        l1 = sigmoidFunction(np.dot(l0_big, syn0))
        l2 = sigmoidFunction(np.dot(l1, syn1))
        l1 = l1.reshape(l1.shape[0],-1)
        l2 = l2.reshape(l2.shape[0],-1)
        #print(test)
        l2_error = (1/(numGuess+1))*(abs((X_log - np.full(X_log.shape, test[i]))))
        cost = np.mean(np.abs(l2_error))
        cost_log = np.append(cost_log, cost)
        l2_error = l2_error.reshape(l2_error.shape[0], -1).T
        l2_delta = (l2_error*sigmoidFunction(l2,True))
        # convert layer_2_delta into an actual vector. A 1 dimensional array != a vector.

        l1_error = l2_delta.dot(syn1.T)
       # print("l1_error shape is: " + str(l1_error.shape))
        l1_delta = l1_error * sigmoidFunction(l1,True)
        syn1 -= alpha * (l1.T.dot(l2_delta))
        #print("syn0 is: " + str(syn0.shape))
        syn0 -= alpha * l0_big.T.dot(l1_delta)

        #if (i % (numIterations/2) == 0):
            #for l in range(X_log.shape[0]):
            #    print("Guess is: " + str((X_log[l])) + " True is: " + str(test[0]))
        syn1_log = np.append(syn1_log,np.mean(syn1))
        syn0_log = np.append(syn0_log,np.mean(syn0))
        i_log = np.append(i_log,i)
       # if i % 20 == 0:
      #      print(str((i/numIterations)*100) + "%" + " of Session " + str(j))
       #     print("Guess at " + str(numGuess) + "and " + str(i) + ": " + str(h) + "." + " TRUE is " + str(n))
        i += 1
        #print("i is: " + str(i))
    j+=1
        # clear X_log
        #X_log = np.random.random((100, 1))
        #l0_big = np.random.random((100, 3))
    #print("done.")
    #print(cost_log)
    #print(l2_error)
    #print(syn1)
    #print(syn1.shape)


    #   l0 = X
    # print(syn0.shape)
    # print(test.shape)
    # l1 = sigmoidfunction(np.dot(syn0.T,X.T))
    # print(l1)
    # print(l1.shape)
    # print(X)
    # l1_error = answers.sum(1) - l1
    # l1_delta = np.multiply(l1_error,sigmoidfunction(l1, True))
    # print(l1_delta.shape)
    # print(l0.T.shape)
    #  syn0 += np.dot(l0.T,l1_delta.T)
    # i+=1
    # print(i)
plt.figure(0)
plt.title("cost_log")
plt.scatter(i_log,cost_log, color = 'k', s=5, marker = 'o')
plt.figure(1)
plt.title("syn_log")
plt.scatter(i_log,syn0_log, color = 'k', s=5, marker = 'o')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(syn0_log, syn1_log, cost_log, c='k', marker='o')
plt.show()