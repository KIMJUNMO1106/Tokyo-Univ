from __future__ import print_function
import time
import random
#import numpy as np
import cupy as np
import tensorflow as tf

change=1  #10x10->0, 28x28->1

def add_noise(images, noise_level):
    noisy_images = np.copy(images)
    for image in noisy_images:
        mask = np.random.rand(*image.shape) < noise_level
        noise = np.random.rand(*image.shape)
        image[mask] = noise[mask]
    return noisy_images

def main(epoch):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

    #x_train = tf.image.resize(x_train, (10, 10)).numpy()
    #x_test = tf.image.resize(x_test, (10, 10)).numpy()

    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))


    model = Sequential()
    if change==0:
      # 10x10 version
      model.addlayer(Convsize3Layer(1, 10))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(10, 10))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize2Layer_nopadding(10, 20))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(20, 20))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize3Layer(20, 40))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(40, 40))
      model.addlayer(ReLULayer())
      model.addlayer(AvgPoolingLayer(2))
      model.addlayer(FlattenLayer())
      model.addlayer(LinearLayer(40, 10))
      classifier = Classifier(model)

    elif change ==1 :
      # 28x28 version
      model.addlayer(Convsize3Layer(1, 28))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(28, 28))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize3Layer(28, 56))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(56, 56))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize3Layer_nopadding(56, 112))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize2Layer_nopadding(112, 112))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize3Layer(112, 224))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(224, 224))
      model.addlayer(ReLULayer())
      model.addlayer(AvgPoolingLayer(2))
      model.addlayer(FlattenLayer())
      model.addlayer(LinearLayer(224, 28))
      classifier = Classifier(model)

    else:
      model.addlayer(Convsize3Layer(1, 10))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(10, 10))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize3Layer(10, 20))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(20, 20))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize2Layer_nopadding(20, 40))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(40, 40))
      model.addlayer(ReLULayer())
      model.addlayer(MaxPoolingLayer(2))
      model.addlayer(Convsize2Layer_nopadding(40, 80))
      model.addlayer(ReLULayer())
      model.addlayer(Convsize3Layer(80, 80))
      model.addlayer(ReLULayer())
      model.addlayer(AvgPoolingLayer(2))
      model.addlayer(FlattenLayer())
      model.addlayer(LinearLayer(80, 28))
      classifier = Classifier(model)



    x_train = np.round(x_train/255.0, 2)
    x_test = np.round(x_test/255.0, 2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #noisy_x_train = add_noise(x_train, 0.2)

    tr, te = (x_train, y_train), (x_test, y_test)

    batchsize = 100
    ntrain = 60000
    ntest = 10000

    for e in range(epoch):
        print('epoch %d'%e)
        randinds = np.random.permutation(ntrain)
        for it in range(0, ntrain, batchsize):
            ind = randinds[it:it+batchsize]
            #ind = randinds[it:it+batchsize].get()
            x = tr[0][ind]
            t = tr[1][ind]
            start = time.time()
            #print('a')
            loss, acc = classifier.update(x, t)
            end = time.time()
            print('train iteration %d, elapsed time %f, loss %f, acc %f'%(it//batchsize, end-start, loss, acc))

        start = time.time()
        acctest = 0
        losstest = 0
        for it in range(0, ntest, batchsize):
            x = te[0][it:it+batchsize]
            t = te[1][it:it+batchsize]
            loss, acc = classifier.predict(x, t)
            acctest += int(acc * batchsize)
            losstest += loss
        acctest /= (1.0 * ntest)
        losstest /= (ntest // batchsize)
        end = time.time()
        print('test, elapsed time %f, loss %f, acc %f'%(end-start, loss, acc))

if __name__ == '__main__':
  main(10)
