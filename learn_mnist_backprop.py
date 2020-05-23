from signal import signal, SIGINT
from sys import exit

from nn_utils import predict_nn, set_weights_from_vector, extract_weight_vector, evaluate_performance
from nn_model import initialize_model, LossHistory
from enkf_algo_spedup import estimate_weights_enkf
from plotting import TrainingLogger

import sys

import numpy as np

import multiprocessing
from joblib import Parallel, delayed
#import dill as pickle

import keras
from keras.datasets import mnist
from keras import backend as K

import cProfile
import pstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logger = TrainingLogger()


def learn_mnist_enkf_plus_backprop(r, batch_size, num_particles, timesteps):

    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()

    history = LossHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf+backprop):', score[0])
    print('Test accuracy (enkf+backprop):', score[1])

    #print(history.val_accuracies)

    #samples_per_test = 5*16
    #logger.plot_errors()
    #plt.plot([i*batch_size+len(logger.val_acc)*samples_per_test for i in range(len(history.val_accuracies))], history.val_accuracies)
    #plt.savefig('test-acc-enkf.png', format='png', dpi=1200)
    #logger.plot_eigenvals()
    #plt.savefig('eigenvals-enkf.png', format='png', dpi=1200)

    #plt.show()


def load_mnist():
    batch_size = 16
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def parse_args(args):
    # default parameters
    r = 0.001
    batch_size=16
    num_particles=200
    timesteps=1001


    for arg in args:
        if "=" in arg:
            tag = arg.split("=")[0]
            value = arg.split("=")[1]
            if tag == "--r": r = float(value); 
            if tag == "--batch_size": batch_size = int(value)
            if tag == "--num_particles": num_particles = int(value)
            if tag == "--timesteps": timesteps = int(value)

    print(r)
    return r, batch_size, num_particles, timesteps

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Saving plots...')

    logger.plot_errors()
    plt.savefig('test-acc-enkf-select.png', format='png', dpi=1200)
    logger.plot_eigenvals()
    plt.savefig('eigenvals-enkf-select.png', format='png', dpi=1200)

    print('Plots saved on exit.')
    exit(0)

def main(args):

    pr = cProfile.Profile()
    pr.enable()

    np.random.seed(1)

    r, batch_size, num_particles, timesteps = parse_args(args)

    learn_mnist_enkf_plus_backprop(r=r, batch_size=batch_size, num_particles=num_particles, timesteps=timesteps)

    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)


if __name__ == "__main__":
    signal(SIGINT, handler)
    main(sys.argv)