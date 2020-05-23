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


def various_amount_of_pretraining(r, batch_size, num_particles, timesteps):

    
    x_train, y_train, x_test, y_test = load_mnist()
    #x_train = x_train[:1000]
    #y_train = y_train[:1000]
    #x_test = x_test[:1000]
    #y_test = y_test[:1000]


    ## Full ENKF Run

    model = initialize_model()
    wvec = extract_weight_vector(model)
    meas_model = lambda wvec, xs: predict_nn(wvec, xs, model)

    logger = TrainingLogger()
    Ts = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500, 700, 900]
    As = estimate_weights_enkf(wvec, x_train, y_train, x_test, y_test, dx=0.1, meas_model=meas_model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles, batch_size=batch_size, r=r, logger=logger, Ts=Ts, parallelize=False)

    #A = As[timesteps-1]
    #np.savetxt('ensemble.csv', A, delimiter=',')

    ## Plain Backprop Run (no initialization)

    loss_function = keras.losses.mean_squared_error

    history_vanilla = train_backprop_from_init(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, batch_size=batch_size, loss_function=loss_function)

    ## Backprop runs with ENKF initialization at each Ts

    histories = {}

    for Tsi in Ts:
        A = As[Tsi]
        wvec = np.mean(A, 1)
        history = train_backprop_from_init(x_train=x_train[Tsi*batch_size:], y_train=y_train[Tsi*batch_size:], x_test=x_test, y_test=y_test, batch_size=batch_size, loss_function=loss_function, initial_weights=wvec)
        histories[Tsi] = history

    ## Plot learning curves

    fig1, (ax1) = plt.subplots(1, 1)
    
    logger.plot_errors(ax=ax1)
    ax1.plot(range(len(history_vanilla.val_accuracies)), history_vanilla.val_accuracies, linewidth=1)
    for Tsi in Ts:
        history = histories[Tsi]
        ax1.plot(range(Tsi, Tsi + len(history.val_accuracies)), history.val_accuracies, linewidth=1)

    legend_items = ['ENKF Only', 'Backprop Only'] + ["MSE Backprop from " + str(Tsi) + " ENKF iterations" for Tsi in Ts]
    ax1.legend(legend_items, fontsize=13)
    ax1.set_ylim([0.0, 1.0])

    plt.title('MNIST TEST', fontsize=20)
    plt.savefig('test-acc-enkf.png', format='png', dpi=1200)


def train_backprop_from_init(x_train, y_train, x_test, y_test, batch_size, loss_function, initial_weights=None):
    model = initialize_model()
    if initial_weights is not None:
        model = set_weights_from_vector(model, initial_weights)

    model.compile(loss=loss_function,
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD(),# (learning_rate=0.0001), #
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = LossHistory()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf+backprop):', score[0])
    print('Test accuracy (enkf+backprop):', score[1])

    return history



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

    various_amount_of_pretraining(r=r, batch_size=batch_size, num_particles=num_particles, timesteps=timesteps)

    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)


if __name__ == "__main__":
    signal(SIGINT, handler)
    main(sys.argv)