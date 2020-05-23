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

def generate_comparison_mnist(r, initial_noise, batch_size, num_particles, num_epochs, timesteps):

    
    x_train, y_train, x_test, y_test = load_mnist()
    


    use_digits = [1, 2, 3]#, 3, 4, 5]
    use_samples_train = np.zeros(len(y_train), dtype=bool)
    maskmult = np.eye(10)
    for i in range(len(y_train)):
        digit = np.argmax(y_train[i])
        if digit in use_digits:
            use_samples_train[i] = True

    use_samples_test= np.zeros(len(y_test), dtype=bool)
    for i in range(len(y_test)):
        digit = np.argmax(y_test[i])
        if digit in use_digits:
            use_samples_test[i] = True

    #x_train = x_train[use_samples_train]#[0:128]
    #y_train = y_train[use_samples_train]#[0:128]
    #x_test = x_test[use_samples_test]#[0:100]
    #y_test = y_test[use_samples_test]#[0:100]

    print(y_train)



    #x_train = x_train[:2000]
    #y_train = y_train[:2000]
    #x_test = x_test[:2000]
    #y_test = y_test[:2000]


    ## Full ENKF Run

    model = initialize_model()
    wvec = extract_weight_vector(model)
    meas_model = lambda wvec, xs: predict_nn(wvec, xs, model)

    pretrain_steps = 200
    Ts = [pretrain_steps, timesteps-1]
    #loss_function = "categorical_crossentropy"
    loss_function = "mse"
    #loss_function=None
    adapt_r = "Deviation-proportional"
    As = estimate_weights_enkf(wvec, x_train, y_train, x_test, y_test, dx=0.1, meas_model=meas_model, timesteps=timesteps, num_epochs=num_epochs, ensemble_size=num_particles, batch_size=batch_size, r=r, logger=logger, adapt_r=adapt_r, loss_function=loss_function, initial_noise=initial_noise, Ts=Ts, parallelize=False)

    A = As[timesteps-1]

    #np.savetxt('ensemble.csv', A, delimiter=',')

    model.compile(loss=keras.losses.mean_squared_error,
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD(),#(learning_rate=0.001), #
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    wvec_final = np.mean(A, 1)
    model = set_weights_from_vector(model, wvec_final)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf):', score[0])
    print('Test accuracy (enkf):', score[1])

    ## Backprop with MSE loss with ENKF initialization

    A = As[pretrain_steps]
    wvec = np.mean(A, 1)
    loss_function = keras.losses.mean_squared_error
    history_mse = train_backprop_from_init(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs=num_epochs, batch_size=batch_size, loss_function=loss_function, initial_weights=wvec)

    ## Backprop with crossentropy loss with ENKF initialization

    A = As[pretrain_steps]
    wvec = np.mean(A, 1)
    loss_function = keras.losses.categorical_crossentropy
    history_xe = train_backprop_from_init(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs=num_epochs, batch_size=batch_size, loss_function=loss_function, initial_weights=wvec)

    ## Backprop with MSE loss without ENKF initialization

    loss_function = keras.losses.mean_squared_error
    history_mse_only = train_backprop_from_init(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs=num_epochs, batch_size=batch_size, loss_function=loss_function)

    ## Backprop with crossentropy loss without ENKF initialization

    loss_function = keras.losses.categorical_crossentropy
    history_xe_only = train_backprop_from_init(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs=num_epochs, batch_size=batch_size, loss_function=loss_function)

    ## Plot learning curves

    fig1, (ax1) = plt.subplots(1, 1)
    
    logger.plot_errors(ax=ax1, invert=False)
    ax1.plot([i+pretrain_steps+1 for i in range(len(history_mse.val_accuracies))], np.array(history_mse.val_accuracies), linewidth=3)
    ax1.plot([i+pretrain_steps+1 for i in range(len(history_xe.val_accuracies))], np.array(history_xe.val_accuracies), linewidth=3)
    ax1.plot(range(len(history_mse_only.val_accuracies)), np.array(history_mse_only.val_accuracies), linewidth=3)
    ax1.plot(range(len(history_xe_only.val_accuracies)), np.array(history_xe_only.val_accuracies), linewidth=3)

    ax1.legend(['ENKF Only', 'ENKF + MSE Backprop', 'ENKF + Crossentropy Backprop', 'MSE Backprop Only', 'Crossentropy Backprop Only'], fontsize=13)
    #ax1.set_yscale("log")
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xticks([i*timesteps for i in range(num_epochs+1)])
    ax1.set_xticklabels(["Epoch " + str(i) for i in range(num_epochs+1)])

    plt.title('MNIST Test Error Rate', fontsize=20)
    plt.savefig('test-acc-enkf.png', format='png', dpi=1200)

    ax1.set_ylim([0.9, 1.0])
    plt.savefig('test-acc-enkf-zoom.png', format='png', dpi=1200)

    #fig2 = plt.figure()
    #logger.plot_eigenvals(fig=fig2)
    #fig2.savefig('eigenvals-enkf.png', format='png', dpi=1200)

    #plt.show()

def train_backprop_from_init(x_train, y_train, x_test, y_test, num_epochs, batch_size, loss_function, initial_weights=None):
    model = initialize_model(input_shape=(28, 28, 1))
    if initial_weights is not None:
        model = set_weights_from_vector(model, initial_weights)

    model.compile(loss=loss_function,
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD(),# (learning_rate=0.0001), #
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = LossHistory(test_data=(x_test, y_test))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,    
              validation_data=(x_test, y_test),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf+backprop):', score[0])
    print('Test accuracy (enkf+backprop):', score[1])

    return history



def load_mnist():
    num_classes = 10

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
    initial_noise=0.03
    batch_size=64
    num_particles=1000
    num_epochs=20
    timesteps=901


    for arg in args:
        if "=" in arg:
            tag = arg.split("=")[0]
            value = arg.split("=")[1]
            if tag == "--r": r = float(value)
            if tag == "--initial_noise": initial_noise = float(value)
            if tag == "--batch_size": batch_size = int(value)
            if tag == "--num_particles": num_particles = int(value)
            if tag == "--num_epochs": num_epochs = int(value)
            if tag == "--timesteps": timesteps = int(value)

    print(r)
    return r, initial_noise, batch_size, num_particles, num_epochs, timesteps

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

    r, initial_noise, batch_size, num_particles, num_epochs, timesteps = parse_args(args)

    generate_comparison_mnist(r=r, initial_noise=initial_noise, batch_size=batch_size, num_particles=num_particles, num_epochs=num_epochs, timesteps=timesteps)

    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)


if __name__ == "__main__":
    signal(SIGINT, handler)
    main(sys.argv)
