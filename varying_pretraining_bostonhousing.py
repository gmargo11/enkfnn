from signal import signal, SIGINT
from sys import exit

from nn_utils import predict_nn, set_weights_from_vector, extract_weight_vector, evaluate_performance
from nn_model import initialize_model_boston_housing, LossHistory
from enkf_algo_spedup import estimate_weights_enkf
from plotting import TrainingLogger

import sys

import numpy as np

import multiprocessing
from joblib import Parallel, delayed
#import dill as pickle

import keras
from keras.datasets import boston_housing
from keras import backend as K

import cProfile
import pstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def varying_pretraining_boston_housing(r, initial_noise, batch_size, num_particles, num_epochs, timesteps):

    
    x_train, y_train, x_test, y_test = load_boston_housing()
    #x_train = x_train[:1000]
    #y_train = y_train[:1000]
    #x_test = x_test[:1000]
    #y_test = y_test[:1000]

    ## Full ENKF Run
    model = initialize_model_boston_housing(input_shape=x_train.shape[1:])
    wvec = extract_weight_vector(model)
    meas_model = lambda wvec, xs: predict_nn(wvec, xs, model)

    Ts = [0, 1, 3, 5, 10, 20, 100, 400, 600, 800, 1000]
    loss_function=None#"mse"

    logger = TrainingLogger()
    As = estimate_weights_enkf(wvec, x_train, y_train, x_test, y_test, dx=0.1, meas_model=meas_model, timesteps=timesteps, num_epochs=num_epochs, ensemble_size=num_particles, batch_size=batch_size, r=r, logger=logger, initial_noise=initial_noise, loss_function=loss_function, Ts=Ts, parallelize=False)
    A = As[Ts[-1]]

    #np.savetxt('ensemble.csv', A, delimiter=',')

    model.compile(loss=keras.losses.mean_squared_error,
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD(),#(learning_rate=0.001), #
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['mse', 'accuracy'])

    wvec_final = np.mean(A, 1)
    model = set_weights_from_vector(model, wvec_final)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf):', score[0])
    print('Test accuracy (enkf):', score[1])


    # train backprop from various initializations Ts
    histories = {}
    loss_function=keras.losses.mean_squared_error
    for Tsi in Ts:
        A = As[Tsi]
        wvec = np.mean(A, 1)
        history = train_backprop_from_init(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs=num_epochs, batch_size=batch_size, loss_function=loss_function, initial_weights=wvec)
        histories[Tsi] = history

    ## Plot learning curves

    fig1, (ax1) = plt.subplots(1, 1)
    
    logger.plot_errors(ax=ax1, invert=False)
    for Tsi in Ts:
        history = histories[Tsi]
        ax1.plot(range(Tsi, len(history.val_losses)), history.val_losses[:len(history.val_losses)-Tsi], linewidth=3)

   
    ax1.legend(['ENKF'] + ['t_switch='+str(Tsi) for Tsi in Ts], fontsize=13, ncol=2)
    ax1.set_ylim([0, 100])
    ax1.set_ylabel("Mean Squared Error")

    plt.title('Varying Pretraining Duration', fontsize=20)
    plt.savefig('test-acc-enkf.png', format='png', dpi=1200)

    ax1.set_ylim([0, 40])
    plt.savefig('test-acc-enkf-zoom.png', format='png', dpi=1200)


    #fig2, (ax2) = plt.subplots(1, 1)
    #logger.plot_eigenvals(ax=ax2)
    #ax1.set_xlabel("# weight updates")
    #ax1.set_ylabel("Eigenvalue magnitude")
    #plt.title("Boston Housing Eigenvalues of Prediction Covariance")
    #plt.savefig('eigenvals-enkf.png', format='png', dpi=1200)

    #plt.show()

def train_backprop_from_init(x_train, y_train, x_test, y_test, num_epochs, batch_size, loss_function, initial_weights=None):
    model = initialize_model_boston_housing()
    if initial_weights is not None:
        model = set_weights_from_vector(model, initial_weights)

    model.compile(loss=loss_function,
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD(),# (learning_rate=0.0001), #
                  #optimizer=keras.optimizers.RMSprop(),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['mae'])

    history = LossHistory(test_data=(x_test, y_test), val_interval=1)
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



def load_boston_housing():
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)

    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds

    print(x_train[0, :])

    # convert class vectors to binary class matrices
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def parse_args(args):
    # default parameters
    r=0.001
    initial_noise=0.03
    batch_size=16
    num_particles=200
    num_epochs=5
    timesteps=1001


    for arg in args:
        if "=" in arg:
            tag = arg.split("=")[0]
            value = arg.split("=")[1]
            if tag == "--r": r=float(value)
            if tag == "--initial_noise": initial_noise = float(value)
            if tag == "--batch_size": batch_size = int(value)
            if tag == "--num_particles": num_particles = int(value)
            if tag == "--num_epochs": num_epochs = int(value)
            if tag == "--timesteps": timesteps = int(value)

    return r, initial_noise, batch_size, num_particles, num_epochs, timesteps


def main(args):

    pr = cProfile.Profile()
    pr.enable()

    np.random.seed(1)

    r, initial_noise, batch_size, num_particles, num_epochs, timesteps = parse_args(args)

    varying_pretraining_boston_housing(r=r, initial_noise=initial_noise, batch_size=batch_size, num_particles=num_particles, num_epochs=num_epochs, timesteps=timesteps)

    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)


if __name__ == "__main__":
    main(sys.argv)