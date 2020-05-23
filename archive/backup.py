from nn_utils import predict_nn, set_weights_from_vector, extract_weight_vector, evaluate_performance
from nn_model import initialize_model, LossHistory
from enkf_algo import estimate_weights_enkf

import numpy as np

import multiprocessing
from joblib import Parallel, delayed
#import dill as pickle

import cProfile
import pstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


timesteps=101
eigenvals_history = np.zeros((10, timesteps))

def learn_mnist_backprop_from_file():
    batch_size = 16

    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()

    # load ensemble from file
    A = np.genfromtxt('ensemble.csv', delimiter=',')

    wvec_final = np.mean(A, 1)
    model = set_weights_from_vector(model, wvec_final)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf):', score[0])
    print('Test accuracy (enkf):', score[1])

    history = LossHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])
    #for i in range(100):
    #    print("==== Batch ", i, " ====")
    #    print(model.train_on_batch(x_train[i*16:(i+1)*16], y_train[i*16:(i+1)*16]))
    #    score = model.evaluate(x_test, y_test, verbose=0)
    #    print('Test loss (enkf+backprop):', score[0])
    #    print('Test accuracy (enkf+backprop):', score[1])


    print(history.val_accuracies)

    plt.figure()
    plt.plot([i*batch_size for i in range(len(history.val_accuracies))], history.val_accuracies)
    plt.savefig('test-acc-backprop-enkf-init.svg', format='svg', dpi=1200)
    plt.show()

def eval_maximum_likelihood():
    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()

    A = np.genfromtxt('ensemble.csv', delimiter=',')

    min_acc = 1
    max_acc = 0

    for i in range(A.shape[1]):
        wvec_final = A[:, i]
        model = set_weights_from_vector(model, wvec_final) 
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss (enkf):', score[0])
        print('Test accuracy (enkf):', score[1])
        if score[1] < min_acc:
            min_acc = score[1]
        if score[1] > max_acc:
            max_acc = score[1]

    wvec_final = np.mean(A, 1)
    model = set_weights_from_vector(model, wvec_final)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf):', score[0])
    print('Test accuracy (enkf):', score[1])

    mean_acc = score[1]

    print("Max:", max_acc, "Min:", min_acc, "Mean:", mean_acc)

def visualize_distribution():
    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()

    A = np.genfromtxt('ensemble.csv', delimiter=',')

    plt.figure()
    plt.scatter(A[1000, :], A[1500, :])
    plt.show()


def learn_mnist_backprop_plus_enkf():
    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss (enkf):', score[0])
    print('Test accuracy (enkf):', score[1])

    
    ### enkf learning here
    wvec = extract_weight_vector(model)
    print(len(wvec))
    #wvec = np.random.randn(len(wvec))
    model = set_weights_from_vector(model, wvec)

    print(predict_nn(wvec, x_test, model))

    meas_model = lambda wvec, xs: predict_nn(wvec, xs, model)
    timesteps=21
    num_particles=300
    As = estimate_weights_enkf(wvec, x_train, y_train, dx=0.1, meas_model=meas_model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles)


    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss (enkf+backprop):', score[0])
    print('Test accuracy (enkf+backprop):', score[1])

    plt.show()


def learn_mnist_backprop():
    batch_size = 16

    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()
    
    wvec = extract_weight_vector(model)
    print("num parameters:", len(wvec))

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

    #print(history.val_accuracies)

    #samples_per_test = 5*16
    #plt.plot([i*samples_per_test for i in range(len(test_performance))], test_performance)
    plt.plot([i*batch_size for i in range(len(history.val_accuracies))], history.val_accuracies)
    #plt.savefig('test-acc-backprop.svg', format='svg', dpi=1200)

    #plt.show()
