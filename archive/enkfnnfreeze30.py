import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import cProfile
import pstats


def learn_mnist_backprop():
    batch_size = 128
    num_classes = 10
    epochs = 12

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

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    wvec = extract_weight_vector(model)
    print("num parameters:", len(wvec))

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def learn_mnist_enkf():
    #K.tf.compat.v1.disable_eager_execution()

    batch_size = 128
    num_classes = 10
    epochs = 12

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

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    
    ### enkf learning here
    wvec = extract_weight_vector(model)
    print(len(wvec))
    #wvec = np.random.randn(len(wvec))
    model = set_weights_from_vector(model, wvec)

    print(predict_nn(wvec, x_test, model))

    meas_model = lambda wvec, xs: predict_nn(wvec, xs, model)
    timesteps=1000
    num_particles=300
    As = fixed_interval_smooth(wvec, x_train, y_train, dx=0.1, meas_model=meas_model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles)


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def predict_nn(wvec, xs, model):
    model = set_weights_from_vector(model, wvec)
    if len(xs) == 1:
        ys_pred = model.predict(xs, batch_size=len(xs))
    else:
        ys_pred = model.predict(xs)
    return ys_pred

def set_weights_from_vector(model, wvec):
    weights = model.get_weights()
    layer_ends = np.cumsum([w.flatten().shape[0] for w in weights])
    #print(layer_ends)
    for i in range(len(weights)):
        if i == 0:
            weights[i] = np.reshape(wvec[:layer_ends[i]], weights[i].shape)
        if i >= 1:
            weights[i] = np.reshape(wvec[layer_ends[i-1]:layer_ends[i]], weights[i].shape)
        
    model.set_weights(weights)
    return model

def extract_weight_vector(model):
    weights = model.get_weights()
    for w in weights:
        print(w.shape)

    weights = np.concatenate([w.flatten() for w in weights])
    return weights

def fixed_interval_smooth(w0, xs, ys, dx, meas_model, timesteps, window, ensemble_size):
    perturb_noise = 0.03
    dt = 0.1
    perturbation = np.random.randn(len(w0), ensemble_size)
    A = np.tile(w0, (ensemble_size, 1)).T + perturbation
    Z = np.eye(ensemble_size)

    M = timesteps
    W = window
    To = range(1, M)
    Ts = range(1, 500, M)
    X5s = {}
    As = {}

    eval_period = 50
    train_performance = np.zeros((1, M // eval_period))
    test_performance = np.zeros((1, M // eval_period))

    for i in range(M):
        print("==== Iteration", i, "====")
        if i % eval_period == 0 and i > 0:
            print("Evaluating...")
            print("Train Performance: ", end='')
            train_performance[0, i // eval_period] = evaluate_performance(A, xs[:i, :, :, :], ys[:i, :], meas_model);
            print("Test Performance: ", end='')
            test_performance[0, i // eval_period] = evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model);

        X5 = np.array([])
        if i in To:
            X5 = compute_X5(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model)
            A = np.matmul(A, X5)
            #Lyerr = compute_Lyerr(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model)
            #A = A + Lyerr

        X5s[i] = X5

        if i in Ts:
            As[i] = A

        if i <= W:
            Z = MultR(Z, X5)
        else:
            Xw = X5s[i-W]
            Z = Mult(Xw, Z, X5)
            if i-W in Ts:
                Aw = As[i-W]
                Aw = np.matmul(Aw,  Z)
                As[i-W] = Aw
        print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))

    return As

def compute_X5_old(A, x, y, meas_model):
    n = y.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality
    print(n, l, q)
    H = np.concatenate((np.eye(n*q), np.matrix(np.zeros((l, n*q))))).T

    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)
    error_y = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T
    Pe = np.matmul(error_y, error_y.T)
    R = np.mean(np.abs(error_y))/2
    #print(R)
    #R = 0.1
    gamma = np.random.randn(ytilde.shape[0], 1) * R
    Re = np.matmul(gamma, gamma.T)

    print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    PRinv = np.linalg.inv(Pe+Re)
    print(np.max(PRinv))
    if np.max(np.abs(PRinv)) > 1000000000:
        print("Singularity Error")
        X5 = np.eye(A.shape[1])
    else:
        X4 = np.matmul(error_y.T, PRinv)
        #print(np.max(X4))
        X5 = np.eye(A.shape[1]) + np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T - ytilde))

    return X5

def compute_X5(A, x, y, meas_model):
    n = y.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality
    print(n, l, q)
    H = np.concatenate((np.eye(n*q), np.matrix(np.zeros((l, n*q))))).T

    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)
    error_y = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T
    #S = np.concatenate((ytilde, A))
    #error_S = S - np.tile(np.mean(S, 1), (S.shape[1], 1)).T
    #Pe = np.matmul(A.T, A)
    Pe = np.matmul(error_y, error_y.T)
    R = np.mean(np.abs(error_y))/2
    #print(R)
    #R = 0.1
    gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * R
    Re = np.matmul(gamma, gamma.T)

    print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    #PRinv = np.linalg.inv(Pe+Re)
    try:
        PRinv = np.linalg.inv(Pe + Re)
    except np.linalg.LinAlgError:
        print("Singularity Error")
        X5 = np.eye(A.shape[1])
        return X5
    
    X4 = np.matmul(error_y.T, PRinv)
    #print(np.max(X4))
    X5 = np.eye(A.shape[1]) + np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

    return X5

def compute_X5_old(A, x, y, meas_model):
    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)
    print(ytilde)
    print(y)
    print(np.mean(ytilde, 1))
    HAp = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T
    Pe = np.matmul(HAp, HAp.T)# / (A.shape[1] - 1)
    R = np.mean(np.abs(HAp))
    gamma = np.random.randn(ytilde.shape[0], 1) * R
    Re = np.matmul(gamma, gamma.T)# / (A.shape[1] - 1)

    w, v = np.linalg.eig(Pe + Re)
    lambda_inv = np.diag(w) # np.linalg.inv(np.diag(w))
    U, S, V = np.linalg.svd(Pe+Re)

    X1 = np.matmul(lambda_inv, U.T)
    X2 = np.matmul(X1, (np.tile(y, (ytilde.shape[1], 1)).T - ytilde))
    X3 = np.matmul(U, X2)
    X4 = np.matmul(HAp.T, X3)
    X5 = np.eye(X4.shape[0]) + X4

    return X5

def MultR(A, B):
    if B.shape[0] > 0:
        C = A * B
    else:
        C = A
    return C

def MultL(A, B):
    if A.shape[0] > 0:
        C = np.linalg.pinv(A) * B
    else:
        C = B
    return C

def Mult(A, B, C):
    D = MultL(A, MultR(B, C))
    return D

def evaluate_performance(A, x, y, meas_model):
    params_estimate = np.mean(A, 1)
    y_predict = meas_model(params_estimate, x)
    q1 = np.argmax(y_predict, 1)
    q2 = np.argmax(y, 1)
    successes = ((q1-q2) == 0)
    success_count = np.sum(successes)
    performance = success_count / len(successes)
    #print(np.matmul(y, successes.T)/np.sum(y, 1))
    print(performance)
    return performance


if __name__ == "__main__":
    #learn_mnist_backprop()
    pr = cProfile.Profile()
    pr.enable()
    learn_mnist_enkf()
    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)