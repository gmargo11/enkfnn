import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

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


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.val_accuracies.append(logs.get('accuracy'))
        #print(logs.get('accuracy'))

def initialize_model():
    batch_size = 16
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

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

def learn_mnist_enkf_plus_backprop():
    batch_size = 16

    model = initialize_model()
    x_train, y_train, x_test, y_test = load_mnist()

    ### enkf learning here
    wvec = extract_weight_vector(model)
    print(len(wvec))
    #wvec = np.random.randn(len(wvec))
    model = set_weights_from_vector(model, wvec)

    print(predict_nn(wvec, x_test, model))

    meas_model = lambda wvec, xs: predict_nn(wvec, xs, model)
    timesteps=1001
    num_particles=200
    As, train_performance, test_performance = fixed_interval_smooth_batch(wvec, x_train, y_train, dx=0.1, meas_model=meas_model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles, batch_size=16, parallel=False)
    #As = fixed_interval_smooth(wvec, x_train, y_train, dx=0.1, meas_model=meas_model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles)

    A_final = As[timesteps-1]
    #np.savetxt('ensemble.csv', A_final, delimiter=',')

    wvec_final = np.mean(A_final, 1)
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
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss (enkf+backprop):', score[0])
    print('Test accuracy (enkf+backprop):', score[1])

    #print(history.val_accuracies)

    samples_per_test = 5*16
    plt.plot([i*samples_per_test for i in range(len(test_performance))], test_performance)
    plt.plot([i*batch_size+len(test_performance)*samples_per_test for i in range(len(history.val_accuracies))], history.val_accuracies)
    plt.savefig('test-acc-enkf.svg', format='svg', dpi=1200)

    #plt.show()

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
    As = fixed_interval_smooth(wvec, x_train, y_train, dx=0.1, meas_model=meas_model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles)


    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss (enkf+backprop):', score[0])
    print('Test accuracy (enkf+backprop):', score[1])

    plt.show()

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
    perturb_noise = 0.3
    dt = 0.1
    perturbation = np.random.randn(len(w0), ensemble_size) * perturb_noise
    A = np.tile(w0, (ensemble_size, 1)).T + perturbation
    Z = np.eye(ensemble_size)

    M = timesteps
    W = window
    To = range(1, M)
    Ts = range(1, M)
    X5s = {}
    As = {}

    eval_period = 10
    train_performance = np.zeros(M // eval_period)
    test_performance = np.zeros(M // eval_period)

    for i in range(M):
        print("==== Iteration", i, "====")
        if i % eval_period == 0 and i > 0:
            print("Evaluating...")
            print("Train Performance: ", end='')
            train_performance[i // eval_period] = evaluate_performance(A, xs[:i, :, :, :], ys[:i, :], meas_model);
            print("Test Performance: ", end='')
            test_performance[i // eval_period] = evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model);

        X5 = np.array([])
        if i in To:
            X5 = compute_X5(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model)
            A += X5 #np.matmul(A, X5)
            #Lyerr = compute_Lyerr(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model)
            #A = A + Lyerr

        X5s[i] = X5

        if i in Ts:
            As[i] = A

        #if i <= W:
        #    Z = MultR(Z, X5)
        #else:
        #    Xw = X5s[i-W]
        #    Z = Mult(Xw, Z, X5)
        #    if i-W in Ts:
        #        Aw = As[i-W]
        #        Aw = np.matmul(Aw,  Z)
        #        As[i-W] = Aw
        print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))

    plt.figure()
    plt.plot(np.array(range(len(train_performance))) * eval_period, train_performance)
    plt.plot(np.array(range(len(test_performance))) * eval_period, test_performance)

    return As


def fixed_interval_smooth_batch(w0, xs, ys, dx, meas_model, timesteps, window, ensemble_size, batch_size, parallel=True):
    perturb_noise = 0.3
    dt = 0.1
    perturbation = np.random.randn(len(w0), ensemble_size) * perturb_noise
    A = np.tile(w0, (ensemble_size, 1)).T + perturbation
    Z = np.eye(ensemble_size)

    M = timesteps
    W = window
    To = range(1, M)
    Ts = range(1, M)
    X5s = {}
    As = {}

    eval_period = 5
    train_performance = np.zeros(M // eval_period+1)
    test_performance = np.zeros(M // eval_period+1)


    for i in range(M):
        print("==== Iteration", i, "====")
        if i % eval_period == 0 and i > 0:
            print("Evaluating...")
            print("Train Performance: ", end='')
            train_performance[i // eval_period] = evaluate_performance(A, xs[:i*batch_size, :, :, :], ys[:i*batch_size, :], meas_model);
            print("Test Performance: ", end='')
            test_performance[i // eval_period] = evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model);

        X5 = np.array([])
        print("Training Sample: ", end='')


        if parallel and i in To:
            inputs = np.array(range(batch_size)) + i * batch_size
            #def process(inp):
            #    compute_X5(A, xs[inp:inp+1, :, :, :], ys[inp:inp+1, :], meas_model)

            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(compute_X5)(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model) for i in inputs)

            print(len(results))

            X5 = np.mean(results, axis=0)
            print(X5.shape)
        elif i in To:
            for j in range(batch_size): # compute batch X5
                print(j, " ", end='')
                idx = i * batch_size + j
                if X5.shape[0] == 0:
                    X5 = compute_X5(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model, i)
                    #X5 = compute_X5_dropout(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model)
                else:
                    X5 = X5 + compute_X5(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model, i)
                    #X5 = X5 + compute_X5_dropout(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model)
            print()
            X5 = X5 / batch_size

        if i in To:
            A += X5 #np.matmul(A, X5)
            #mean = np.mean(A, 1)
            #x = np.random.randn(A.shape[0], A.shape[1])
            #std = np.std(A, 1)
            #std = std / np.mean(std) / 30
            #x = np.multiply(x, np.tile(std, (x.shape[1], 1)).T)
            #print(x.shape)
            #A = np.tile(mean, (x.shape[1], 1)).T + x
            #print(A.shape)
        X5s[i] = X5

        if i in Ts:
            As[i] = A

        #if i <= W:
        #    Z = MultR(Z, X5)
        #else:
        #    Xw = X5s[i-W]
        #    Z = Mult(Xw, Z, X5)
        #    if i-W in Ts:
        #        Aw = As[i-W]
        #        Aw = np.matmul(Aw,  Z)
        #        As[i-W] = Aw
        print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))

    #plt.figure()
    #plt.plot(np.array(range(len(train_performance))) * eval_period, train_performance)
    #plt.plot(np.array(range(len(test_performance))) * eval_period, test_performance)

    return As, train_performance, test_performance


def compute_X5(A, x, y, meas_model, t):
    n = y.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality

    #errmag = max(1/t, 0.00001)
    errmag = 0.001
    gamma = np.random.randn(y.shape[1], A.shape[1]) * errmag


    #print(n, l, q)
    #H = np.concatenate((np.eye(n*q), np.matrix(np.zeros((l, n*q))))).T

    ytilde = np.zeros((y.shape[1], A.shape[1]))
    yerr = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)

    print(ytilde.shape)

    ytilde_mean = np.mean(ytilde, 1)
    #P_yy = np.zeros((ytilde.shape[0], ytilde.shape[0]))
    #for i in range(A.shape[1]):
    #    yerr = ytilde[:, i] - ytilde_mean
    #    P_yy += np.outer(yerr, yerr)

    A_mean = np.mean(A, 1)
    Aerr = np.zeros((A.shape[0], A.shape[1]))
    P_xy = 0
    for i in range(A.shape[1]):
        Aerr[:, i] = A[:, i] - A_mean
        yerr[:, i] = ytilde[:, i] - ytilde_mean
        #print(Aerr.shape, yerr.shape)
        P_xy += np.outer(Aerr[:, i], yerr[:, i])

    yerr = yerr + gamma

    #P_xy = P_xy / A.shape[1]

    #print(P_yy)

    #w, v = np.linalg.eig(P_yy / A.shape[1])
    #errmag = np.sort(w)[2]
    u, s, vh = np.linalg.svd(yerr)

    #if(np.sort(s)[5] < errmag**2 / 100):
    #    print("inflate covariance!!")
    s = s  / np.mean(s) * errmag**2

    smat = np.zeros((u.shape[1], u.shape[1]), dtype=float)
    smat[:s.shape[0], :s.shape[0]] = np.diag(s)
    P_yy = np.dot(u, np.dot(smat, np.dot(smat, u.T))) / A.shape[1]

    smat2 = np.zeros((vh.shape[1], u.shape[1]), dtype=float)
    smat2[:s.shape[0], :s.shape[0]] = np.diag(s)
    P_xy = np.dot(Aerr, np.dot(vh, np.dot(smat2, u.T))) / A.shape[1]
    #print(P_yy)

    R = np.eye(y.shape[1]) * errmag**2
    print(errmag)
    print(s)

    eigenvals_history[:, t] = s


    #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    #PRinv = np.linalg.inv(Pe+Re)
    try:
        PRinv = np.linalg.inv(P_yy + R)
    except np.linalg.LinAlgError:
        print("Singularity Error")
        X5 = np.zeros_like(A)
        return X5
    #if np.max(PRinv) > 1000000000:
    #    print("Exploding Value Error")
    #    X5 = np.zeros_like(A)
    #    return X5
    
    X4 = np.matmul(P_xy, PRinv)

    update = np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

    return update


def compute_X5_dropout(A, x, y, meas_model):
    # dropout rate 0.25
    A_dropout = np.copy(A)
    #for i in range(A.shape[1]):
    indices = np.random.choice(A.shape[0], replace=False, size=int(A.shape[0]*0.10))
    A_dropout[indices, :] = 0


    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A_dropout[:, i], x)

    print(ytilde.shape)

    ytilde_mean = np.mean(ytilde, 1)
    P_yy = np.zeros((ytilde.shape[0], ytilde.shape[0]))
    for i in range(A.shape[1]):
        yerr = ytilde[:, i] - ytilde_mean
        P_yy += np.outer(yerr, yerr)

    #w, v = np.linalg.eig(P_yy / A.shape[1])
    #errmag = np.sort(w)[-1]/2
    errmag = 100
    R = np.eye(y.shape[1]) * errmag
    print(errmag)

    P_yy = P_yy / A.shape[1] + R

    A_mean = np.mean(A_dropout, 1)
    P_xy = 0
    for i in range(A.shape[1]):
        Aerr = A_dropout[:, i] - A_mean
        yerr = ytilde[:, i] - ytilde_mean
        #print(Aerr.shape, yerr.shape)
        P_xy += np.outer(Aerr, yerr)

    P_xy = P_xy / A.shape[1]

    #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    #PRinv = np.linalg.inv(Pe+Re)
    try:
        PRinv = np.linalg.inv(P_yy)
    except np.linalg.LinAlgError:
        print("Singularity Error")
        X5 = np.zeros_like(A)
        return X5
    #if np.max(PRinv) > 1000000000:
    #    print("Exploding Value Error")
    #    X5 = np.zeros_like(A)
    #    return X5
    
    X4 = np.matmul(P_xy, PRinv)

    gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * errmag
    update = np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

    return update

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
    #learn_mnist_backprop_from_file()
    learn_mnist_enkf_plus_backprop()
    #learn_mnist_backprop()
    #learn_mnist_enkf_plus_backprop()
    #eval_maximum_likelihood()
    #visualize_distribution()
    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)