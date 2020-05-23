from nn_utils import predict_nn, set_weights_from_vector, extract_weight_vector, evaluate_performance

import numpy as np

import multiprocessing
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def estimate_weights_enkf(w0, xs, ys, dx, meas_model, timesteps, window, ensemble_size, batch_size, r, logger, parallel=False):
    perturb_noise = 0.03
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
    #train_performance = np.zeros(M // eval_period+1)
    #test_performance = np.zeros(M // eval_period+1)

    for i in range(M):
        print("==== Iteration", i, "====")
        if i % eval_period == 0 and i > 0:
            print("Evaluating...")
            print("Train Performance: ", end='')
            #train_performance[i // eval_period] = evaluate_performance(A, xs[:i*batch_size, :, :, :], ys[:i*batch_size, :], meas_model);
            logger.log_train_acc(evaluate_performance(A, xs[:i*batch_size, :, :, :], ys[:i*batch_size, :], meas_model), i)
            print("Test Performance: ", end='')
            #test_performance[i // eval_period] = evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model);
            logger.log_val_acc(evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model), i)

        X5 = np.array([])
        print("Training Sample: ", end='')


        num_updates = 0
        if i in To:
            for j in range(batch_size): # compute batch X5
                print(j, " ", end='')
                idx = i * batch_size + j
                if X5.shape[0] == 0:
                    X5i, noise_flag = compute_ensemble_update(A, xs[idx, :, :, :], ys[idx, :], meas_model, i, r, logger)
                    if not noise_flag:
                        X5 = X5i
                        num_updates = 1
                else:
                    X5i, noise_flag = compute_ensemble_update(A, xs[idx, :, :, :], ys[idx, :], meas_model, i, r, logger)
                    if not noise_flag:
                        X5 = X5 + X5i
                        num_updates += 1
            print()
            if num_updates > 0:
                X5 = X5 / num_updates
            #X5 = X5 / batch_size

        if i in To and num_updates > 0:
            A += X5

        X5s[i] = X5

        if i in Ts:
            As[i] = A

       
        print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))

    #plt.figure()
    #plt.plot(np.array(range(len(train_performance))) * eval_period, train_performance)
    #plt.plot(np.array(range(len(test_performance))) * eval_period, test_performance)

    return As

def compute_ensemble_update(A, x, y, meas_model, t, r, logger):
    n = y.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality

    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)

    gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * r
    error_y = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T + gamma
    error_A = A - np.tile(np.mean(A, 1), (A.shape[1], 1)).T
    #print(error_y[:, 0])
    #print(error_y.shape)
    #S = np.concatenate((ytilde, A))
    #error_S = S - np.tile(np.mean(S, 1), (S.shape[1], 1)).T
    #Pe = np.matmul(A.T, A)
    Pe = np.matmul(error_y, error_y.T) / A.shape[1]

    Re = r**2 * np.eye(ytilde.shape[0]) #np.matmul(gamma, gamma.T)

    w, v = np.linalg.eig(Pe)
    #print(np.sort(w))

    noise_flag = False
    #if np.sort(w)[5] < r**2 * 1.5:
    #    noise_flag = True

    logger.log_eigenvals(np.sort(w), t)
    wR, vR = np.linalg.eig(Re)
    #print(np.sort(wR))

    #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    #PRinv = np.linalg.inv(Pe+Re)
    try:
        PRinv = np.linalg.inv(Pe + Re)
    except np.linalg.LinAlgError:
        print("Singularity Error")
        X5 = np.eye(A.shape[1])
        return X5
    
    X4 = np.matmul(np.matmul(error_A, error_y.T) / A.shape[1], PRinv)
    #print(np.max(X4))
    X5 = np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

    return X5, noise_flag


# def compute_ensemble_update(Ap, x, y, meas_model, t, r, logger, dropout=False):
#     A = np.copy(Ap)

#     if dropout:
#         dropout_rate=0.10
#         indices = np.random.choice(A.shape[0], replace=False, size=int(A.shape[0]*dropout_rate))
#         A[indices, :] = 0

#     n = A.shape[1] # batch size
#     l = A.shape[0] # number of weights
#     q = y.shape[1] # output dimensionality

#     #r = max(1/t, 0.00001)
#     #r = 0.001
#     gamma = np.random.randn(q, n) * r

#     ytilde = np.zeros((q, n))
#     for i in range(n):
#         ytilde[:, i] = meas_model(A[:, i], x)

#     A_mean = np.mean(A, 1)
#     ytilde_mean = np.mean(ytilde, 1)

#     Aerr = np.zeros((l, n))
#     yerr = np.zeros((q, n))
#     for i in range(n):
#         Aerr[:, i] = A[:, i] - A_mean
#         yerr[:, i] = ytilde[:, i] - ytilde_mean

#     yerr = yerr + gamma

#     #w, v = np.linalg.eig(P_yy / A.shape[1])
#     #r = np.sort(w)[2]
#     u, s, vh = np.linalg.svd(yerr)

#     #if(np.sort(s)[5] < r**2 / 100):
#     #    print("inflate covariance!!")
#     #s = s  / np.mean(s) * r**2

#     smat = np.zeros((u.shape[1], u.shape[1]), dtype=float)
#     smat[:s.shape[0], :s.shape[0]] = np.diag(s)
#     P_yy = np.dot(u, np.dot(smat, np.dot(smat, u.T))) / n

#     smat2 = np.zeros((vh.shape[1], u.shape[1]), dtype=float)
#     smat2[:s.shape[0], :s.shape[0]] = np.diag(s)
#     P_xy = np.dot(Aerr, np.dot(vh, np.dot(smat2, u.T))) / n

#     R = np.eye(y.shape[1]) * r**2
#     #R = np.matmul(gamma, gamma.T)
#     print(r)
#     #print(s)
#     w, v = np.linalg.eig(P_yy)
#     print(np.sort(w))
#     logger.log_eigenvals(np.sort(w), t)
#     wR, vR = np.linalg.eig(R)
#     print(np.sort(wR))

#     #eigenvals_history[:, t] = s


#     #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

#     try:
#         PRinv = np.linalg.inv(P_yy + R)
#     except np.linalg.LinAlgError:
#         print("Singularity Error")
#         X5 = np.zeros_like(A)
#         return X5
#     #if np.max(PRinv) > 1000000000:
#     #    print("Exploding Value Error")
#     #    X5 = np.zeros_like(A)
#     #    return X5
    
#     X4 = np.matmul(P_xy, PRinv)

#     print(ytilde[:, 0], ytilde[:, 1], y)

#     update = np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

#     return update