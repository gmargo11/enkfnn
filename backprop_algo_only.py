from nn_utils import predict_nn, set_weights_from_vector, extract_weight_vector, evaluate_performance, extract_weight_gradient

import numpy as np

import multiprocessing
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
import tensorflow as tf


def estimate_weights_enkf_augmented(model, w0, xs, ys, x_test, y_test, dx, meas_model, timesteps, num_epochs, ensemble_size, batch_size, r, logger, adapt_r="None", loss_function="mse", initial_noise=0.03, Ts=None, save_interval=100, parallelize=False):
    perturb_noise = initial_noise
    dt = 0.1
    perturbation = np.random.randn(len(w0), ensemble_size) * perturb_noise
    print(perturb_noise)
    A = np.tile(w0, (ensemble_size, 1)).T + perturbation
    Z = np.eye(ensemble_size)
    r_max = r

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=0.0000001))


    To = range(timesteps)

    if Ts is None:
        Ts = [timesteps-1]
    #X5s = {}
    As = {}

    eval_period = 25
    #train_performance = np.zeros(M // eval_period+1)
    #test_performance = np.zeros(M // eval_period+1)
    parallel = None
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        print("Cores: ", num_cores)
        parallel = Parallel(n_jobs = num_cores, prefer="threads")

    for epoch in range(num_epochs):
        for i in range(timesteps):
            print("==== Iteration", i, "====")
            if i % eval_period == 0:
                print("Evaluating...")
                print("Train Performance: ", end='')
                #train_performance[i // eval_period] = evaluate_performance(A, xs[:i*batch_size, :, :, :], ys[:i*batch_size, :], meas_model);
                logger.log_train_acc(evaluate_performance(A, xs, ys, meas_model), i + epoch*timesteps)
                print("Test Performance: ", end='')
                #test_performance[i // eval_period] = evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model);
                logger.log_val_acc(evaluate_performance(A, x_test, y_test, meas_model), i + epoch*timesteps)

            if i % save_interval == 0:
                filename = "particles_epoch" + str(epoch) + "_t" + str(i) + ".npy"
                np.save(filename, A)
                log_file = "log_epoch" + str(epoch) + "_t" + str(i) + ".p"
                logger.save_log(filename)

            X5 = np.array([])   
            print("Training Sample: ", end='')


            num_updates = 0
            
            #for j in range(batch_size): # compute batch X5
            #    print(j, " ", end='')
            #    idx = i * batch_size + j
            #    if X5.shape[0] == 0:
            #        X5i, noise_flag = compute_ensemble_update(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model, i, r, logger)
            #        if not noise_flag:
            #            X5 = X5i
            #            num_updates = 1
            #    else:
            #        X5i, noise_flag = compute_ensemble_update(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model, i, r, logger)
            #        if not noise_flag:
            #            X5 = X5 + X5i
            #            num_updates += 1
            '''
            ytildes = compute_ytildes(A, xs[i*batch_size:(i+1)*batch_size, ...], ys[i*batch_size:(i+1)*batch_size, ...], meas_model, i, r, logger)
            
        
            if adapt_r == "Annealing":
                r = r * 0.9
            if adapt_r == "Deviation-proportional":
                #print(ytildes - ys[i*batch_size:(i+1)*batch_size, ..., None])
                #yerrs = ytildes - ys[i*batch_size:(i+1)*batch_size, ..., None]
                #for j in range(yerrs.shape[2]):
                #    yerrs[..., j][ys[i*batch_size:(i+1)*batch_size, ...]==0] = 0
                #ysqerrs = np.square(yerrs)
                #variances = np.mean(ysqerrs, axis=2)
                #stds = np.sqrt(variances)
                #print(stds)
                #r_adaptive = np.mean(stds)/2 * 10
                #if r_adaptive < r_max:
                #    r = r_adaptive
                #else:
                #    r = r_max
                r_adaptive = 0
                for j in range(batch_size):
                    #stds = np.sqrt(np.mean(np.square(ytildes[j, ...] - ys[i*batch_size + j, ..., None]), 1))        
                    stds = np.std(ytildes[j, ...], 1)
                    #print('stds', stds, 'error', np.mean(ytildes[j, ...], 1) - ys[i*batch_size + j, ...])
                    r_adaptive += np.mean(stds)#np.dot(stds, ys[i*batch_size + j, ...])
                    #r_adaptive += np.sum(stds)
                   #print('r: ', r)
                r_adaptive = r_adaptive / batch_size
                #r_adaptive = max(r, 0.1)
                r = r_adaptive / 2

                print(r_adaptive, r_max)
                print(r)


            if parallelize:
                inputs = range(batch_size)
                
                #print(num_cores)
                results = parallel(delayed(compute_ensemble_update)(A, ys[i * batch_size + j, ...], ytildes[j, ...], i, r, loss_function, logger) for j in inputs)
                X5 = np.array([result[0] for result in results])
                X5 = np.mean(X5, axis=0)
                num_updates = batch_size

            else:         
                for j in range(batch_size):
                    idx = i * batch_size + j
                    print(j, " ", end='')
                    #ytildes = compute_ytildes(A, xs[None, idx, :, :, :], ys[None, idx, :], meas_model, i, r, logger)
                    X5i, noise_flag = compute_ensemble_update(A, ys[idx, ...], ytildes[j, ...], i+epoch*timesteps, r, loss_function, logger)
                    #X5i, noise_flag = compute_ensemble_update(A, ys[idx, :], ytildes[0, :, :], i, r, logger)
                    if X5.shape[0] == 0:
                        if not noise_flag:
                            X5 = X5i
                            num_updates += 1
                    else:
                        if not noise_flag:
                            X5 = X5 + X5i
                            num_updates += 1

                print()
                if num_updates > 0:
                    X5 = X5 / num_updates
                #X5 = X5 / batch_size

            #if num_updates > 0:
            #    A += X5

            #X5s[i] = X5

            if i+epoch*timesteps in Ts:
                As[i+epoch*timesteps] = np.copy(A)

           
            print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))
            '''


            # take a SGD update step
            #for i in range(A.shape[1]):
            '''
            wvec = np.mean(A, 1)
            model = set_weights_from_vector(model, wvec)
            model.fit(xs[i*batch_size:(i+1)*batch_size, ...], 
                      ys[i*batch_size:(i+1)*batch_size, ...], 
                      batch_size=batch_size,
                      epochs=1,
                      verbose=0)
            '''
            w = np.mean(A, axis=1)#extract_weight_vector(model)
            model = set_weights_from_vector(model, w)
            w_gradients = extract_weight_gradient(model, xs[i*batch_size:(i+1)*batch_size, ...], ys[i*batch_size:(i+1)*batch_size, ...])

            #A = A - wvec[:, None] + extract_weight_vector(model)[:, None]
            #print("SGDUpdateMax", np.max(extract_weight_vector(model)[:, None] - wvec[:, None]))
            learning_rate = 0.01
            #A = A + learning_rate * w_gradients
            SGDUpdate = -1 * learning_rate * w_gradients
            ENKFUpdate = np.mean(X5, axis=1)[:, None]

            print("SGDUpdateMax", np.max(SGDUpdate), "ENKFUpdateMax", np.max(ENKFUpdate))
            print("SGDUpdateNorm", np.linalg.norm(SGDUpdate), "ENKFUpdateNorm", np.linalg.norm(ENKFUpdate))

            print(SGDUpdate.shape, ENKFUpdate.shape)

            print("Cosine", np.matmul(SGDUpdate.T, ENKFUpdate) / (np.linalg.norm(SGDUpdate) * np.linalg.norm(ENKFUpdate)))


            A += SGDUpdate #/ np.linalg.norm(SGDUpdate) * np.linalg.norm(ENKFUpdate) 




    #plt.figure()
    #plt.plot(np.array(range(len(train_performance))) * eval_period, train_performance)
    #plt.plot(np.array(range(len(test_performance))) * eval_period, test_performance)

    return As


def compute_ytildes(A, x, y, meas_model, t, r, logger):
    n = y.shape[0] # batch size
    p = A.shape[1] # number of particles
    if len(y.shape) > 1:
        q = y.shape[1] # output dimensionality
    else:
        q = 1

    ytildes = np.zeros((n, q, p))
    for i in range(p):
        ytildes[:, :, i] = meas_model(A[:, i], x)
        if(np.any(np.isnan(ytildes[:, :, i]))):
            print("NAN ALERT!!")
            print(np.max(A[:, i]), np.min(A[:, i]))
    print('ytildes', np.max(ytildes), np.min(ytildes))
    #ytildes = np.maximum(ytildes, r/10)
    #ytildes = np.minimum(ytildes, 1-r)

    return ytildes


def compute_ensemble_update(A, y, ytilde, t, r, loss_function, logger):
    n = ytilde.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = ytilde.shape[1] # output dimensionality

    #ytilde = np.zeros((y.shape[1], A.shape[1]))
    #for i in range(A.shape[1]):
    #    ytilde[:, i] = meas_model(A[:, i], x)

    #stds = np.std(ytilde, 1)
    #print('stds', stds)
    #r = np.dot(stds, y)
    #print('r: ', r)

    
    #error_y = ytilde - np.mean(ytilde, 1)[:, None] + gamma
   #gamma[ytilde==0] = 0 # evaluate; does this have significant impact?
    error_A = A - np.mean(A, 1)[:, None]
    
    if loss_function is None:
        # MSE Kalman Filter Style
        gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * r
        z = y[..., None] + gamma - ytilde
        dz = ytilde - np.mean(ytilde, 1)[:, None] + gamma
    
    elif loss_function == "mse":
        # MSE taylor expansion style
        gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * r
        #print(ytilde.shape)
        zwdw = np.square(y[..., None] - ytilde)
        zw = np.square(y[..., None] - np.mean(ytilde, 1)[:, None])
        #print('zw', zw)
        #print('zwdw', zwdw)
        #print(zw.shape, zwdw.shape)
        dy = ytilde - np.mean(ytilde, 1)[:, None]# + gamma
        z = np.divide(zwdw - zw, dy) + gamma
        #print("taylor:", z)
        #z2 = y[..., None] - ytilde
        #print("analytical", z2)
        dz =  np.mean(z, 1)[:, None] - z + gamma
        #dz = ytilde - np.mean(ytilde, 1)[:, None] + gamma

    elif loss_function == "categorical_crossentropy":
        gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * r
        #z = np.matmul(y[..., None].T, -np.log(ytilde)) + gamma # + np.zeros_like(ytilde)
        #print(y[..., None] + np.zeros_like(ytilde), ytilde)
        #dL = - np.divide(y[..., None] + np.zeros_like(ytilde), ytilde)
        #err = y[..., None] + gamma - ytilde
        print(np.any(np.isnan(y)), np.any(np.isnan(ytilde)))
        zwdw = np.multiply(y[..., None] + np.zeros_like(ytilde), -np.log(ytilde))
        zw = np.multiply(y[..., None] + np.zeros_like(ytilde), -np.log(np.mean(ytilde, 1)[:, None] + np.zeros_like(ytilde)))
        #zw = np.matmul(y[..., None].T, -np.log(np.mean(ytilde, 1)[:, None] + np.zeros_like(ytilde)))
        #zwdw = np.matmul(y[..., None].T, -np.log(ytilde))
        dy = ytilde - np.mean(ytilde, 1)[:, None]# + gamma
        #print('dy,', dy)
        #print('dy+gamma', dy+gamma)
        z =  np.divide(zwdw - zw, dy) + gamma #[:, None] + gamma
        dz = np.mean(z, 1)[:, None] - z + gamma
        dz = np.minimum(dz, 1000)
        z2 = - np.divide(y[..., None] + np.zeros_like(ytilde), ytilde)
        #print('approx', z, 'analytical', z2)
        print('dz', np.max(dz))
        print(np.any(np.isnan(zwdw)), np.any(np.isnan(zw)), np.any(np.isnan(dy)), np.any(np.isnan(z)), np.any(np.isnan(dz)))
        #print(dz)
#
    Czz = np.matmul(dz, dz.T) / A.shape[1]
    Cwz = np.matmul(error_A, dz.T) / A.shape[1]
    Re = r**2 * np.eye(dz.shape[0])
    #dL = np.matmul(err, np.matmul(err.T, L))
    #print(np.linalg.eig(Czz)[0])
    X5 = np.matmul(np.matmul(Cwz, np.linalg.inv(Czz+Re)), z)
    noise_flag = np.isnan(np.max(dz))

        

    return X5, noise_flag

def compute_ensemble_update_mse(A, y, ytilde, t, r, loss_function, logger):
    n = ytilde.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = ytilde.shape[1] # output dimensionality

    #ytilde = np.zeros((y.shape[1], A.shape[1]))
    #for i in range(A.shape[1]):
    #    ytilde[:, i] = meas_model(A[:, i], x)

    #stds = np.std(ytilde, 1)
    #print('stds', stds)
    #r = np.dot(stds, y)
    #print('r: ', r)

    
    #error_y = ytilde - np.mean(ytilde, 1)[:, None] + gamma
   #gamma[ytilde==0] = 0 # evaluate; does this have significant impact?
    error_A = A - np.mean(A, 1)[:, None]
    
    Pe = np.matmul(error_y, error_y.T) / A.shape[1]
    Re = r**2 * np.eye(error_y.shape[0]) #np.matmul(gamma, gamma.T)

    w, v = np.linalg.eig(Pe)
    #print(np.sort(w))

    noise_flag = False
    #if np.sort(w)[5] < r**2 * 1.5:
    #    noise_flag = True

    logger.log_eigenvals(np.sort(w), t)
    #wR, vR = np.linalg.eig(Re)
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
    #print(y.shape, ytilde.shape)
    #print(y)

    dL = y[..., None] + gamma - ytilde
    X5 = np.matmul(X4, dL)
    
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
