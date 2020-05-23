import numpy as np
import keras

def predict_nn(wvec, xs, model):
    model = set_weights_from_vector(model, wvec)
    if len(xs) == 1:
        ys_pred = model.predict(xs, batch_size=len(xs))
    else:
        ys_pred = model.predict(xs)
    return ys_pred
'''
def set_weights_from_vector(model, wvec):
    #weights = model.get_weights()

    weights = model.trainable_weights
    weights = [keras.backend.eval(weight) for weight in weights]
    layer_ends = np.cumsum([w.flatten().shape[0] for w in weights])
    print([w.shape for w in weights])
    print([np.array(model.layers[i].weights).shape for i in range(len(model.layers))])
    #print(layer_ends)
    i = 0
    for j in range(len(model.layers)):
        if model.layers[j].trainable:
            print(np.concatenate([np.array(l).flatten() for l in model.layers[j].get_weights()]).flatten())
            if i == 0:
                #weights[i] = np.reshape(wvec[:layer_ends[i]], weights[i].shape)
                model.layers[j].set_weights(np.reshape(wvec[:layer_ends[i]], weights[i].shape))
            if i >= 1:
                #weights[i] = np.reshape(wvec[layer_ends[i-1]:layer_ends[i]], weights[i].shape)
                model.layers[j].set_weights(np.reshape(wvec[layer_ends[i-1]:layer_ends[i]], weights[i].shape))
            i += 1
        
    #model.set_weights(weights)
    return model

def extract_weight_vector(model):
    #weights = model.get_weights()
    #for w in weights:
    #    print(w.shape)

    weights = model.trainable_weights
    weights = [keras.backend.eval(weight) for weight in weights]
    print(weights)



    weights = np.concatenate([w.flatten() for w in weights])
    print('Number of weights:', weights.shape)
    return weights
'''

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
    #for w in weights:
    #    print(w.shape)

    weights = np.concatenate([w.flatten() for w in weights])
    return weights

def extract_weight_gradient(model, xs, ys):
    weights = model.trainable_weights # weight tensors
    #weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
    weights = [weight for weight in weights]
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     keras.backend.learning_phase(), # train or test mode
    ]

    get_gradients = keras.backend.function(inputs=input_tensors, outputs=gradients)

    inputs = [xs, # X
              [1 for i in range(xs.shape[0])], # sample weights
              ys, # y
              0 # learning phase in TEST mode
    ]

    w_gradients_trainable = [g for g in get_gradients(inputs)]
    w_gradients_trainable_flat = [g.flatten() for g in w_gradients_trainable]

    w_gradients = [w for w in model.get_weights()]
    w_gradients_flat = [w.flatten() for w in w_gradients]
    #print(np.concatenate(w_gradients).shape)
    #layer_ends = np.cumsum([w.flatten().shape[0] for w in weights])
    j = 0 # trainable layer index
    for i in range(len(weights)):
        #print(w_gradients[i].flatten().shape)
        print(w_gradients_trainable[j].shape, w_gradients[i].shape)
        if w_gradients_trainable[j].shape == w_gradients[i].shape: #matching layers
            #if j == 0:
            #w_gradients[i] = np.reshape(w_gradients_trainable[j], w_gradients[i].shape)
            w_gradients_flat[i] = w_gradients_trainable_flat[j]
            #if j >= 1:
            #    weights[i] = np.reshape(wvec[layer_ends[i-1]:layer_ends[i]], weights[i].shape)
            j += 1
        else:
            w_gradients_flat[i] = np.zeros_like(w_gradients_flat[i])
        
    #print(w_gradients)
    w_gradients_flat = np.concatenate(w_gradients_flat)
    #print(w_gradients.shape)
    print(w_gradients_flat)

    return w_gradients_flat[:, None]

def evaluate_performance(A, x, y, meas_model):
    params_estimate = np.mean(A, 1)
    y_predict = meas_model(params_estimate, x)
    print(x.shape)
    if(len(x.shape) > 2): #classification
        q1 = np.argmax(y_predict, 1)
        q2 = np.argmax(y, 1)
        successes = ((q1-q2) == 0)
        success_count = np.sum(successes)
        performance = success_count / len(successes)
        #print(y.shape, successes.shape)
        print(performance)
        print(np.matmul(y.T, successes)/np.sum(y, 0))

    else:
        error = y_predict - y.reshape(y_predict.shape)
        performance = np.mean(np.power(error, 2))
        print('MSE:', performance)

    return performance