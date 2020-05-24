import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K


class LossHistory(keras.callbacks.Callback):
    def __init__(self, test_data, val_interval=10):
        self.test_data = test_data
        self.val_interval = val_interval

    def on_train_begin(self, logs={}):
        self.counter = 0
        self.val_accuracies = []
        self.val_losses = []

        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)    
        self.val_accuracies.append(acc)
        self.val_losses.append(loss)

    def on_batch_end(self, batch, logs={}):
        self.counter += 1
        if(self.counter % self.val_interval == 0):
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)    
            self.val_accuracies.append(acc)
            self.val_losses.append(loss)
        else:
            self.val_accuracies.append(self.val_accuracies[-1])
            self.val_losses.append(self.val_losses[-1])
       # self.val_accuracies.append(logs.get('accuracy'))
       # self.val_losses.append(logs.get('loss'))
        #print(logs.get('accuracy'))

def initialize_model(input_shape=(28, 28, 1)):
    return initialize_model_mnist(input_shape)

def initialize_model_mnist(input_shape=(28, 28, 1)):
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    #model.add(Conv2D(16, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same',))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')) # by some mistake we spend several weeks using relu activations in the last layer... this affected some results prior to 3/17

    #model.compile(loss=keras.losses.mean_squared_error,
    #model.compile(loss=keras.losses.categorical_crossentropy,
                  #optimizer=keras.optimizers.SGD(),#(learning_rate=0.001), #
    #              optimizer=keras.optimizers.Adadelta(),
    #              metrics=['accuracy'])

    return model

def initialize_model_cifar(input_shape=(32, 32, 3)):
    num_classes = 10
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def initialize_model_boston_housing(input_shape=(13,)):
    
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=input_shape))
    #model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
    #model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
    #model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
    #model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
    #model.add(Dense(64, activation='sigmoid', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    return model
