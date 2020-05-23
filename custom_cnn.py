import numpy as np
import gzip

## Returns idexes of maximum value of the array
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

def maxpool(X, f, s):
    (l, w, w) = X.shape
    pool = np.zeros((l, (w-f)/s+1,(w-f)/s+1))
    for jj in range(0,l):
        i=0
        while(i<w):
            j=0
            while(j<w):
                pool[jj,i/2,j/2] = np.max(X[jj,i:i+f,j:j+f])
                j+=s
            i+=s
    return pool

def softmax_cost(out,y):
    eout = np.exp(out, dtype=np.float)#we dont have 128 a typo fixed
    probs = eout/sum(eout)
    
    p = sum(y*probs)
    cost = -np.log(p)   ## (Only data loss. No regularised loss)
    return cost,probs   

def initialize_param(f, l):
    return 0.01*np.random.rand(l, f, f)

def initialize_theta(NUM_OUTPUT, l_in):
    return 0.01*np.random.rand(NUM_OUTPUT, l_in)

def initialise_param_lecun_normal(FILTER_SIZE, IMG_DEPTH, scale=1.0, distribution='normal'):
    
    if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)

    distribution = distribution.lower()
    if distribution not in {'normal'}:
        raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)

    scale = scale
    distribution = distribution
    fan_in = FILTER_SIZE*FILTER_SIZE*IMG_DEPTH
    scale = scale
    stddev = scale * np.sqrt(1./fan_in)
    shape = (IMG_DEPTH,FILTER_SIZE,FILTER_SIZE)
    return np.random.normal(loc = 0,scale = stddev,size = shape)


## Predict class of each row of matrix X
def predict(image, params) #filt1, filt2, bias1, bias2, theta3, bias3):
    
    ## l - channel
    ## w - size of square image
    ## l1 - No. of filters in Conv1
    ## l2 - No. of filters in Conv2
    ## w1 - size of image after conv1
    ## w2 - size of image after conv2

    filt1, filt2, bias1, bias2, theta3, bias3

    (l,w,w)=image.shape
    (l1,f,f) = filt2[0].shape
    l2 = len(filt2)
    w1 = w-f+1
    w2 = w1-f+1
    conv1 = np.zeros((l1,w1,w1))
    conv2 = np.zeros((l2,w2,w2))
    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):
                conv1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*filt1[jj])+bias1[jj]
    conv1[conv1<=0] = 0 #relu activation
    ## Calculating second Convolution layer
    for jj in range(0,l2):
        for x in range(0,w2):
            for y in range(0,w2):
                conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f]*filt2[jj])+bias2[jj]
    conv2[conv2<=0] = 0 # relu activation
    ## Pooled layer with 2*2 size and stride 2,2
    pooled_layer = maxpool(conv2, 2, 2) 
    fc1 = pooled_layer.reshape(((w2/2)*(w2/2)*l2,1))
    out = theta3.dot(fc1) + bias3   #10*1
    eout = np.exp(out, dtype=np.float)
    probs = eout/sum(eout)
    # probs = 1/(1+np.exp(-out))

    # print out
    # print np.argmax(out), np.max(out)
    return np.argmax(probs), np.max(probs)


def extract_data(filename, num_images, IMAGE_WIDTH):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels