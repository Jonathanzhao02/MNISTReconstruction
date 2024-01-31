"""network3.py
~~~~~~~~~~~~~~
A Theano-based program for training and running simple neural
networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.
Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).
Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

Edited by Jonathan Zhao.
Renamed to project2.py.
Updated a few functions to use Python 3.8 and Theano 1.0.0.
Added different optimizers such as momentum and Nesterov-accelerated gradient.
Changed load_data_shared to create 7x7 compressions of the 28x28 images for input.
The most efficient parameters I found are at the bottom, leading to a loss of ~0.0106.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool
from PIL import Image

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Constants
GPU = False
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "project2.py\nto set the GPU flag to False.")
    try: theano.config.device = 'cuda'
    except: pass # it's already set
else:
    print("Running with a CPU.  If this is not desired, then the modify "+\
        "project2.py to set\nthe GPU flag to True.")


# Cast all float values to float32
theano.config.floatX = 'float32'

#### Load the MNIST data
def load_data_shared(filename="./data/mnist.pkl.gz"):
    sensing_matrix = np.random.normal(scale=1/49, size=(49, 784))
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    def transformed(data):
        """Transform the data using the sensing matrix and replace labels with
        the original data.
        """
        def matmul(x):
            return np.matmul(sensing_matrix, x)
        return (np.apply_along_axis(matmul, axis=1, arr=data[0]), data[0])
    def shared(data):
        """Transform and place the data into shared variables. This allows Theano
        to copy the data to the GPU, if one is available, and provides the labels
        and outputs for reconstruction.
        """
        data = transformed(data)
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, theano.config.floatX)
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in np.arange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def momentum(self, cost, eta, args):
        """SGD with momentum, requires gamma as an argument for momentum."""
        assert args['gamma']
        updates = []
        gamma = args['gamma']
        for param in self.params:
            param_update = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)
            updates.append((param, param - eta * param_update))
            updates.append((param_update, param_update * (1 - gamma) + gamma * T.grad(cost, param)))
        return updates

    def sgd(self, cost, eta, args):
        """Standard gradient descent optimizer, no extra arguments"""
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                    for param, grad in zip(self.params, grads)]
        return updates

    def train(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, optimizer=sgd, optimizer_args={}, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        updates = optimizer(self, cost, eta, optimizer_args)

        # define functions to train a mini-batch, and to compute the
        # loss in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_loss = theano.function(
            [i], self.layers[-1].cost(self),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_loss = theano.function(
            [i], self.layers[-1].cost(self),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        
        # Do the actual training
        iteration = 0
        best_validation_loss = 1e6
        losses = np.zeros(shape=(3,epochs))
        print("Starting training...")
        for epoch in np.arange(epochs):
            costs = np.zeros(num_training_batches)
            for minibatch_index in np.random.permutation(num_training_batches): # shuffles the order of mini-batches
                iteration = iteration + 1
                costs[minibatch_index] = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_loss = np.mean(
                        [validate_mb_loss(j) for j in np.arange(num_validation_batches)])
                    print("Epoch: " + str(epoch) + " Validation loss: " + str(validation_loss))
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_iteration = iteration
                        if test_data:
                            test_loss = np.mean(
                                [test_mb_loss(j) for j in np.arange(num_test_batches)])
                            losses[2,epoch] = test_loss
                    losses[0,epoch] = np.mean(costs)
                    losses[1,epoch] = validation_loss
        print("Finished training network.")
        print("Best validation loss of {0:.8} obtained at iteration {1}".format(
            best_validation_loss, best_iteration))
        print("Corresponding test loss of {0:.8}".format(test_loss))
        return (losses, best_iteration)

#### Define layer types
# Deleted unused ConvPoolLayer & SoftmaxLayer objects
class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def cost(self, net):
        "Return the MSE loss."
        return T.mean((self.output_dropout - net.y) ** 2)

class ConvTransposeLayer(object):
    """Used to create a simple implementation of a
    transposed 2D convolutional layer for upscaling.
    """

    def __init__(self, filters, kernel, channels, mini_batch_size, input_shape,
                 output_shape=None, activation_fn=sigmoid):
        """
        `filters` is the number of filters to apply.
        `kernel` is a tuple of length 2 that specifies the size of each filter.
        `channels` is the number of input feature maps.
        `mini_batch_size` is the number of samples in a minibatch.
        `input_shape` is a tuple of length 2 that specifies the 2D dimensions of the input.
        `output_shape` is an optional tuple of length 2 that specifies the 2D dimensions of the output.
        Can be specified to check that the output shape is what the user thinks it should be.
        `activation_fn` is an optional function to be used as the activation function in this layer.
        """
        self.filter_shape = (channels, filters, kernel[0], kernel[1])
        self.image_shape = (mini_batch_size, channels, input_shape[0], input_shape[1])
        if output_shape is not None:
            self.output_shape = (mini_batch_size, filters, output_shape[0], output_shape[1])
        else:
            # Calculates the output shape automatically if unspecified
            self.output_shape = (mini_batch_size, filters,
                                 input_shape[0] + kernel[0] - 2,
                                 input_shape[1] + kernel[1] - 2)
        self.activation_fn = activation_fn
        n_out = (channels * np.prod(kernel))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=self.filter_shape),
                dtype=theano.config.floatX
            ), borrow=True
        )
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=output_shape),
                dtype=theano.config.floatX
            ), borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = nnet.conv2d_transpose(input=self.inpt, filters=self.w, output_shape=self.output_shape, input_dilation=self.input_dilation)
        self.output = self.activation_fn(
            conv_out + self.b)
        self.output_dropout = self.output # no dropout in the convolutional layers

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].shape.eval()[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)


# Network parameters
hidden_neurons = 32
epochs = 1000
mini_batch_size = 10
lr = 0.5
lmbda = 0.0001
gamma = 0.9

# Get all MNIST inputs and labels
training_data, validation_data, testing_data = load_data_shared()

# Create and train the network
# net = Network([FullyConnectedLayer(49, hidden_neurons), FullyConnectedLayer(hidden_neurons, 784)], 10)
net = Network([ConvTransposeLayer(filters=hidden_neurons,
                                  kernel=(3,3),
                                  channels=1,
                                  input_shape=(7,7),
                                  output_shape=(15,15),
                                  mini_batch_size=mini_batch_size,
                                  input_dilation=(2,2),
                                  activation_fn=ReLU),
                FullyConnectedLayer(15 * 15 * hidden_neurons, 784)], 10)
net.train(training_data, epochs, mini_batch_size, lr, validation_data, testing_data,
          optimizer=Network.momentum, optimizer_args={'gamma': gamma}, lmbda=lmbda)

# Save one random mini-batch of originals and predictions to visualize network effectiveness
rand_index = np.random.randint(0, 10000 / mini_batch_size)
imgs = net.test_mb_predictions(rand_index)

for i in np.arange(10):
    img_real = Image.fromarray(np.reshape(testing_data[1][rand_index * mini_batch_size + i].eval() * 255.0, (28, 28)), 'F')
    img_real.convert('RGB').save('real' + str(i) + '.jpg')
    img_output = Image.fromarray(np.reshape(imgs[i] * 255.0, (28, 28)), 'F')
    img_output.convert('RGB').save('output' + str(i) + '.jpg')
