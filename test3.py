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
Most edits are listed below.
Renamed to project2.py.
Updated a few functions to use Python 3.8 and Theano 1.0.0.
Added different optimizers such as momentum.
Changed all usages of T.grad to manual backpropagation calculations.
Changed load_data_shared to create 7x7 compressions of the 28x28 images for input.
The parameters I used for testing are at the bottom.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from PIL import Image

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
def sigmoid(z): return 2. / (T.exp(-z) + 1) - 1
def sigmoid_prime(z): return 2. * T.exp(-z) / T.pow(1 + T.exp(-z), 2)

#### Constants
# Without T.grad, the GPU cannot be utilized anyway.
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
        self.params = [param for layer in self.layers for param in layer.params] # delete when transitioned over to completely manual gradients
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

    def momentum(self, cost, cost_prime, lmbda, eta, args):
        """SGD with momentum optimizer, requires gamma as an argument for momentum."""
        assert args['gamma']
        gamma = args['gamma']
        updates = []
        # Initial gradients are the derivative of the outputs in respect to the loss function, MSE.
        grads = cost_prime
        # Update rule for momentum.
        def update_rule(old_val, new_val):
            return old_val * gamma + new_val
        # L2 normalization rule.
        def norm_rule(param, update):
            return (1 - lmbda * eta / self.mini_batch_size) * param - eta * update
        for layer in self.layers[::-1]: # traverses through list backwards for easier backpropagation calculation
            grads = layer.backpropagate(grads, updates, update_rule, norm_rule)
        return updates

    def sgd(self, cost, cost_prime, lmbda, eta, args):
        """Standard gradient descent optimizer, no extra arguments"""
        updates = []
        # Initial gradients are the derivative of the outputs in respect to the loss function, MSE.
        grads = cost_prime
        # Update rule for standard gradient descent.
        def update_rule(old_val, new_val):
            return new_val
        # L2 normalization rule.
        def norm_rule(param, update):
            return (1 - lmbda * eta / self.mini_batch_size) * param - eta * update
        for layer in self.layers[::-1]: # traverses through list backwards for easier backpropagation calculation
            grads = layer.backpropagate(grads, updates, update_rule, norm_rule)
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
        cost = self.layers[-1].cost(self)
        cost_prime = self.layers[-1].cost_prime(self)
        updates = optimizer(self, cost, cost_prime, lmbda, eta, optimizer_args)

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
        self.mini_batch_size = mini_batch_size
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def loss_gradient(self, grads):
        """Calculate the gradient of the layer parameters in respect to the loss."""
        layer_grads = sigmoid_prime(self.output_dropout) * grads
        # Must divide by mini batch size since T.dot sums up all the batch results.
        return T.mean(layer_grads, axis=0), T.dot(T.transpose(layer_grads), self.inpt) / self.mini_batch_size

    def backpropagate(self, grads, updates, update_rule, norm_rule):
        # Calculate the derivatives of the layer parameters in respect to the loss.
        bias_adj, weight_adj = self.loss_gradient(grads)
        weights = self.w
        biases = self.b
        # Create shared Theano variables representing the gradient updates. Initialize values to zero.
        weight_update = theano.shared(weights.get_value()*0, broadcastable=weights.broadcastable)
        bias_update = theano.shared(biases.get_value()*0, broadcastable=biases.broadcastable)
        # Due to how Theano works, gradient updates are appended to an array, which is evaluated and applied every mini batch.
        # General idea of backpropagation stays the same.
        updates.append((weights, norm_rule(weights, weight_update)))
        updates.append((biases, norm_rule(biases, bias_update)))
        # Apply the update rule to the gradients. This is important for momentum.
        updates.append((weight_update, update_rule(weight_update, T.transpose(weight_adj))))
        updates.append((bias_update, update_rule(bias_update, bias_adj)))
        # Calculate the gradients to backpropagated to the next layer.
        grads = T.dot(weights, bias_adj)
        return grads

    def cost_prime(self, net):
        "Return the MSE derivative."
        return (self.output_dropout - net.y) / self.n_out

    def cost(self, net):
        "Return the MSE loss."
        return T.mean((self.output_dropout - net.y) ** 2) / 2

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].shape.eval()[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

#### Main code
# Network parameters
hidden_neurons = 588
epochs = 1000 # of course, higher is better (to a limit), but definitely lower if you want to test the network within your lifetime
mini_batch_size = 10
lr = 0.5
lmbda = 0.0001
gamma = 0.9

# Get all MNIST inputs and labels
training_data, validation_data, testing_data = load_data_shared()

layer_1 = FullyConnectedLayer(49, hidden_neurons)
layer_2 = FullyConnectedLayer(hidden_neurons, 784)

# Create and train the network
net = Network([layer_1, layer_2], 10)
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
