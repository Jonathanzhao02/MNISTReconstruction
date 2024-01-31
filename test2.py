"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard libraries
import pickle
import gzip
import random
import sys

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    sensing_matrix = np.random.normal(scale=1/49, size=(49, 784))

    def transformed(data):
        """Transform the data using the sensing matrix and replace labels with
        the original data.
        """
        def matmul(x):
            return np.matmul(sensing_matrix, x)
        return np.apply_along_axis(matmul, axis=1, arr=data), data

    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_data = transformed(training_inputs)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = transformed(validation_inputs)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = transformed(test_inputs)

    return training_data, validation_data, test_data

"""network2.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.
"""

#### Miscellaneous functions
def convolve(ar, filters, strides=(1,1)):
    """Naive 2D convolution transformation that assumes a square kernel and only one input feature map.
    Also assumes strides should perfectly align over the image.
    """
    assert filters.shape[1] == filters.shape[2]
    # Flip and transpose filters for convolution
    filters = np.reshape(np.flip(filters, (1,2)), (filters.shape[0], filters.shape[2], filters.shape[1]))
    # Calculate shape of output after dilation
    num_filters = filters.shape[0]
    output_shape = (num_filters, (ar.shape[0] - filters.shape[1]) // strides[0] + 1, (ar.shape[1] - filters.shape[2]) // strides[1] + 1)

    # Calculate convolution
    output = np.zeros(shape=output_shape)
    # Filter loop
    for i in np.arange(output_shape[0]):
        # Filter dim0 loop
        for j in np.arange(0, output_shape[1], strides[0]):
            # Filter dim1 loop
            for k in np.arange(0, output_shape[2], strides[1]):
                output[i,j,k] = np.sum(np.multiply(ar[j:j+filters.shape[1], k:k+filters.shape[2]], filters[i]))

    return output

def convolve_gradients(grads, filters, output_shape, input_shape, strides=(1,1)):
    """Used to calculate the gradients of a convolutional layer.
    It does so by convolving each filter over their corresponding gradients.
    This multiplies the weights by the derivative of the loss in respect to the
    unactivated output, effectively calculating the derivative of the loss in
    respect to the inputs.
    """
    grads = np.reshape(grads, output_shape)
    output = np.zeros(shape=input_shape)
    for i in np.arange(filters.shape[0]):
        # Filters are not convolved over the entire gradient tensor, only over the relevant gradients
        output += np.sum(convolve(grads[i], filters[i, np.newaxis], strides), axis=0)
    return output

def transform(ar, dilation_rate, padding):
    """Transforms an array for transposed convolution"""
    ar = dilate(ar, dilation_rate)
    return pad(ar, padding)

def pad(ar, padding):
    """Pads a 2D array around the edges"""
    if padding[0] == padding[1]:
        return np.pad(ar, padding[0])
    else:
        for i in np.arange(padding[0]):
            zeros = np.zeros((1, ar.shape[1]))
            ar = np.concatenate((zeros, ar, zeros), axis=0)
        for i in np.arange(padding[1]):
            zeros = np.zeros((ar.shape[0], 1))
            ar = np.concatenate((zeros, ar, zeros), axis=1)
        return ar

def dilate(ar, dilation_rate):
    """Dilates a 2D array such that the values are spread out across a larger array with zeroes between them"""
    if dilation_rate[0] == 1 and dilation_rate[1] == 1: return ar # saves a few calculations if no dilation
    output = np.zeros(shape=((ar.shape[0] - 1) * dilation_rate[0] + 1, (ar.shape[1] - 1) * dilation_rate[1] + 1))

    for i in np.arange(ar.shape[0]):
        for j in np.arange(ar.shape[1]):
            output[i * dilation_rate[0], j * dilation_rate[1]] = ar[i, j]
    
    return output

def sigmoid(z):
    """Sigmoid function"""
    return 1. / (1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of sigmoid function"""
    # store result just to avoid calling sigmoid twice
    temp = sigmoid(z)
    return temp * (1 - temp)

def relu(z):
    """ReLU function"""
    return np.where(z > 0, z, 0)

def relu_prime(z):
    """Derivative of ReLU function"""
    return np.where(z > 0, 1, 0)

def leaky_relu(z):
    """Leaky ReLU function"""
    return np.where(z > 0, z, 0.1 * z)

def leaky_relu_prime(z):
    """Derivative of leaky ReLU function"""
    return np.where(z > 0, 1, 0.1)


#### Define the cost function
class MeanSquaredErrorCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        return np.mean((a - y) ** 2) / 2

    @staticmethod
    def delta(a, y):
        """Return the error delta from the output layer."""
        return (a - y) / len(y)


#### Main Network class
class Network(object):
    def __init__(self, layers):
        """Defines the network layers, cost function, and cost prime functions"""
        self.layers = layers
        self.cost = layers[-1].cost
        self.cost_prime = layers[-1].cost_prime

    def feedforward(self, inpt):
        """Feeds an input forward through the entire network"""
        for layer in self.layers:
            inpt = layer.feedforward(inpt)
        return inpt

    def total_cost(self, inpt, labels):
        """Gets the average cost of the network over all passed inputs and labels"""
        self.feedforward(inpt)
        return self.cost(labels)

    def train(self, training_data, validation_data, testing_data, lr, epochs, mini_batch_size, lmbda=0., gamma=1.):
        """Trains the network with the given data and parameters"""
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.prev_adj = None
        losses = []
        iterations = 0
        print("Starting training...")
        for i in np.arange(epochs):
            # randomly permutes training indices to simulate shuffling
            for j in np.random.permutation(len(training_data[0]) // mini_batch_size):
                if iterations % 1000 == 0:
                    print("On mini batch " + str(iterations))
                iterations += 1
                j = j * mini_batch_size
                cost = self.backpropagate(training_data[0][j:j+mini_batch_size], training_data[1][j:j+mini_batch_size])
                losses.append(cost)
            print("Finished epoch " + str(i))
            print("Validation loss: " + str(self.total_cost(validation_data[0], validation_data[1])))

        print("Test loss: " + str(self.total_cost(testing_data[0], testing_data[1])))
        return losses

    def backpropagate(self, inpt, labels):
        """Backpropagates network for better predictions on the passed inputs and labels"""
        # implicitly calls forward pass
        cost = self.total_cost(inpt, labels)

        # backwards pass
        grads = self.cost_prime(labels)
        layer_adjustments = []

        # iterates through layers backwards to propagate error
        for layer in self.layers[::-1]:
            w_adj, b_adj, grads = layer.backpropagate(grads)
            layer_adjustments.append((w_adj, b_adj))

        # apply layer parameter updates
        for i in np.arange(len(self.layers)):
            w_adj, b_adj = layer_adjustments[-i-1]
            layer = self.layers[i]
            # weight and bias updates with L2 regularization and momentum
            if self.prev_adj is None:
                layer.w = layer.w * (1 - self.lmbda * self.lr / self.mini_batch_size) - w_adj * self.lr
                layer.b = layer.b * (1 - self.lmbda * self.lr / self.mini_batch_size) - b_adj * self.lr
            else:
                prev_w_adj, prev_b_adj = self.prev_adj[-i-1]
                layer.w = layer.w * (1 - self.lmbda * self.lr / self.mini_batch_size) - (w_adj * self.gamma + (1 - self.gamma) * prev_w_adj) * self.lr
                layer.b = layer.b * (1 - self.lmbda * self.lr / self.mini_batch_size) - (b_adj * self.gamma + (1 - self.gamma) * prev_b_adj) * self.lr

        self.prev_adj = layer_adjustments
        # returns cost from before backpropagation
        return cost

### Layer objects

class FullyConnectedLayer(object):
    """
    Layer object for a dense, or fully connected, layer
    """

    def __init__(self, n_in, n_out, activation_fn=sigmoid, activation_prime=sigmoid_prime):
        """Defines the layer information.
        `n_in` is an integer that specifies the number of layer inputs.
        `n_out` is an integer that specifies the number of layer outputs.
        `activation_fn` is an optional function that specifies the activation function for the layer.
        `activation_prime` is an optional function that should be the derivative of the activation_fn.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.activation_prime = activation_prime
        self.w = np.random.normal(loc=0, scale=np.sqrt(1./n_out), size=(n_out, n_in))
        self.b = np.random.normal(loc=0, scale=1, size=(n_out, 1))

    def feedforward(self, inpt):
        """Feeds inputs through the layer"""
        self.inpt = np.reshape(inpt, (-1, self.n_in, 1))
        raw = np.matmul(self.w, self.inpt)
        self.outpt = self.activation_fn(raw + self.b)
        return self.outpt

    # Must call feedforward somehow before calling any gradient or cost functions
    # This allows feedforward to only have to be called once for both cost and backpropagation, rather than once for each.
    def backpropagate(self, grads):
        """Backpropagates errors through the layer, returning layer adjustments and the derivative of the loss with respect to the inputs"""
        grads = np.reshape(grads, (grads.shape[0], self.n_out, 1))
        layer_grads = np.multiply(self.activation_prime(self.outpt), grads)
        # For bias gradients: take the mean of the layer gradients over the entire batch
        bias_adj = np.mean(layer_grads, axis=0)
        # For weight gradients: take the mean after matrix multiplying the layer gradients by the transposed input of the entire batch
        weight_adj = np.matmul(
            layer_grads,
            np.reshape(self.inpt, (self.inpt.shape[0], self.inpt.shape[2], self.inpt.shape[1]))
        )
        weight_adj = np.mean(weight_adj, axis=0)
        # To calculate the derivative of the loss in respect to the inputs, multiply the layer's transposed weights by the layer gradients
        return weight_adj, bias_adj, np.matmul(np.transpose(self.w), layer_grads)

    def cost_prime(self, labels):
        """Derivative of the cost function, which is always MSE, with respect to the passed labels"""
        return MeanSquaredErrorCost.delta(self.outpt, labels)

    def cost(self, labels):
        """Value of the cost function, which is always MSE, when comparing layer outputs to passed labels"""
        return MeanSquaredErrorCost.fn(self.outpt, labels)

class TransposedConvolutionLayer(object):
    """
    Layer object for a transposed convolutional, or deconvolutional, layer.
    Used for upscaling smaller 2D arrays.
    Uses a naive implementation of convolution that expects single-channel input.
    """

    def __init__(self, filters, kernel, input_shape,
                 output_shape=None, activation_fn=leaky_relu, activation_prime=leaky_relu_prime, dilation_rate=(1,1)):
        """Defines the layer information.
        `filters` is the number of filters in the transposed convolutional layer.
        `kernel` is a tuple of length 2 that specifies the shape of each filter.
        `input_shape` is a tuple of length 2 that specifies the shape of the input.
        `output_shape` is an optional tuple of length 2 that specifies the output shape.
        `activation_fn` is an optional function that specifies the activation function for the layer.
        `activation_prime` is an optional function that should be the derivative of the activation_fn.
        `dilation_rate` is an optional tuple of length 2 that specifies how much the inputs should be dilated.
        """
        self.filter_shape = (filters, kernel[0], kernel[1])
        self.input_shape = input_shape
        # calculates the intermediate shape of the input after transformation for convolution
        self.transformed_shape = (
            dilation_rate[0] * (input_shape[0] - 1) + 2 * (kernel[1] - 1) + 1,
            dilation_rate[1] * (input_shape[1] - 1) + 2 * (kernel[0] - 1) + 1)
        if output_shape is not None:
            self.output_shape = (filters, output_shape[0], output_shape[1])
        else:
            # calculates the output shape automatically if unspecified
            self.output_shape = (filters,
                                 (input_shape[0] - 1) * dilation_rate[0] + kernel[1],
                                 (input_shape[1] - 1) * dilation_rate[1] + kernel[0])
        self.activation_fn = activation_fn
        self.activation_prime = activation_prime
        self.dilation_rate = dilation_rate
        n_out = np.prod(kernel)
        self.w = np.random.normal(scale=np.sqrt(1.0/n_out), size=self.filter_shape)
        self.b = np.random.normal(size=(self.output_shape[1], self.output_shape[2]))

    def feedforward(self, inpt):
        """Feeds input through the convolution layer"""
        def inpt_transform(x):
            """Transforms the input for transposed convolution (uses same underlying operation of convolution)"""
            x = np.reshape(x, self.input_shape)
            return np.ravel(transform(x, self.dilation_rate, (self.filter_shape[2] - 1, self.filter_shape[1] - 1)))

        def convolve_over_inpt(x):
            """Applies convolution, activation transformation, and bias"""
            x = np.reshape(x, self.transformed_shape)
            return self.activation_fn(convolve(x, self.w) + self.b)

        self.inpt = inpt.reshape((-1, np.prod(self.input_shape)))
        self.transformed_inpt = np.apply_along_axis(inpt_transform, 1, self.inpt)
        raw = np.apply_along_axis(convolve_over_inpt, 1, self.transformed_inpt)
        self.outpt = np.reshape(raw, (-1,) + self.output_shape)
        return self.outpt

    def backpropagate(self, grads):
        """Backpropagates error through the layer, returning layer adjustments and the derivative of the loss with respect to the inputs"""
        mini_batch_size = grads.shape[0]
        grads = grads.reshape((mini_batch_size,) + self.output_shape)
        layer_grads = np.multiply(self.activation_prime(self.outpt), grads)
        # for bias gradients: take the mean of the layer gradients over the entire batch and all the filters (axes 0 and 1)
        bias_adj = np.mean(layer_grads, axis=(0,1))
        # for weight gradients: take the mean after convolving gradients over the transformed input of the entire batch
        weight_adj = np.mean([convolve(np.reshape(self.transformed_inpt[i], self.transformed_shape), layer_grads[i])
                              for i in np.arange(mini_batch_size)], axis=0)
        # to calculate the derivative of the loss in respect to the inputs, convolve filters over corresponding layer gradients
        grads = [convolve_gradients(layer_grads[i], self.w, self.output_shape, self.input_shape, strides=self.dilation_rate)
                 for i in np.arange(mini_batch_size)]
        grads = np.reshape(grads, (mini_batch_size,) + self.input_shape)
        return weight_adj, bias_adj, grads

### Execution
# Load data
tr_d, va_d, ts_d = load_data_wrapper()

# Network parameters
lr = 0.01 # learning rate
epochs = 50 # total epochs
mini_batch_size = 10 # size of each mini batch
lmbda = 0.0001 # l2 regularization constant
gamma = 0.95 # momentum constant

# Create network architecture
conv_layer1 = TransposedConvolutionLayer(filters=16, kernel=(5,3), input_shape=(7,7), dilation_rate=(3, 2))
layer2 = FullyConnectedLayer(np.prod(conv_layer1.output_shape), 784)

net = Network([conv_layer1, layer2])
print("Created network!")

# Training
losses = net.train(tr_d, va_d, ts_d, lr, epochs, mini_batch_size, lmbda, gamma)

# Plot losses, and display real and predicted images for one sample from testing set
import matplotlib.pyplot as plt

rand_index = np.random.randint(0,10000)
real = ts_d[1][rand_index:rand_index+1]
pred = net.feedforward(ts_d[0][rand_index:rand_index+1])
fig = plt.figure(figsize=(12,12))
fig.add_subplot(3, 1, 1)
plt.plot(np.arange(len(losses)), losses, c='g')
fig.add_subplot(3, 1, 2)
plt.imshow(real.reshape(28, 28), cmap='gray')
fig.add_subplot(3, 1, 3)
plt.imshow(pred.reshape(28, 28), cmap='gray')
plt.show()
