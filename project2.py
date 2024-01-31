"""project2.py
~~~~~~~~~~~~~~
A combination of mnist_loader.py and network2.py from Michael Nielsen's
deep learning tutorial. mnist_loader has been modified to return
7x7 images and 28x28 images as inputs and labels, while network2.py
has been heavily modified to work using an array of layers instead
of one weight matrix. This allows different layer types
to be used in a single network, such as convolutional layers,
instead of solely fully connected/dense layers.

More specifics of changes can be found in the corresponding
module doc strings that I left below.

The execution code can be found at the very bottom of the file.
"""


"""
mnist_loader.py
~~~~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.

Edited by Jonathan Zhao
Changed `load_data_wrapper` to multiply original 28x28 images by a
randomly generated sensing matrix, producing a 7x7 image.
Replaced inputs with 7x7 images, and labels with the original 28x28
images.
Updated code to work with python 3.
"""

#### Libraries
# Standard libraries
import pickle
import gzip
import sys

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_path='./data/mnist.pkl.gz'):
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
    This is a nice data format, but for use in our neural network we
    must modify the format of the ``training_data`` significantly.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(data_path, 'rb')
    # changed deprecated cPickle to pickle with latin1 encoding for numpy.ndarray
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper(data_path='./data/mnist.pkl.gz'):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is
    specific to our problem.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 2-dimensional numpy.ndarray
    containing the 7x7 transformed image.  ``y`` is a 2-dimensional
    numpy.ndarray containing the original 28x28 image.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 2-dimensional
    numpy.ndarry containing the 7x7 transformed image, and ``y`` is the
    corresponding original 28x28 image.
    """
    # generate the sensing matrix
    sensing_matrix = np.random.normal(scale=1/49, size=(49, 784))

    # utility function for numpy's apply_along_axis
    def matmul_sensing(x):
        """Returns the matrix multiplication of the sensing matrix
        and the argument."""
        return np.matmul(sensing_matrix, x)

    # converts original images into desired format of compressed and original signals
    def transformed(data):
        """Return a tuple containing the transformed 7x7 image
        and the original 28x28 image.
        """
        return np.apply_along_axis(matmul_sensing, axis=1, arr=data), data

    tr_d, va_d, te_d = load_data(data_path)

    # reformat all data
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

Edited by Jonathan Zhao
Removed unused cost functions for only the mean-squared-error function.
Rewrote Network object completely to work through independent layers.
Added two new layer objects for dense and transposed convolutional layers.
Added momentum to stochastic gradient descent.
Added ReLU and leaky ReLU activation functions, along with their derivatives.
Added static methods for convolution and corresponding transformations.
Updated code to work with python 3.
"""

#### Miscellaneous functions
def convolve(ar, filters, strides=(1,1)):
    """Naive 2D convolution transformation over one channel.
    Assumes square kernel and perfectly-aligning strides.
    """
    # flip filters for convolution
    filters = np.flip(filters, (1,2))
    # calculate shape of output after dilation
    output_shape = (
        filters.shape[0],
        (ar.shape[0] - filters.shape[1]) // strides[0] + 1,
        (ar.shape[1] - filters.shape[2]) // strides[1] + 1
    )

    # calculate convolution
    output = np.zeros(shape=output_shape)
    # filter loop
    for i in np.arange(output_shape[0]):
        # kernel dim0 loop
        for j in np.arange(0, output_shape[1], strides[0]):
            # kernel dim1 loop
            for k in np.arange(0, output_shape[2], strides[1]):
                # apply filter to one section of the array and sum the results
                filter_result = np.multiply(ar[j:j+filters.shape[1], k:k+filters.shape[2]], filters[i])
                output[i,j,k] = np.sum(filter_result)

    return output

def convolve_gradients(grads, filters, input_shape, strides=(1,1)):
    """Used to calculate the gradients of a convolutional layer.
    It does so by convolving each filter over their corresponding gradients.
    This multiplies the weights by the derivative of the loss in respect to the
    unactivated output, effectively calculating the derivative of the loss in
    respect to the inputs.
    """
    output = np.zeros(shape=input_shape)
    for i in np.arange(filters.shape[0]):
        # filters are not convolved over the entire gradient tensor,
        # only over the relevant gradients, thus the grads[i] indexing
        output += np.sum(convolve(grads[i], filters[i, np.newaxis], strides), axis=0)
    return output

def transform(ar, dilation_rate, padding):
    """Transforms an array for transposed convolution"""
    ar = dilate(ar, dilation_rate)
    return pad(ar, padding)

def pad(ar, padding):
    """Pads a 2D array around the edges"""
    # due to behavior of numpy's pad function, two cases must be accounted for:
    # when the padding is the same all around, and when padding differs around each axis
    # this is to ensure padding is applied how we want it around the matrix
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
    """Dilates a 2D array"""
    # saves a few calculations if no dilation is necessary
    if dilation_rate[0] == 1 and dilation_rate[1] == 1:
        return ar
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
    """Leaky ReLU function, slope of 0.1"""
    return np.where(z > 0, z, 0.1 * z)

def leaky_relu_prime(z):
    """Derivative of leaky ReLU function"""
    return np.where(z > 0, 1, 0.1)


#### Define the cost function
class MeanSquaredErrorCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return np.mean((a - y) ** 2) / 2

    @staticmethod
    def delta(a, y):
        """Return the error delta from the output layer."""
        # the divisor takes the product of the shape excluding the first axis
        # since the first axis represents the size of the mini batch
        return (a - y) / np.prod(a.shape[1:])


#### Main Network class
class Network(object):
    def __init__(self, layers):
        """Defines the network's architecture"""
        self.layers = layers
        self.cost = layers[-1].cost
        self.cost_prime = layers[-1].cost_prime

    def feedforward(self, inpt):
        """Feeds inputs forward through the entire network"""
        outpt = inpt
        for layer in self.layers:
            outpt = layer.feedforward(outpt)
        return outpt

    def total_cost(self, inpt, labels):
        """Returns the average cost of the network over passed inputs and labels"""
        self.feedforward(inpt)
        return self.cost(labels)

    def train(self, training_data, validation_data, testing_data,
              lr, epochs, mini_batch_size, lmbda=0., gamma=1., print_loss=False):
        """Trains the network with the given data and parameters.
        `training_data` is the set of inputs and labels to be used for training.
        `validation_data` is the set of inputs and labels to be used for validation at the end of each epoch.
        `testing_data` is the set of inputs and labels to be used for testing at the end of training.
        `lr` is the learning rate.
        `epochs` is the total number of epochs to train.
        `mini_batch_size` is the size of each mini batch.
        `lmbda` is the L2 regularization constant (0 = disabled).
        `gamma` is the momentum constant (1 = disabled).
        `print_loss` specifies whether the training loss should be printed after each mini batch.
        """
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.prev_adj = None
        training_losses = []
        validation_losses = []
        iterations = 0
        print("Starting training...")
        for i in np.arange(epochs):
            # randomly permutes training indices to simulate shuffling
            for j in np.random.permutation(len(training_data[0]) // mini_batch_size):
                if iterations % 1000 == 0:
                    print("On mini batch " + str(iterations))
                j = j * mini_batch_size

                # calculate and record training loss
                training_cost = self.backpropagate(
                    training_data[0][j:j+mini_batch_size],
                    training_data[1][j:j+mini_batch_size]
                )
                training_losses.append(training_cost)
                if print_loss:
                    print("Mini batch " + str(iterations) + " loss: " + str(training_cost))
                
                iterations += 1
            print("Finished epoch " + str(i))

            # calculate and record validation loss
            validation_cost = self.total_cost(validation_data[0], validation_data[1])
            validation_losses.append(validation_cost)
            print("Validation loss: " + str(validation_cost))

        # calculate and record testing loss
        test_cost = self.total_cost(testing_data[0], testing_data[1])
        print("Test loss: " + str(test_cost))
        return training_losses, validation_losses, test_cost

    def backpropagate(self, inpt, labels):
        """Backpropagates errors on passed inputs and labels using
        stochastic gradient descent with momentum"""
        # implicitly calls forward pass
        cost = self.total_cost(inpt, labels)

        # backwards pass
        grads = self.cost_prime(labels)
        layer_adjustments = []
        for layer in self.layers[::-1]: # iterates backwards through layers to backpropagate errors
            w_adj, b_adj, grads = layer.backpropagate(grads)
            layer_adjustments.append((w_adj, b_adj))

        # apply layer parameter updates
        for i in np.arange(len(self.layers)):
            w_adj, b_adj = layer_adjustments[-i-1]
            layer = self.layers[i]
            # weight and bias updates with L2 regularization and momentum
            if self.prev_adj is None: # if no previous adjustments are present, simply apply the update
                layer.w = layer.w * (1 - self.lmbda * self.lr / self.mini_batch_size) - \
                          w_adj * self.lr
                layer.b = layer.b * (1 - self.lmbda * self.lr / self.mini_batch_size) - \
                          b_adj * self.lr
            else: # if previous adjustments are present, use them for momentum update
                prev_w_adj, prev_b_adj = self.prev_adj[-i-1]
                layer.w = layer.w * (1 - self.lmbda * self.lr / self.mini_batch_size) - \
                          (w_adj * self.gamma + (1 - self.gamma) * prev_w_adj) * self.lr
                layer.b = layer.b * (1 - self.lmbda * self.lr / self.mini_batch_size) - \
                          (b_adj * self.gamma + (1 - self.gamma) * prev_b_adj) * self.lr

        self.prev_adj = layer_adjustments # store adjustments for momentum
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

    # Must call feedforward before calling any gradient or cost functions.
    # This allows feedforward to only have to be called once for both cost and backpropagation.
    def backpropagate(self, grads):
        """Backpropagates errors through the layer, returning layer adjustments and
        the derivative of the loss with respect to the inputs
        """
        grads = np.reshape(grads, (grads.shape[0], self.n_out, 1))
        layer_grads = np.multiply(self.activation_prime(self.outpt), grads)
        # for bias gradients: take the mean of the layer gradients over the entire batch
        bias_adj = np.mean(layer_grads, axis=0)
        # for weight gradients: take the mean after multiplying the
        # layer gradients by the transposed input of the entire batch
        weight_adj = np.matmul(
            layer_grads,
            np.reshape(self.inpt, (self.inpt.shape[0], self.inpt.shape[2], self.inpt.shape[1]))
        )
        weight_adj = np.mean(weight_adj, axis=0)
        # to calculate the derivative of the loss in respect to the inputs,
        # multiply the layer's transposed weights by the layer gradients
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
    Uses a naive implementation of convolution that assumes that all parameter shapes are square,
    (i.e. (1,1), (2,2), (3,3), etc. for kernel, input_shape, and dilation_rate),
    and that the input contains only one input feature map (AKA channel).
    """

    def __init__(self, filters, kernel, input_shape,
                 output_shape=None, dilation_rate=(1,1),
                 activation_fn=leaky_relu, activation_prime=leaky_relu_prime):
        """Defines the layer information.
        `filters` is the number of filters in the transposed convolutional layer.
        `kernel` is a tuple of length 2 that specifies the shape of each filter.
        `input_shape` is a tuple of length 2 that specifies the shape of the input.
        `output_shape` is an optional tuple of length 2 that specifies the output shape.
        `dilation_rate` is an optional tuple of length 2 that specifies how much the inputs should be dilated.
        `activation_fn` is an optional function that specifies the activation function for the layer.
        `activation_prime` is an optional function that should be the derivative of the activation_fn.
        """
        # ensure our shape assumptions hold
        assert kernel[0] == kernel[1]
        assert input_shape[0] == input_shape[1]
        assert dilation_rate[0] == dilation_rate[1]

        self.filter_shape = (filters, kernel[0], kernel[1])
        self.input_shape = input_shape
        # calculates the intermediate shape of the input
        self.transformed_shape = (
            dilation_rate[0] * (input_shape[0] - 1) + 2 * (kernel[0] - 1) + 1,
            dilation_rate[1] * (input_shape[1] - 1) + 2 * (kernel[1] - 1) + 1
        )
        if output_shape is not None:
            self.output_shape = (filters, output_shape[0], output_shape[1])
        else:
            # calculates the output shape automatically if unspecified
            self.output_shape = (
                filters,
                self.transformed_shape[0] - kernel[0] + 1,
                self.transformed_shape[1] - kernel[1] + 1
            )
        self.activation_fn = activation_fn
        self.activation_prime = activation_prime
        self.dilation_rate = dilation_rate
        n_out = np.prod(kernel)
        self.w = np.random.normal(scale=np.sqrt(1.0/n_out), size=self.filter_shape)
        self.b = np.random.normal(size=(self.output_shape[1], self.output_shape[2]))

    def feedforward(self, inpt):
        """Feeds input through the transposed convolution layer"""
        self.inpt = np.reshape(inpt, (-1,) + self.input_shape)
        # saves input tranformation for gradient calculations
        self.transformed_inpt = [transform(
            x,
            self.dilation_rate,
            (self.filter_shape[1] - 1, self.filter_shape[2] - 1)
        ) for x in self.inpt]
        # convolve and apply biases and activation fn
        self.outpt = np.array(
            [self.activation_fn(
                convolve(x, self.w) + self.b
            ) for x in self.transformed_inpt]
        )
        return self.outpt

    # Must call feedforward before calling any gradient or cost functions.
    # This allows feedforward to only have to be called once for both cost and backpropagation.
    def backpropagate(self, grads):
        """Backpropagates error through the layer, returning layer adjustments
        and the derivative of the loss with respect to the inputs
        """
        mini_batch_size = grads.shape[0]
        grads = grads.reshape((mini_batch_size,) + self.output_shape)
        layer_grads = np.multiply(self.activation_prime(self.outpt), grads)
        # for bias gradients: take the mean of the layer gradients over the
        # entire batch and all the filters (axes 0 and 1)
        bias_adj = np.mean(layer_grads, axis=(0,1))
        # for weight gradients: take the mean of the entire batch (axis 0)
        # after convolving gradients over the transformed input
        weight_adj = [convolve(
            self.transformed_inpt[i],
            layer_grads[i]
        ) for i in np.arange(mini_batch_size)]
        weight_adj = np.mean(weight_adj, axis=0)
        # to calculate the derivative of the loss in respect to the inputs,
        # convolve filters over the corresponding layer gradients
        grads = [convolve_gradients(
            layer_grads[i],
            self.w,
            self.input_shape,
            self.dilation_rate
        ) for i in np.arange(mini_batch_size)]
        grads = np.reshape(grads, (mini_batch_size,) + self.input_shape)
        return weight_adj, bias_adj, grads

    # These are never used since the transposed convolutional layer
    # should not be last layer for our problem
    def cost_prime(self, labels):
        """Derivative of the cost function, which is always MSE, with respect to the passed labels"""
        return MeanSquaredErrorCost.delta(self.outpt, labels)

    def cost(self, labels):
        """Value of the cost function, which is always MSE, when comparing layer outputs to passed labels"""
        return MeanSquaredErrorCost.fn(self.outpt, labels)


### Execution code
# Load data from path
if len(sys.argv) > 1:   # if command line argument exists, load from specified path
    tr_d, va_d, ts_d = load_data_wrapper(sys.argv[1])
else:                   # otherwise load from default path './data/mnist.pkl.gz'
    tr_d, va_d, ts_d = load_data_wrapper()

# Network parameters
lr = 0.1                # learning rate
epochs = 50             # total epochs
mini_batch_size = 10    # size of each mini batch
lmbda = 0.0001          # l2 regularization constant
gamma = 0.95            # momentum constant
print_loss = True       # whether costs should be printed after every batch

# Create network architecture
conv_layer1 = TransposedConvolutionLayer(filters=16, kernel=(3, 3), input_shape=(7, 7))
layer2 = FullyConnectedLayer(np.prod(conv_layer1.output_shape), 784)
net = Network([conv_layer1, layer2])
print("Created network!")

# Training
# Note: Significantly faster without a convolutional layer
training_losses, validation_losses, test_loss = net.train(
    training_data=tr_d,
    validation_data=va_d,
    testing_data=ts_d,
    lr=lr,
    epochs=epochs,
    mini_batch_size=mini_batch_size,
    lmbda=lmbda,
    gamma=gamma,
    print_loss=print_loss
)

# Plot training losses, and display real and predicted images for one sample from testing set
rand_index = np.random.randint(0, 10000)
real = ts_d[1][rand_index:rand_index+1]
pred = net.feedforward(ts_d[0][rand_index:rand_index+1])
fig = plt.figure(figsize=(6, 6))
fig.add_subplot(3, 1, 1)
plt.plot(np.arange(len(training_losses)), training_losses, c='g')
fig.add_subplot(3, 1, 2)
plt.imshow(np.reshape(real, (28, 28)), cmap='gray')
fig.add_subplot(3, 1, 3)
plt.imshow(np.reshape(pred, (28, 28)), cmap='gray')
plt.show()
