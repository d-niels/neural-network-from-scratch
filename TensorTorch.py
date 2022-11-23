import numpy as np
from copy import deepcopy
from scipy.special import logsumexp


def assert_dimensionality(arr1: np.ndarray, arr2: np.ndarray):
    assert arr1.shape == arr2.shape, \
        """
        Mismatched shape; 
        first shape is {0}
        and second shape is {1}.
        """.format(tuple(arr2.shape), tuple(arr1.shape))
    return None


def softmax(x, axis=-1):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class Encoder():
    """
    Encodes labels
    """
    def __init__(self):
        self.encodings: dict[_, int] = {}
    
    def fit(self, labels: np.ndarray):
        for i, label in enumerate(np.unique(labels)):
            self.encodings[label] = i
    
    def encode(self, labels: np.ndarray) -> np.ndarray:
        encodings = np.zeros((len(labels), len(self.encodings)))
        for i, l in enumerate(labels):
            encodings[i, self.encodings[l]] = 1
        return encodings


class Operation():
    """
    All operations in a neural network must follow a set of rules
    Networks have forward and backward passes
    In the forward pass, an ouput is computed from an input
    In the backward pass, an input gradient is computed from an output gradient
    """
    def __init__(self):
        pass

    def forward(self, input_: np.ndarray):
        """
        Stores the input and computes the output
        """
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the input gradient
        """
        assert_dimensionality(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_dimensionality(self.input_, self.input_grad)
        return self.input_grad
    
    def _output(self) -> np.ndarray:
        pass
    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        pass


class ParamOperation(Operation):
    """
    Operations that involve trainable parameters parameters
    """
    def __init__(self, param: np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Calculates input gradient and parameter gradient
        """
        # assert_dimensionality(self.ouput, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_dimensionality(self.input_, self.input_grad)
        assert_dimensionality(self.param, self.param_grad)

        return self.input_grad
    
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        pass


class WeightMultiply(ParamOperation):
    """
    Operation for multiplying inputs by weights
    """
    def __init__(self, W: np.ndarray):
        super().__init__(W)
    
    def _output(self) -> np.ndarray:
        """
        Inputs dot weights
        """
        return np.dot(self.input_, self.param)
    
    def _input_grad(self, output_grad) -> np.ndarray:
        """
        Inputs_T dot output_grad
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))
    
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    Operation for adding the bias term
    """
    def __init__(self, B: np.ndarray) -> np.ndarray:
        assert B.shape[0] == 1
        super().__init__(B)
    
    def _output(self) -> np.ndarray:
        return self.input_ + self.param
    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input_) * output_grad
    
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    """
    Operation for sigmoid activation
    """
    def __init__(self):
        super().__init__()
    
    def _output(self) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * self.input_))
    
    def _input_grad(self, output_grad) -> np.ndarray:
        return self.output * (1 - self.output) * output_grad


class Layer():
    """
    Basic rules that layers must follow
    """
    def __init__(self, neurons: int):
        self.neurons: int = neurons
        self.first: bool = True
        self.params: list[np.ndarray] = []
        self.param_grads: list[np.ndarray] = []
        self.operations: list[Operation] = []
        self.seed: int = None

    def _setup_layer(self, num: int):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient going backwards
        """
        assert_dimensionality(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()

        return input_grad

    def _param_grads(self) -> np.ndarray:
        """
        Get the _param_grads
        These will be used by the optimizer
        """
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> np.ndarray:
        """
        Get the _params
        These will be used by the optimizer
        """
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    """
    A layer of neurons
    """
    def __init__(self, neurons: int, activation: Operation = Sigmoid()):
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray) -> None:
        """
        Setup the operations used in a dense layer
        """
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]


class Loss():
    """
    Rules for loss functions
    """
    def __init__(self):
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the loss
        """
        assert_dimensionality(prediction, target)
        self.prediction = prediction
        self.target = target
        loss = self._output()

        return loss

    def backward(self) -> np.ndarray:
        """
        Computes gradient of the loss wrt the input to the loss function
        """
        self.input_grad = self._input_grad()
        assert_dimensionality(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        raise NotImplementedError()


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, eps: float=1e-9):
        super().__init__()
        self.eps = eps
      
    def _output(self) -> float:
      softmax_preds = softmax(self.prediction, axis=1)
      self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps) # clip to prevent losses of 0
      loss_vector = -1 * self.target * np.log(self.softmax_preds)
      return np.sum(loss_vector)
    
    def _input_grad(self) -> np.ndarray:
      return self.softmax_preds - self.target


class NeuralNetwork():
    def __init__(self, layers, loss: Loss, seed: int = 1):
        self.layers: list[Layer] = layers
        self.loss = loss
        self.seed = seed
        for layer in self.layers:
          layer.seed = seed     

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Forward pass
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: np.ndarray):
        """
        backward pass
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Training protocol
        """
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss
    
    def params(self):
        """
        Get the parameters
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """
        Get the parameters gradients
        """
        for layer in self.layers:
            yield from layer.param_grads    


class Optimizer():
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


class Trainer(object):
    """
    Trains a neural network
    """
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        """
        Requires a neural network and an optimizer in order for training to occur. 
        Assign the neural network as an instance variable to the optimizer.
        """
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)
        
    def generate_batches(self, x: np.ndarray, y: np.ndarray, size: int = 32):
        """
        Generates batches for training 
        """
        for i in range(0, x.shape[0], size):
            xi, yi = x[i:i+size], y[i:i+size]
            yield xi, yi

            
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            epochs: int=100,
            batch_size: int=32,
            seed: int = 1,
            restart: bool = True,
            early_stop: bool = True):
        """
        Runs training epochs over the network
        """
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        for e in range(epochs):
            last_model = deepcopy(self.net)
            batch_generator = self.generate_batches(x_train, y_train, batch_size)

            for (X_batch, y_batch) in batch_generator:
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            test_preds = self.net.forward(x_valid)
            loss = self.net.loss.forward(test_preds, y_valid)

            print(f"Epoch {e+1}: val_acc: {round(self.evaluate(x_valid, y_valid), 3)}")
            if loss < self.best_loss:      
                self.best_loss = loss
            elif early_stop:
                self.net = last_model
                setattr(self.optim, 'net', self.net)
                print(f"Loss increased after epoch {e+1}, training haulted")
                break
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Computes the accuracy of the model on a test set of data
        """
        predictions = self.net.forward(x_test)
        count = 0
        for pred, target in zip(predictions, y_test):
            if np.argmax(pred) == np.argmax(target):
                count += 1
        return count / len(x_test)
