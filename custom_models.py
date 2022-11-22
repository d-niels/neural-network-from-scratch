import tensorflow as tf

# Callback to stop training when a particular validation accuracy is reached
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_acc') > 0.989):   
            print("\nReached 0.96 validation accuracy, so stopping training!!")   
            self.model.stop_training = True

# Softmax activation function
def softmax(x, axis=-1):
    numerator = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    denomerator = tf.reduce_sum(numerator, axis=axis, keepdims=True)
    logits = numerator / denomerator
    logits._keras_logits = x
    return logits

# Insurability model
def insurability(bias=True, lr=0.01):
    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(3,), use_bias=bias),
        tf.keras.layers.Dense(3, use_bias=bias),
        tf.keras.layers.Activation(softmax)
    ])

    # Compile the model with SGD and softmax
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )

    return model

# MNIST model
def mnist_model(input_size, lr=0.01, activation='sigmoid'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2048, activation=activation, input_shape=(input_size,)),
        tf.keras.layers.Dense(10),
    ])

    # Compile the model with SGD and softmax
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )

    return model

# Regularized MNIST model
def mnist_model_regularized(input_size, lr=0.01, activation='sigmoid'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            2048, 
            activation=activation, 
            input_shape=(input_size,), 
            activity_regularizer=tf.keras.regularizers.L1(0.005),
        ),
        tf.keras.layers.Dense(10),
    ])

    # Compile the model with SGD and softmax
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )

    return model