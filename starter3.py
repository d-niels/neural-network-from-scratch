import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import custom_models as cm

# Transforms for regularizing a column of data
def create_transforms(arr):
    transforms = []
    for col in arr.T:
        max_val = max(col)
        min_val = min(col)
        transforms.append((min_val, max_val - min_val))
    return transforms

# Apply transforms to the columns of data to regularize each column
def transform(features, t):
    """
    Applies transforms to every column of features to normalize

    Args:
        features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
    Returns:
        features_t (np.ndarray): transformed array
    """
    for i in range(features.shape[1]):
        if t[i][1] == 0:
            features[:, i] = 0 * features[:, i]
        else:
            features[:, i] = (features[:, i] - t[i][0]) / t[i][1]

# Creates the graphs that will go on the matplotlib figures
def create_plot(ax1, ax2, title, history, epochs=500, ymax=1.2):
    ax1.plot(history.history['acc'], label='train')
    ax1.plot(history.history['val_acc'], label='valid')
    ax1.set(ylabel='Accuracy', xlabel='Epoch', title=title, xlim=[0, epochs], ylim=[0, 1])
    ax1.legend(loc='lower right')

    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='valid')
    ax2.set(ylabel='Loss', xlabel='Epoch', title=title, xlim=[0, epochs], ylim=[0, ymax])
    ax2.legend(loc='upper right')

# Read in the mnist data
def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)

# Print the mnist data to the console 
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

# Read the insurability data             
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)
               
def classify_insurability():
    
    train = read_insurability('./data/three/three_train.csv')
    valid = read_insurability('./data/three/three_valid.csv')
    test = read_insurability('./data/three/three_test.csv')
    
    # insert code to train simple FFNN and produce evaluation metrics
    num_epochs = 500
    results = {}

    fig_acc, axs_acc = plt.subplots(2, 2)
    fig_acc.set_size_inches(18.5, 10.5)
    fig_acc.tight_layout(pad=5.0)

    fig_loss, axs_loss = plt.subplots(2, 2)
    fig_loss.set_size_inches(18.5, 10.5)
    fig_loss.tight_layout(pad=5.0)

    x_train = np.array([x[1] for x in train])
    y_train = np.array([x[0][0] for x in train])

    x_valid = np.array([x[1] for x in valid])
    y_valid = np.array([x[0][0] for x in valid])

    x_test = np.array([x[1] for x in test])
    y_test = np.array([x[0][0] for x in test])
    
    # Regularize
    for i in range(x_train.shape[1]):
        train_min = min(x_train[:, i])
        train_max = max(x_train[:, i])
        x_train[:, i] = (x_train[:, i] - train_min) / (train_max - train_min)
        x_valid[:, i] = (x_valid[:, i] - train_min) / (train_max - train_min)
        x_test[:, i] = (x_test[:, i] - train_min) / (train_max - train_min)


    # Create the baseline model
    run_name = 'Baseline'
    model = cm.insurability()

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[0, 0], axs_loss[0, 0], run_name, history)


    # Increase the lr
    run_name = 'Increased lr'
    model = cm.insurability(lr=0.1)

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[0, 1], axs_loss[0, 1], run_name, history)


    # Try no bias
    run_name = 'Increased lr, no bias'
    model = cm.insurability(lr=0.1, bias=False)

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[1, 0], axs_loss[1, 0], run_name, history)


    # Early stopping when validation accuracy >= 0.99
    run_name = 'Increased lr, early stop'
    model = cm.insurability(lr=0.1)

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[cm.myCallback()],
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[1, 1], axs_loss[1, 1], run_name, history)
    fig_acc.savefig('./graphs/insurability_acc.png')
    fig_loss.savefig('./graphs/insurability_loss.png')

    # Print results
    for key1 in results:
        result = results[key1]
        print('\n' + f'{key1}')
        for key2 in result:
            print(key2, ':', result[key2])
    
    cnf = np.zeros((3, 3))
    predictions = model.predict(x_test)
    for i, p in enumerate(predictions):
        cnf[int(y_test[i]), list(p).index(max(p))] += 1
    print(cnf)

    for i, name in enumerate(['good', 'neutral', 'bad']):
        print(f'\n{name}')
        tp = cnf[i, i]
        fp = sum(cnf[:, i]) - tp
        fn = sum(cnf[i, :]) - tp
        print('f1:', 2*tp / (2*tp + fp + fn))

    if float(results['Increased lr, early stop']['acc']) < 0.95:
        classify_insurability()
    
def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    num_epochs = 20
    results = {}

    fig_acc, axs_acc = plt.subplots(2, 2)
    fig_acc.set_size_inches(18.5, 10.5)
    fig_acc.tight_layout(pad=5.0)

    fig_loss, axs_loss = plt.subplots(2, 2)
    fig_loss.set_size_inches(18.5, 10.5)
    fig_loss.tight_layout(pad=5.0)

    x_train = np.array([x[1] for x in train], dtype=np.int16)
    y_train = np.array([x[0] for x in train], dtype=np.int16)
    x_valid = np.array([x[1] for x in valid], dtype=np.int16)
    y_valid = np.array([x[0] for x in valid], dtype=np.int16)
    x_test = np.array([x[1] for x in test], dtype=np.int16)
    y_test = np.array([x[0] for x in test], dtype=np.int16)
    
    # Drop 0 columns
    drops = []
    for i in range(x_train.shape[1]):
        if max(x_train[:, i]) == min(x_train[:, i]):
            drops.append(i)
    
    x_train = np.delete(x_train, drops, 1)
    x_valid = np.delete(x_valid, drops, 1)
    x_test = np.delete(x_test, drops, 1)

    # Average every 2 points together
    x_train_temp = np.zeros((x_train.shape[0], x_train.shape[1] // 2))
    x_valid_temp = np.zeros((x_valid.shape[0], x_valid.shape[1] // 2))
    x_test_temp = np.zeros((x_test.shape[0], x_test.shape[1] // 2))
    for i in range(x_train.shape[1] // 2):
        x_train_temp[:, i] = (x_train[:, i*2] + x_train[:, i*2 + 1]) / 2
        x_valid_temp[:, i] = (x_valid[:, i*2] + x_valid[:, i*2 + 1]) / 2
        x_test_temp[:, i] = (x_test[:, i*2] + x_test[:, i*2 + 1]) / 2
    
    x_train = x_train_temp
    x_valid = x_valid_temp
    x_test = x_test_temp

    # Regularize
    t = create_transforms(x_train)
    transform(x_train, t)
    transform(x_valid, t)
    transform(x_test, t)

    # Baseline model : 1 hidden layer of 2048 nodes with sigmoid activation
    run_name = 'Baseline'
    model = cm.mnist_model(x_train.shape[1], lr=0.01)

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[0, 0], axs_loss[0, 0], run_name, history, epochs=num_epochs)


    # Decrease the learning rate
    run_name = 'Decreased lr'
    model = cm.mnist_model(x_train.shape[1], lr=0.001)

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[0, 1], axs_loss[0, 1], run_name, history, epochs=num_epochs)


    # Use Relu activation instead of sigmoid
    run_name = 'Relu activation'
    model = cm.mnist_model(x_train.shape[1], lr=0.01, activation='relu')

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[1, 0], axs_loss[1, 0], run_name, history, epochs=num_epochs)


    # Use relu activation and decrease the learning rate
    run_name = 'Relu activation and decreased lr'
    model = cm.mnist_model(x_train.shape[1], lr=0.001, activation='relu')

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs_acc[1, 1], axs_loss[1, 1], run_name, history, epochs=num_epochs)

    fig_acc.savefig('./graphs/mnist_acc.png')
    fig_loss.savefig('./graphs/mnist_loss.png')

    # Print results
    for key1 in results:
        result = results[key1]
        print('\n' + f'{key1}')
        for key2 in result:
            print(key2, ':', result[key2])
    
    cnf = np.zeros((10, 10))
    predictions = model.predict(x_test)
    for i, p in enumerate(predictions):
        cnf[int(y_test[i]), list(p).index(max(p))] += 1
    print(cnf)

    for i in range(10):
        print(i)
        tp = cnf[i, i]
        fp = sum(cnf[:, i]) - tp
        fn = sum(cnf[i, :]) - tp
        print('f1:', 2*tp / (2*tp + fp + fn))

def classify_mnist_reg():
    
    train = read_mnist('./data/mnist/mnist_train.csv')
    valid = read_mnist('./data/mnist/mnist_valid.csv')
    test = read_mnist('./data/mnist/mnist_test.csv')
    # show_mnist('./data/mnist/mnist_test.csv','pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # insert code to train simple FFNN and produce evaluation metrics
    num_epochs = 40
    results = {}

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout(pad=5.0)

    x_train = np.array([x[1] for x in train], dtype=np.int16)
    y_train = np.array([x[0] for x in train], dtype=np.int16)
    x_valid = np.array([x[1] for x in valid], dtype=np.int16)
    y_valid = np.array([x[0] for x in valid], dtype=np.int16)
    x_test = np.array([x[1] for x in test], dtype=np.int16)
    y_test = np.array([x[0] for x in test], dtype=np.int16)
    
    # Drop 0 columns
    drops = []
    for i in range(x_train.shape[1]):
        if max(x_train[:, i]) == min(x_train[:, i]):
            drops.append(i)
    
    x_train = np.delete(x_train, drops, 1)
    x_valid = np.delete(x_valid, drops, 1)
    x_test = np.delete(x_test, drops, 1)

    # Average every 2 points together
    x_train_temp = np.zeros((x_train.shape[0], x_train.shape[1] // 2))
    x_valid_temp = np.zeros((x_valid.shape[0], x_valid.shape[1] // 2))
    x_test_temp = np.zeros((x_test.shape[0], x_test.shape[1] // 2))
    for i in range(x_train.shape[1] // 2):
        x_train_temp[:, i] = (x_train[:, i*2] + x_train[:, i*2 + 1]) / 2
        x_valid_temp[:, i] = (x_valid[:, i*2] + x_valid[:, i*2 + 1]) / 2
        x_test_temp[:, i] = (x_test[:, i*2] + x_test[:, i*2 + 1]) / 2
    
    x_train = x_train_temp
    x_valid = x_valid_temp
    x_test = x_test_temp

    # Regularize
    t = create_transforms(x_train)
    transform(x_train, t)
    transform(x_valid, t)
    transform(x_test, t)

    # Baseline model : 1 hidden layer of 2048 nodes with sigmoid activation
    run_name = 'Baseline'
    model = cm.mnist_model(x_train.shape[1], lr=0.001, activation='relu')

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs[0, 0], axs[1, 0], run_name, history, epochs=num_epochs)


    # Decrease the learning rate
    run_name = 'Regularized'
    model = cm.mnist_model_regularized(x_train.shape[1], lr=0.001, activation='relu')

    # Train for x epochs
    history = model.fit(
        x_train, 
        y_train, 
        epochs=num_epochs,
        validation_data=(x_valid, y_valid),
        callbacks=None,
    )

    # Evaluate the model
    print('\nevaluating...')
    result = model.evaluate(x_test, y_test, verbose=2)
    results[run_name] = {
        'acc': "%.4f" % round(result[1], 4),
        'loss': "%.4f" % round(result[0], 4)
    }
    create_plot(axs[0, 1], axs[1, 1], run_name, history, epochs=num_epochs)

    fig.savefig('./graphs/mnist_reg.png')

    # Print results
    for key1 in results:
        result = results[key1]
        print('\n' + f'{key1}')
        for key2 in result:
            print(key2, ':', result[key2])
    
    cnf = np.zeros((10, 10))
    predictions = model.predict(x_test)
    for i, p in enumerate(predictions):
        cnf[int(y_test[i]), list(p).index(max(p))] += 1
    print(cnf)

    for i in range(10):
        print(i)
        tp = cnf[i, i]
        fp = sum(cnf[:, i]) - tp
        fn = sum(cnf[i, :]) - tp
        print('f1:', 2*tp / (2*tp + fp + fn)) 

def classify_insurability_manual():
    
    train = read_insurability('./data/three/three_train.csv')
    valid = read_insurability('./data/three/three_valid.csv')
    test = read_insurability('./data/three/three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
    
def main():
    # classify_insurability()
    # classify_mnist()
    classify_mnist_reg()
    # classify_insurability_manual()
    
if __name__ == "__main__":
    main()
