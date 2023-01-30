import numpy as np
import matplotlib.pyplot as plt
import TensorTorch as tt

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
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
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
    
    encoder = tt.Encoder()
    encoder.fit(y_train)
    y_train = encoder.encode(y_train)
    y_valid = encoder.encode(y_valid)
    y_test = encoder.encode(y_test)

    # Create the model
    three_model = tt.NeuralNetwork(
        layers=[tt.Dense(2, activation=tt.Sigmoid()),
                tt.Dense(3)],
        loss = tt.SoftmaxCrossEntropyLoss()
    )
    trainer = tt.Trainer(three_model, tt.SGD(lr=0.1))
    trainer.fit(x_train, y_train, x_valid, y_valid, epochs=100, batch_size=1, early_stop=False)
    print('testing accuracy:', trainer.evaluate(x_test, y_test))
    
if __name__ == "__main__":
    classify_insurability()
