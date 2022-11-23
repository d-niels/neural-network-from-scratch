import TensorTorch as tt
import numpy as np
import matplotlib.pyplot as plt

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

def test_network():
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
    trainer.fit(x_train, y_train, x_valid, y_valid, epochs=50, batch_size=len(x_test), eval_every=1)
    print(trainer.evaluate(x_test, y_test))

if __name__ == '__main__':
    test_network()
