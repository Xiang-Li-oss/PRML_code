from typing import List
import numpy as np
import matplotlib.pyplot as plt

all_datas = [
    [[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
     [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
     [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
     [-0.76, 0.84, -1.96]],
    [[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
     [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
     [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
     [0.46, 1.49, 0.68]],
    [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
     [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
     [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
     [0.66, -0.45, 0.08]]
]

class Module():
    def __init__(self):
        pass

    def forward(self, X: np.array) -> np.array:
        raise NotImplementedError

    def backward(self, gradient=None):
        raise NotImplementedError

    def upgrade(self, learning_rate=None):
        pass


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.result = None

    def forward(self, X):
        self.result = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
        return self.result

    def backward(self, gradient=None):
        return (1 - self.result * self.result) * gradient


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.result = None

    def forward(self, X: np.array) -> np.array:
        self.result = 1 / (1 + np.exp(-X))
        return self.result

    def backward(self, gradient=None):
      
        return self.result * (1 - self.result) * gradient
        

class MSELOSS():
    def __init__(self) -> None:
        self.pred = None
        self.gold = None

    def forward(self, pred: np.matrix, gold: np.matrix):
        self.pred = pred
        self.gold = gold
        return 0.5 * np.sum((pred - gold) ** 2)

    def backward(self):
        return -(self.gold - self.pred)

    
class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(1, output_dim)

        self.input = None

    def forward(self, X: np.array) -> np.array:
        self.input = X
        return np.matmul(X, self.W) + self.b
        

    def backward(self, gradient: np.array):
        self.dW = np.matmul(self.input.T, gradient)
        self.db = np.sum(gradient, axis=0, keepdims=True)
        return np.matmul(gradient, self.W.T)
   

    def upgrade(self, learning_rate=None):
 
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate

    def reset(self):
        self.W = np.random.randn(self.input_dim, self.output_dim)
        self.b = np.random.randn(1, self.output_dim)



class MLP(Module):
    def __init__(self, input_dim=3, hidden_dim=10, output_dim=3, learning_rate=0.01):
        super(MLP, self).__init__()

        self.linear1 = Linear(input_dim, hidden_dim)
        self.act1 = Tanh()
        self.linear2 = Linear(hidden_dim, output_dim)
        self.act2 = Sigmoid()

        self.layers: List[Module] = [
            self.linear1,
            self.act1,
            self.linear2,
            self.act2
        ]

        self.learning_rate = learning_rate

    def forward(self, X: np.array) -> np.array:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        
        return output

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(gradient=error)

    def upgrade(self):
        for layer in self.layers:
            layer.upgrade(learning_rate=self.learning_rate)

    def reset(self):
        self.linear1.reset()
        self.linear2.reset()


class DataLoader:
    def __init__(self, datas: List[np.array], labels: List[np.array], sample='batch'):
        self.datas = datas
        self.labels = labels
        self.sample = sample

    def __iter__(self):
        if self.sample == 'batch':
            yield self.datas, self.labels
        else:
            for data, label in zip(self.datas, self.labels):
                yield data.reshape(1, -1), label.reshape(1, -1)

    def __len__(self):
        return len(self.labels)


def train(model: Module, dataloader: DataLoader, epoch:int):
    loss_his, acc_his = [], []
    for i in range(epoch):
        total_loss = 0
        total_correct_num = 0
        for x, y in dataloader:
            logits = model.forward(x)
            
            loss = loss_function.forward(logits, y)
            model.backward(error=loss_function.backward())
            model.upgrade()

            pred = np.argmax(logits, axis=1)
            correct_num = np.sum(y[range(len(y)),pred] == 1)

            total_loss += loss
            total_correct_num += correct_num

        if i % 100 == 0:
            print(total_loss / len(dataloader), total_correct_num / len(dataloader))
    
        loss_his.append(total_loss / len(dataloader))
        acc_his.append(total_correct_num / len(dataloader))
        

    return loss_his, acc_his


def exp_a():
    for dim in [8, 16, 32, 64, 128]:
        model = MLP(hidden_dim=dim)
        loss_his, acc_his = train(model, batch_data_loader, epoch)
        ax[0].plot(loss_his, label=f'dim: {dim}' )
        ax[1].plot(acc_his, label=f'dim: {dim}')


def exp_b():
    for lr in [0.5, 0.1, 0.01]:
        model = MLP(learning_rate=lr)
        loss_his, acc_his = train(model, batch_data_loader, epoch)
        ax[0].plot(loss_his, label=f'learning_rate: {lr}')
        ax[1].plot(acc_his, label=f'learning_rate: {lr}')


def exp_c():
    for data_loader in [batch_data_loader, single_data_loader]:
        model = MLP()
        loss_his, acc_his = train(model, data_loader, epoch)
        ax[0].plot(loss_his, label='loss-' + data_loader.sample)
        ax[1].plot(acc_his, label='acc-' + data_loader.sample)

if __name__ == '__main__':
    model = MLP()
    loss_function = MSELOSS()
    epoch = 500

    X = []
    Y = []
    for i, datas in enumerate(all_datas):
        for data in datas:
            X.append(data)
            Y.append(i)

    X = np.array(X)
   
    Y = np.eye(3)[Y]

    batch_data_loader = DataLoader(X, Y, 'batch')
    single_data_loader = DataLoader(X, Y, 'single')

    plt.rcParams["figure.figsize"] = (8, 10)
    fig, ax = plt.subplots(2)

    # exp_a()
    # exp_b()
    exp_c()

    ax[0].legend()
    ax[0].set_title('loss - epoch')
    ax[1].legend()
    ax[1].set_title('accuracy - epoch')
    plt.show()
