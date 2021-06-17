import sys 

import sklearn.datasets
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt 

print(f"python: {sys.version}")
print(f"pytorch: {torch.__version__}")

mnist = sklearn.datasets.fetch_openml('mnist_784',data_home="mnist_784")

x_train = torch.tensor(mnist.data[:60000],dtype=torch.float) / 255
y_train = torch.tensor([int(x) for x in mnist.target[:60000]])
x_test = torch.tensor(mnist.data[60000:],dtype=torch.float) / 255
y_test = torch.tensor([int(x) for x in mnist.target[60000:]])

fig, axes = plt.subplots(2,4,constrained_layout = True)

for i, ax in enumerate(axes.flat):
  ax.imshow(1 - x_train[i].reshape((28, 28)), cmap="gray", vmin=0, vmax=1)
  ax.set(title=f"{y_train[i]}")
  ax.set_axis_off()

def log_softmax(x):
  return x-x.exp().sum(dim=-1).log().unsqueeze(-1)

def model(x,weights,bias):
  return log_softmax(x @ weights + bias)

def neg_likelihood(log_pred, y_true):
  return -log_pred[torch.arange(y_true.size()[0]),y_true].mean()

def accuracy(log_pred, y_true):
  y_pred = torch.argmax(log_pred, dim =1 )
  return (y_pred == y_true).to(torch.float).mean()

def print_loss_accuracy(log_pred, y_true,loss_function):
  with torch.no_grad():
    print(f"Loss:{neg_likelihood(log_pred,y_true):.6f}")
    print(f"Accuray:{100 * accuracy(log_pred, y_true).item():.2f}%")

loss_function = neg_likelihood

batch_size = 100
learning_rate = 0.5
n_epochs = 5 

weights = torch.randn(784,10,requires_grad=True)
bias = torch.randn(10,requires_grad=True)

for epoch in range(n_epochs):
  #Batch 반복
  for i in range(x_train.size()[0] // batch_size):
    start_index = i * batch_size
    end_index = start_index + batch_size
    x_batch = x_train[start_index:end_index]
    y_batch_true = y_train[start_index:end_index]

    #forward
    y_batch_log_pred = model(x_batch, weights, bias)
    loss = loss_function(y_batch_log_pred,y_batch_true)

    #backward
    loss.backward()

    #update
    with torch.no_grad():
      weights.sub_(learning_rate * weights.grad)
      bias.sub_(learning_rate * bias.grad)

    #zero the parameter gradients
    weights.grad.zero_()
    bias.grad.zero_()

  with torch.no_grad():
    y_test_log_pred = model(x_test,weights,bias)
  print(f"End of epoch{epoch + 1}")
  print_loss_accuracy(y_test_log_pred, y_test, loss_function)
  print("---")
  
  #PyTorch 활용
  
  class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

loss_function = F.cross_entropy
  
batch_size = 100
learning_rate = 0.5
n_epochs = 5

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

model = Model()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    for x_batch, y_batch_true in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        y_batch_log_pred = model(x_batch)
        loss = loss_function(y_batch_log_pred, y_batch_true)

        # Backword
        loss.backward()

        # Update
        optimizer.step()

    with torch.no_grad():
        y_test_log_pred = model(x_test)
    print(f"End of epoch {epoch + 1}")
    print_loss_accuracy(y_test_log_pred, y_test, loss_function)
    print("---")
