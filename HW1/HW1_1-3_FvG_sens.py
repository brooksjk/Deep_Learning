import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def load_data(train_batch_size, test_batch_size):
    trainset = datasets.MNIST('data', train=True, download=True, 
                transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.MNIST('data', train=False, download=False, 
                transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

class model(nn.Module):
  def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        
  def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def sensitivity(model):
    norm_total = 0
    count = 0
            
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = p.grad
            norm = torch.linalg.norm(grad).numpy()
            norm_total += norm
            count += 1

    return norm_total / count

def define_optimizer(model):
    return optim.SGD(model.parameters(), lr = 1e-4, momentum=0.9, nesterov=True)
    
def train_model(model, optimizer, train_loader, loss_fn):
    model.train()

    for batch, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def loss_calc(model, loader, loss_fn):
    num_correct = 0
    total = 0
    costTotal = 0
    costCount = 0
    
    with torch.no_grad():
        for batch in loader:
            
            data, target = batch
            output = model(data)
            
            cost = loss_fn(output, target)
            costTotal += cost
            costCount += 1
            
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == target[i]:
                    num_correct += 1
                total += 1
                
    return costTotal / costCount, round(num_correct/total, 3)

def train_and_test(model, optimizer, epochs):
    
    for epoch in range(1, epochs + 1):
        train_model(model, optimizer, train_loader, loss_fn)
        train_loss, train_acc = loss_calc(model, train_loader, loss_fn)
        test_loss, test_acc = loss_calc(model, test_loader, loss_fn)
    
        loss_train.append(train_loss)
        loss_test.append(test_loss)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
    
        print("Train loss: ", train_loss)
        print("Test loss: ", test_loss)
    
    return loss_train, loss_test, train_accuracy, test_accuracy

# Store final results for each batch size
final_loss_train = []
final_loss_test = []
final_train_accuracy = []
final_test_accuracy = []
sens = []

batch_size=[10, 200, 1000, 10000, 20000]

epochs = 30
# For each batch size, train the model and store the final results
for batch in batch_size:
    
    # Reset lists for each batch size
    loss_train = []
    loss_test = []
    train_accuracy = []
    test_accuracy = []
    
    torch.manual_seed(1)  # Ensure reproducibility

    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Load data with current batch size
    train_loader, test_loader = load_data(batch, batch)
    
    # Initialize model and optimizer
    model_sens = model()
    optimizer = define_optimizer(model_sens)
    
    # Train and test the model
    loss_train, loss_test, train_acc, test_acc = train_and_test(model_sens, optimizer, epochs)
    
    # For each batch size, collect the final epoch's results
    final_loss_train.append(loss_train[-1])  # Final training loss
    final_loss_test.append(loss_test[-1])    # Final testing loss
    final_train_accuracy.append(train_acc[-1])  # Final training accuracy
    final_test_accuracy.append(test_acc[-1])    # Final testing accuracy
    sens.append(sensitivity(model_sens))  # Sensitivity for the model

# Plot Loss against Batch Size
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(batch_size, final_loss_train, color="g", linestyle='dashed', label='Train Loss')
ax.plot(batch_size, final_loss_test, color="g", label='Test Loss')
ax2 = ax.twinx()
ax2.plot(batch_size, sens, color="b", label='Sensitivity')
ax2.set_yscale('log') 
plt.title('Model Loss and Sensitivity vs Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Loss')
ax2.set_ylabel('Sensitivity')
ax.set_xlim([min(batch_size), max(batch_size)])
plt.legend()
plt.savefig("sens_loss_sensitivity.png", format="png", dpi=300)
plt.show()

# Plot Accuracy against Batch Size
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(batch_size, final_train_accuracy, color="g", linestyle='dashed', label='Train Accuracy')
ax.plot(batch_size, final_test_accuracy, color="g", label='Test Accuracy')
ax2 = ax.twinx()
ax2.plot(batch_size, sens, color="b", label='Sensitivity')
ax2.set_yscale('log') 
plt.title('Model Accuracy and Sensitivity vs Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Accuracy')
ax2.set_ylabel('Sensitivity')
ax.set_xlim([min(batch_size), max(batch_size)])
plt.legend()
plt.savefig("sens_accuracy_sensitivity.png", format="png", dpi=300)
plt.show()