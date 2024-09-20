import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

#MNIST
def load_data(train_batch_size, test_batch_size):
    trainset = datasets.MNIST('data', train=True, download=True, 
                transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.MNIST('data', train=False, download=False, 
                transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

class Model_1 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

class Model_2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 5)
        self.fc2 = nn.Linear(5, 9)
        self.fc3 = nn.Linear(9, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    
class Model_3 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    
class Model_4 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 60)
        self.fc2 = nn.Linear(60, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

class Model_5 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 280)
        self.fc2 = nn.Linear(280, 560)
        self.fc3 = nn.Linear(560, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
        
class Model_6 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 150)
        self.fc2 = nn.Linear(150, 300)
        self.fc3 = nn.Linear(300, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

class Model_7 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

class Model_8 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

class Model_9 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 350)
        self.fc2 = nn.Linear(350, 600)
        self.fc3 = nn.Linear(600, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
        
class Model_10 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 250)
        self.fc2 = nn.Linear(250, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, val):
        val = val.view(val.size(0), -1)
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val


train_batch_size = 100
test_batch_size = 100
train_loader, test_loader = load_data(train_batch_size, test_batch_size)
epochs = 10

def calculate_number_of_params(model):
    return sum(p.numel() for p in model.parameters())

def define_optimizer(model):
    return optim.Adam(model.parameters(), lr = 0.0001)\
    
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
            
            predict = output.argmax(dim=1, keepdim=True)  
            num_correct += predict.eq(target.view_as(predict)).sum().item()
            total += target.size(0)
    
    return costTotal / costCount, round(num_correct/total, 3)

def train_and_test(model, optimizer):
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        train_model(model, optimizer, train_loader, loss_fn)
        train_loss, train_acc = loss_calc(model, train_loader, loss_fn)
        test_loss, test_acc = loss_calc(model, test_loader, loss_fn)
    
    loss_train.append(train_loss)
    loss_test.append(test_loss)
    
    params.append(calculate_number_of_params(model))
    print("Train loss: ", train_loss)
    print("Test loss: ", test_loss)
    
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

loss_train = []
loss_test = []
test_accuracy = []
train_accuracy = []
params = []

# Run all of the models
model1 = Model_1()
print("Model 1 - Number of parameters: ", (calculate_number_of_params(model1)))
optimizer = define_optimizer(model1)
train_and_test(model1, optimizer)

model2 = Model_2()
print("Model 2 - Number of parameters: ", (calculate_number_of_params(model2)))
optimizer = define_optimizer(model2)
train_and_test(model2, optimizer)

model3 = Model_3()
print("Model 3 - Number of parameters: ", (calculate_number_of_params(model3)))
optimizer = define_optimizer(model3)
train_and_test(model3, optimizer)

model4 = Model_4()
print("Model 4 - Number of parameters: ", (calculate_number_of_params(model4)))
optimizer = define_optimizer(model4)
train_and_test(model4, optimizer)

model5 = Model_5()
print("Model 5 - Number of parameters: ", (calculate_number_of_params(model5)))
optimizer = define_optimizer(model5)
train_and_test(model5, optimizer)

model6 = Model_6()
print("Model 6 - Number of parameters: ", (calculate_number_of_params(model6)))
optimizer = define_optimizer(model6)
train_and_test(model6, optimizer)

model7 = Model_7()
print("Model 7 - Number of parameters: ", (calculate_number_of_params(model7)))
optimizer = define_optimizer(model7)
train_and_test(model7, optimizer)

model8 = Model_8()
print("Model 8 - Number of parameters: ", (calculate_number_of_params(model8)))
optimizer = define_optimizer(model8)
train_and_test(model8, optimizer)

model9 = Model_9()
print("Model 9 - Number of parameters: ", (calculate_number_of_params(model9)))
optimizer = define_optimizer(model9)
train_and_test(model9, optimizer)

model10 = Model_10()
print("Model 10 - Number of parameters: ", (calculate_number_of_params(model10)))
optimizer = define_optimizer(model10)
train_and_test(model10, optimizer)

plt.figure(figsize=(12,6))
plt.scatter(params, loss_train, color = "g")
plt.scatter(params, loss_test, color = "b")
plt.title('Loss Comparision Over 10 Models')
plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel('Number of Parameters')
plt.ylabel('Loss')
plt.show()
plt.savefig("NumParams-loss.png", format="png", dpi=300)

plt.figure(figsize=(12,6))
plt.scatter(params, train_accuracy, color = "g")
plt.scatter(params, test_accuracy, color = "b")
plt.title('Accuracy Comparision Over 10 Models')
plt.legend(['Train Acuracy', 'Test Accuracy'])
plt.xlabel('Number of Parameters')
plt.ylabel('Loss')
#plt.show()
plt.savefig("NumParams-acc.png", format="png", dpi=300)