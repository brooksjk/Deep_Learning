import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import statistics as s
import copy

torch.manual_seed(1)

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

def calculate_number_of_params(model):
    return sum(p.numel() for p in model.parameters())

def define_optimizer(model, learn):
    return optim.SGD(model.parameters(), lr = learn, momentum=0.9, nesterov=True)

'''
def train_model(model, num_epochs, batch_size, loss_fn, optimizer):

    train_loader, test_loader =  load_data(batch_size, batch_size)
    
    model.train()
    total_steps = len(train_loader)
    train_loss = [] 
    train_epoch = []
    train_accuracy = []
    convergence = False
    AvgLoss = []
    AvgAccuracy = []
    epoch = 0

   

    while convergence == False:
        epoch += 1
        num_correct = 0
        samples = 0
        lossSum =0
        totalaccuracy =0

        for i, (images, labels) in enumerate(train_loader):  
            
            images, labels = Variable(images),Variable(labels)
            optimizer.zero_grad()
            prediction = model(images)
            loss = loss_fn(prediction, labels)
            lossSum += loss.detach().numpy()
            loss.backward()
            
            optimizer.step()

            _, predicted = torch.max(prediction.data, 1)
            samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()
            accuracy = 100.0 * num_correct / samples
            totalaccuracy += accuracy

            train_loss.append(loss.item())
            train_accuracy.append(accuracy)
            train_epoch.append(epoch)
            
            if epoch == num_epochs:
                #print("Max Epoch Reached")
                convergence = True
            elif (epoch > 5) and  (train_loss[-1] < 0.001):
                if abs(train_loss[-3] - train_loss[-2]) < 1.0e-05 and abs(train_loss[-2] - train_loss[-1]) < 1.0e-05:
                    #print("Convergeance reached for loss:",train_loss[-1])
                    convergence = True

            
        epoch_accuracy = totalaccuracy/(i+1)
        AvgLoss.append(lossSum/total_steps)    
        AvgAccuracy.append(epoch_accuracy)

    return train_epoch, train_loss, train_accuracy, AvgLoss, AvgAccuracy
     
def test_model(model, loss_fn, batch_size):

    train_loader, test_loader = load_data(batch_size, batch_size)

    model.eval()

    with torch.no_grad():
        num_correct = 0
        samples = 0
        testLoss = 0
        count = 0
        for images, labels in test_loader:
            
            count +=1
            
            images, labels = Variable(images),Variable(labels)
            
            prediction = model(images)
            testLoss += loss_fn(prediction,labels).item()
            _, predicted = torch.max(prediction.data, 1)
            samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()
            

    Test_loss = testLoss/count
    Test_accuracy = 100.0 * num_correct / samples
    
    return Test_loss, Test_accuracy

def calc_alpha(model, params1, params2, loss_fn, alpha, optimizer, num_epochs):

    theta_vals =[]

    batch_size = 100

    train_loader, test_loader = load_data(batch_size, batch_size)

    for i in range(len(alpha)):
        theta = (1-alpha[i]) * params1 + alpha[i] * params2
        theta_vals.append(theta)
        
    alpha_train_loss = []
    alpha_train_accuracy = []
    alpha_test_loss = []
    alpha_test_accuracy = []
    
    for i in range(len(theta_vals)):
        #torch.manual_seed(1)

        model_copy = copy.deepcopy(model)
        theta = (1-alpha[i]) * params1 + alpha[i] * params2
        torch.nn.utils.vector_to_parameters(theta, model_copy.parameters())
        
        train_epoch, train_loss, train_accuracy, AvgLoss, AvgAccuracy = train_model(model_copy, num_epochs, batch_size, loss_fn, optimizer)
        
        test_loss, test_accuracy = test_model(model_copy, loss_fn, batch_size)

        alpha_train_loss.append(np.mean(train_loss))
        alpha_train_accuracy.append(np.mean(train_accuracy))
        alpha_test_loss.append(test_loss)
        alpha_test_accuracy.append(test_accuracy)

        print(f"Train Loss: {train_loss}, Test Loss: {test_loss}, Train Acc: {train_accuracy}, Test Acc: {test_accuracy}")

    return alpha_train_loss, alpha_train_accuracy, alpha_test_loss, alpha_test_accuracy
'''
def train(model, optimizer, train_loader):
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images), Variable(labels)
        optimizer.zero_grad()

        # Forward propagation
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

def calculate_loss(model, loader, loss_fn):
    correct = 0
    total = 0
    costTotal = 0
    costCounter = 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch
            output = model(data)
            cost = loss_fn(output, target)
            costTotal += cost
            costCounter += 1
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == target[i]:
                    correct += 1
                total += 1
    return costTotal / costCounter, round(correct/total, 3)
    
def compute(model_calc, optimizer, batch_size):
    loss_train = []
    loss_test = []
    test_acc = []
    train_acc = []

    train_loader, test_loader = load_data(batch_size, batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, num_epochs + 1):
        train(model_calc, optimizer, train_loader)
        tr_loss, tr_acc = calculate_loss(model_calc, train_loader, loss_fn)
        t_loss, t_acc = calculate_loss(model_calc, test_loader, loss_fn)
        print("Model Train loss: ", tr_loss)
        print("Model Train acc: ", tr_acc)
        loss_train.append(tr_loss)
        loss_test.append(t_loss)
        train_acc.append(tr_acc)
        test_acc.append(t_acc)
    return loss_train, loss_test, train_acc, test_acc

loss_fn = torch.nn.CrossEntropyLoss()

model_opt = model()

#optimizer_1 = define_optimizer(model_opt, 1e-3)

#optimizer_2 = define_optimizer(model_opt, 1e-2)

batch_size_1 = 64

batch_size_2 = 1024

num_epochs = 30

#learning rate 1e-3
#64 batch size
torch.manual_seed(1)

m1 = model()
optimizer_1 = define_optimizer(m1, 1e-3)
#m1_epoch, m1_train_loss, m1_train_accuracy, m1_AvgLoss, m1_AvgAccuracy = train_model(m1, num_epochs, batch_size_1, loss_fn, optimizer_1)
m1_train_loss, m1_test_loss, m1_train_accuracy, m1_test_accuracy = compute(m1, optimizer_1, batch_size_1)

m1_params = torch.nn.utils.parameters_to_vector(m1.parameters())

#1024 batch size
torch.manual_seed(1)

m2 = model()
optimizer_2 = define_optimizer(m2, 1e-3)
#m2_epoch, m2_train_loss, m2_train_accuracy, m2_AvgLoss, m2_AvgAccuracy = train_model(m2, num_epochs, batch_size_2, loss_fn, optimizer_1)
m2_train_loss, m2_test_loss, m2_train_accuracy, m2_test_accuracy = compute(m2, optimizer_2, batch_size_2)

m2_params = torch.nn.utils.parameters_to_vector(m2.parameters())

plt.figure(figsize=(12,6))
plt.plot(m1_train_loss, color = "g")
plt.plot(m2_train_loss, color = "b")
plt.title('Model Loss Comparision - LR = 1e-3')
plt.legend(['Batch Size 64', 'Batch Size 1024'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("1e-3_Loss.png", format="png", dpi=300)

plt.figure(figsize=(12,6))
plt.plot(m1_train_accuracy, color = "g")
plt.plot(m2_train_accuracy, color = "b")
plt.title('Model Accuracy Comparision - LR = 1e-3')
plt.legend(['Batch Size 64', 'Batch Size 1024'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("1e-3_Accuracy.png", format="png", dpi=300)

theta_vals = []
alpha_train_loss = []
alpha_test_loss = []
alpha_train_accuracy  = []
alpha_test_accuracy = []

model_alpha = model()

alpha = np.linspace(-2.0, 2.0, num = 50)

for i in range(len(alpha)):
    theta = (1-alpha[i]) * m1_params + alpha[i] * m2_params
    theta_vals.append(theta)
    
for i in range (len(theta_vals)):
    torch.manual_seed(1)
    theta = (1-alpha[i])* m1_params + alpha[i] * m2_params
    #model_new = model()
    torch.nn.utils.vector_to_parameters(theta, model_alpha.parameters())
        #loss_func = nn.CrossEntropyLoss()

    batch_size = 100

    train_loader, test_loader = load_data(batch_size, batch_size)

    temp = []
    for param in model_alpha.parameters():
        temp.append(torch.numel(param))

    train_loss, train_acc = calculate_loss(model_alpha, train_loader, loss_fn)
    test_loss, test_acc = calculate_loss(model_alpha, test_loader, loss_fn)
    alpha_train_loss.append(train_loss)
    alpha_train_accuracy.append(train_acc)
    alpha_test_loss.append(test_loss)
    alpha_test_accuracy.append(test_acc)

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(alpha_train_loss, color = "g", label='Train Loss')
ax.plot(alpha_test_loss, color = "b", label='Test Loss')
ax2 = ax.twinx()
ax2.plot(alpha_train_accuracy, color = "g", linestyle='dashed', label='Train Accuracy')
ax2.plot(alpha_test_accuracy, color = "b", linestyle='dashed', label='Test Accuracy')
ax.set_yscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
plt.legend(loc='upper left')
plt.title('Train/Test Comparison - LR = 1e-3')
plt.tight_layout()
plt.savefig("1e-3_Interpol.png", format="png", dpi=300)

#Learning rate 1e-2

#64 batch size
m1_2 = model()
optimizer_1_2 = define_optimizer(m1_2, 1e-2)
#m1_epoch_2, m1_train_loss_2, m1_train_accuracy_2, m1_AvgLoss_2, m1_AvgAccuracy_2 = train_model(m1_2, num_epochs, batch_size_1, loss_fn, optimizer_2)
m1_train_loss_2, m1_test_loss_2, m1_train_accuracy_2, m1_test_accuracy_2 = compute(m1_2, optimizer_1_2, batch_size_1)

m1_params_2 = torch.nn.utils.parameters_to_vector(m1_2.parameters())

#1024 batch size
torch.manual_seed(1)

m2_2 = model()
optimizer_2_2 = define_optimizer(m2_2, 1e-2)
#m2_epoch_2, m2_train_loss_2, m2_train_accuracy_2, m2_AvgLoss_2, m2_AvgAccuracy_2 = train_model(m2_2, num_epochs, batch_size_2, loss_fn, optimizer_2)
m2_train_loss_2, m2_test_loss_2, m2_train_accuracy_2, m2_test_accuracy_2 = compute(m2_2, optimizer_2_2, batch_size_2)

m2_params_2 = torch.nn.utils.parameters_to_vector(m2_2.parameters())

plt.figure(figsize=(12,6))
plt.plot(m1_train_loss_2, color = "g")
plt.plot(m2_train_loss_2, color = "b")
plt.title('Model Loss Comparision - LR = 1e-2')
plt.legend(['Batch Size 64', 'Batch Size 1024'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("1e-2_Loss.png", format="png", dpi=300)

plt.figure(figsize=(12,6))
plt.plot(m1_train_accuracy_2, color = "g")
plt.plot(m2_train_accuracy_2, color = "b")
plt.title('Model Accuracy Comparision - LR = 1e-2')
plt.legend(['Batch Size 64', 'Batch Size 1024'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("1e-2_Accuracy.png", format="png", dpi=300)

theta_vals_2 = []
alpha_train_loss_2 = []
alpha_test_loss_2 = []
alpha_train_accuracy_2  = []
alpha_test_accuracy_2 = []

model_alpha_2 = model()

alpha_2 = np.linspace(-2.0, 2.0, num = 50)

for i in range(len(alpha_2)):
    theta_2 = (1-alpha_2[i]) * m1_params_2 + alpha_2[i] * m2_params_2
    theta_vals_2.append(theta_2)
    
for i in range (len(theta_vals_2)):
    torch.manual_seed(1)
    theta_2 = (1-alpha_2[i])* m1_params_2 + alpha_2[i] * m2_params_2
    #model_new = model()
    torch.nn.utils.vector_to_parameters(theta_2, model_alpha_2.parameters())
        #loss_func = nn.CrossEntropyLoss()

    batch_size = 100

    train_loader, test_loader = load_data(batch_size, batch_size)

    temp = []
    for param in model_alpha_2.parameters():
        temp.append(torch.numel(param))

    train_loss_2, train_acc_2 = calculate_loss(model_alpha_2, train_loader, loss_fn)
    test_loss_2, test_acc_2 = calculate_loss(model_alpha_2, test_loader, loss_fn)
    alpha_train_loss_2.append(train_loss_2)
    alpha_train_accuracy_2.append(train_acc_2)
    alpha_test_loss_2.append(test_loss_2)
    alpha_test_accuracy_2.append(test_acc_2)

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(alpha_train_loss_2, color = "g", label='Train Loss')
ax.plot(alpha_test_loss_2, color = "b", label='Test Loss')
ax2 = ax.twinx()
ax2.plot(alpha_train_accuracy_2, color = "g", linestyle='dashed', label='Train Accuracy')
ax2.plot(alpha_test_accuracy_2, color = "b", linestyle='dashed', label='Test Accuracy')
ax.set_yscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
plt.legend(loc='upper left')
plt.title('Train/Test Comparison - LR = 1e-2')
plt.tight_layout()
plt.savefig("1e-2_Interpol.png", format="png", dpi=300)