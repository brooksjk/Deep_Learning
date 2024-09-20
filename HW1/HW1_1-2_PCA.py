import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.manual_seed(1)

def load_data(train_batch_size, test_batch_size):
    trainset = datasets.MNIST('data', train=True, download=True, 
                transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    testset = datasets.MNIST('data', train=False, download=False, 
                transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

def calculate_loss(model, loader, loss_fn):
    correct = 0
    total = 0
    costTotal = 0
    costCounter = 0
    with torch.no_grad():
        for batch in loader:
            data, target = batch
            output = model(data.view(-1, 784))
            cost = loss_fn(output, target)
            costTotal += cost
            costCounter += 1
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == target[i]:
                    correct += 1
                total += 1
    return costTotal / costCounter, round(correct/total, 3)

def trainFunc(model, num_epochs):
    model.train()
    epoch = 0 
    
    df = pd.DataFrame()
    for epoch in range(num_epochs):
        epoch += 1
        for _, (images, labels) in enumerate(train_loader):  
            images, labels = Variable(images), Variable(labels)     
            optimizer.zero_grad()
            prediction = model(images.view(-1, 784))
            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()

        temp_df = pd.DataFrame()
        for name, parameter in model.named_parameters():
            if 'weight' in name:
                weights = torch.nn.utils.parameters_to_vector(parameter).detach().numpy() 
                temp_df = pd.concat([temp_df, pd.DataFrame(weights).T], axis=1)
        df = pd.concat([df, temp_df], axis=0)
        train_loss, train_acc = calculate_loss(M, train_loader, loss_fn)
        test_loss, test_acc = calculate_loss(M, test_loader, loss_fn)

        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    return df

def pca_torch(data, n_components):
    data = torch.tensor(data, dtype=torch.float32)
    mean = data.mean(dim=0)
    data_centered = data - mean
    covariance_matrix = torch.mm(data_centered.T, data_centered) / (data_centered.size(0) - 1)
    
    # Use torch.linalg.eig
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    
    # Only keep the real part of eigenvalues and eigenvectors
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Sort eigenvectors by the eigenvalues in descending order
    _, indices = eigenvalues.sort(descending=True)
    eigenvectors = eigenvectors[:, indices]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Project the data onto the top n_components eigenvectors
    projected_data = torch.mm(data_centered, eigenvectors)
    
    return projected_data.numpy()

train_batch_size = 1000
test_batch_size = 1000
train_loader, test_loader = load_data(train_batch_size, test_batch_size)
train_loss_arr = []
test_loss_arr = []
train_acc_arr = []
test_acc_arr = []
max_epochs = 45
all_df = pd.DataFrame()
columns = ["x", "y", "Times"]

loss_fn = nn.CrossEntropyLoss()

for count in range(8):
    print("Training: " + str(count))
    M = DNN()
    optimizer = torch.optim.Adam(M.parameters(), lr=0.0004, weight_decay=1e-4)
    model_name1 = "Training: " + str(count)    
    temp_df = trainFunc(M, max_epochs)
    all_df = pd.concat([all_df, temp_df], axis=0)

df = all_df
df = np.array(df)

new_data = pca_torch(df, n_components=2)
df = pd.DataFrame(new_data, columns=['x', 'y'])
df['Accuracy'] = train_acc_arr
df['Loss'] = train_loss_arr
final_df = df.iloc[::3, :]
for i in range(120):
    m = list(final_df['Accuracy'])[i]
    plt.scatter(final_df['x'][i*3], final_df['y'][i*3], marker=f'${m}$')
    
plt.title("PCA for model")
plt.savefig("pca_plot.png", format="png", dpi=300)

layer_1 = all_df.iloc[:, 0:7840]
df = layer_1
df = np.array(df)

new_data = pca_torch(df, n_components=2)
df = pd.DataFrame(new_data, columns=['x', 'y'])
df['Accuracy'] = train_acc_arr
df['Loss'] = train_loss_arr
final_df = df.iloc[::3, :]
for i in range(120):
    m = list(final_df['Accuracy'])[i]
    plt.scatter(final_df['x'][i*3], final_df['y'][i*3], marker=f'${m}$')

plt.title("PCA for 1 layer")
plt.savefig("pca_plot_layer_1.png", format="png", dpi=300)