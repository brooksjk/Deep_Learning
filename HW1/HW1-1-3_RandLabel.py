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

# using CIFAR-10
def load_data(train_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Resize((32, 32)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                         ])),
        batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])),
        batch_size=test_batch_size, shuffle=False)

    return (train_loader, test_loader)

# part 1 - random labels

train_batch_size = 100
test_batch_size = 100
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

#randomize labels
rand_label = torch.tensor(np.random.randint(0, 10, (len(train_loader)),))
train_loader.targets = rand_label

# CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)        
        self.fc2 = nn.Linear(128, 10)                

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   
        x = x.view(x.size(0), -1)                    
        x = F.relu(self.fc1(x))                    
        x = self.fc2(x)                             
        return x

def train(model, optimizer, train_loader):
    model.train()
    num_correct = 0
    train_loss = 0
    count = 0

    for epoch, (data, target) in enumerate(train_loader):
        count += 1
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data

        predict = np.argmax(output.data, axis=1)
        
        num_correct += np.equal(predict, target.data).sum()
    
    train_loss = (train_loss * 100) / len(train_loader.dataset)
    
    accuracy = 100.0 * num_correct / len(train_loader.dataset)

    print('TRAINING: Avg. loss: {:.4f}, Accuracy: {:.0f}%'.format(train_loss, accuracy))
    
    return train_loss, accuracy

def test(model, epoch, test_loader):
    model.eval()
    num_correct = 0
    test_loss = 0
    

    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        
        output = model(data)
        loss = loss_fn(output, target)
        test_loss += loss.data

        predict = np.argmax(output.data, axis = 1)
        num_correct = num_correct + np.equal(predict, target.data).sum()

    test_loss = (test_loss * 100) / len(test_loader.dataset)

    accuracy = 100.0 * num_correct / len(test_loader.dataset)
    
    print('TEST: Epoch {} , Avg. loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, test_loss, accuracy))
    
    return test_loss

model = CNN()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loss = []
test_loss = []
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 100

for epoch in range(1, epochs + 1):
    loss, acc = train(model, optimizer, train_loader)
    train_loss.append(loss)
    loss_test = test(model, epoch, test_loader)
    test_loss.append(loss_test)

plt.figure(figsize=(10,6))
plt.plot(train_loss, color = "g")
plt.plot(test_loss, color = "b")
plt.title('Random Label Train/Test Loss Comparision')
plt.legend(['Train Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig("Rand_Label.png", format="png", dpi=300)


