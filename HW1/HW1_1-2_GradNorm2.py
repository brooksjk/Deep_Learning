import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from autograd_lib import autograd_lib
import copy

torch.manual_seed(1)

def training_model(model, x, y):
    max_epoch_value = 2000
    epoch_list = []
    loss_list = []
    grad_list = []
    convergence = False 
    epoch_count = 0
    while convergence == False:
        epoch_count += 1
        prediction_model = model(x)
        loss = loss_function(prediction_model, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_list.append(epoch_count)
        loss_list.append(loss.detach().numpy())

        grad_all = 0.0
        for p in model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all += grad
        grad_norm = grad_all ** 0.5

        grad_list.append(grad_norm)
        
        if epoch_count % 100 == 0:
            print(f'epoch value = {epoch_count} loss value = {loss.item():.6f}')
        
        if epoch_count == max_epoch_value:
            convergence = True
            print('max reached')

        elif (epoch_count > 5) and (loss_list[-1] < 0.001):
            if (abs(loss_list[-3] - loss_list[-2]) < 1.0e-05) and (abs(loss_list[-2] - loss_list[-1]) < 1.0e-05):
                convergence = True
                print('convergence reached')


    return epoch_list, loss_list, prediction_model, grad_list

# function used is sin(5*pi*x)/5*pi*x
x = np.expand_dims(np.arange(-1, 1, 0.01), 1)
y = np.sin(5 * np.pi * x) / (5 * np.pi * x) 

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color="green")
plt.title('Function Plot')
plt.grid(True) 
#plt.show()
plt.savefig('GradNorm-Function.png', format='png', dpi=700)

class model(nn.Module):
    def __init__(self,):
        super(model, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 18)
        self.linear3 = nn.Linear(18, 15)
        self.linear4 = nn.Linear(15, 4)
        self.predict = nn.Linear(4, 1)
    
    def forward(self,x):
        x = nn.functional.leaky_relu(self.linear1(x))
        x = nn.functional.leaky_relu(self.linear2(x))
        x = nn.functional.leaky_relu(self.linear3(x))
        x = nn.functional.leaky_relu(self.linear4(x))
        x = self.predict(x)
        return x
 
model_predict = model()

autograd_lib.register(model_predict)

optimizer = torch.optim.RMSprop(model_predict.parameters(), lr = 1e-3, weight_decay = 1e-4)
loss_function = torch.nn.MSELoss() 
pytorch_params = sum(p.numel() for p in model_predict.parameters())
print(pytorch_params)
model_epoch, model_loss, model_prediction, model_grad = training_model(model_predict, x, y)

plt.figure(figsize=(8,4))
plt.plot(model_epoch, model_grad, color = 'g', label = 'Gradient Norm')
plt.plot(model_epoch, model_loss, color = 'b', label = 'Loss')
plt.title('Gradient Norm and Loss During Training')
plt.xlabel("Epoch")
plt.ylabel("Loss / Gradient Norm Value")
plt.legend()
#plt.show()
plt.savefig('GradNorm-Loss.png', format='png', dpi=700)

y_function = lambda x: torch.sin(5 * np.pi * x) / (5 * np.pi * x)
num_of_rows = 300
x = torch.unsqueeze(torch.linspace(-1, 1, num_of_rows), dim=1)
y = y_function(x)

dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

class Math_Regress(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, x):
        return self.regressor(x)
        
    def step_training(self, batch, loss_fn):
        inputs, targets = batch
        out = self(inputs)
        loss = loss_fn(out, targets)
        return loss

    def step_validation(self, batch, loss_fn):
        inputs, targets = batch
        out = self(inputs)
        loss = loss_fn(out, targets)
        return {'val_loss': loss.detach()}

    def finalize_validation(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def train_step(self, batch, loss_fn):
        inputs, targets = batch
        out = self(inputs)
        loss = loss_fn(out, targets)
        return {'train_loss': loss.detach()}
"""
# Define GradientNormLoss if needed (not used in training here)
class GradientNormLoss(nn.Module):
    def __init__(self, base_loss_fn, alpha=1.0):
        super(GradientNormLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.alpha = alpha 
        
    def forward(self, model, inputs, targets):
        # Compute the base loss
        predictions = model(inputs)
        loss = self.base_loss_fn(predictions, targets)
        
        loss.backward(retain_graph=True) 
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        total_loss = loss + self.alpha * grad_norm
        return total_loss, loss.item(), grad_norm
"""
def calculate_gradient_and_hessian(model, loss_function, train, target):
    model.train()
    model.zero_grad()
    output = model(train)
    loss = loss_function(output, target)
    loss.backward()

    grads = []
    for p in model.children():  # Loop over layers directly
        if isinstance(p, nn.Linear):
            param_norm = p.weight.grad.norm(2).item()
            grads.append(param_norm)

    grad_mean = np.mean(grads)
    return grad_mean


def store_activations(layer, A, _):
    activations[layer] = A


def calculate_hessian(layer, _, B):
    A = activations[layer]
    BA = torch.einsum('nl,ni->nli', B, A) 
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)


def get_min_ratio(model, loss_function, train, target):
    model.zero_grad()

    with autograd_lib.module_hook(store_activations):
        output = model(train)
        loss = loss_function(output, target)

    with autograd_lib.module_hook(calculate_hessian):
        autograd_lib.backward_hessian(output, loss='LeastSquares')

    layer_hess = list(hess.values())
    min_ratio = []

    for h in layer_hess:
        size = h.shape[0] * h.shape[1]
        h = h.reshape(size, size)
        h_eig = torch.linalg.eigvalsh(h)  
        num_greater = torch.sum(h_eig > 0).item()
        min_ratio.append(num_greater / len(h_eig))

    return np.mean(min_ratio)


def get_norm_min_ratio(model, loss_function, grad_counter):
    gradient_norm = calculate_gradient_and_hessian(model, loss_function, x, y)
    minimum_ratio = get_min_ratio(model, loss_function, x, y)

    #if grad_counter % 100 == 0:
        #print('gradient norm: {}, minimum ratio: {}'.format(gradient_norm, minimum_ratio))
    
    result = {}
    result["grad_norm"] = gradient_norm
    result["ratio"] = minimum_ratio
    
    return result

def evaluate_model(model, loss_function, val_loader):
    model.eval()  
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():  
        for batch in val_loader:
            inputs, targets = batch
            output = model(inputs)
            loss = loss_function(output, targets)
            total_loss += loss.item()
            num_batches += 1
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {'val_loss': average_loss}

def train_model(epochs, model, data_loader, loss_function, optimizer):
    loss_hist = []
    grad_norm_per_epoch = []  
    grad_count = 0

    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        for batch in data_loader:
            inputs, targets = batch
            prediction = model(inputs)
            loss = loss_function(prediction, targets)
            loss.backward()

        grad_count += 1
        
        grad_norm_min_ratio = get_norm_min_ratio(model, loss_function, grad_count)
        grad_norm_per_epoch.append(grad_norm_min_ratio)
    
        optimizer.step()
        optimizer.zero_grad()

        result = evaluate_model(model, loss_function, data_loader)
        loss_hist.append(result)

    return loss_hist, grad_norm_per_epoch, model


def train_with_grad(num_runs, model_class, data_loader, loss_function, optimizer_class, optimizer_kwargs, num_epochs):
    all_loss_hist = []
    all_grad_norms = []

    for run in range(num_runs):
        print(f"Training run {run + 1} out of {num_runs}")

        model_copy = model_class()

        optimizer_copy = optimizer_class(model_copy.parameters(), **optimizer_kwargs)

        autograd_lib.register(model_copy)

        global activations, hess
        activations = defaultdict(int)
        hess = defaultdict(float)

        loss_hist, grad_norm_per_epoch, trained_model = train_model(
            num_epochs, model_copy, data_loader, loss_function, optimizer_copy
        )

        all_loss_hist.append(loss_hist)
        all_grad_norms.append(grad_norm_per_epoch)

    return all_loss_hist, all_grad_norms

model_class = Math_Regress
optimizer_class = torch.optim.RMSprop
optimizer_kwargs = {'lr': 1e-4, 'weight_decay': 1e-4}
loss_function = nn.MSELoss()

num_epochs = 100
num_runs = 100

all_loss_hist, all_grad_norms = train_with_grad(
    num_runs,
    model_class,
    data_loader,
    loss_function,
    optimizer_class,
    optimizer_kwargs,
    num_epochs
)

#print(all_histories[0]) 
#print(all_grad_norms[0])

val_losses_1 = [
    epoch_dict['val_loss']
    for run in all_loss_hist
    for epoch_dict in run
]

min_ratio_val = [
    epoch_dict['ratio']
    for run in all_grad_norms
    for epoch_dict in run
]

plt.figure(figsize=(10, 10))
plt.scatter(val_losses_1, min_ratio_val, color='purple')
plt.xscale('log')
plt.ylabel('Minimum Ratio')
plt.xlabel('Loss')
plt.title('Loss vs. Minimum Ratio')
plt.savefig('GradNorm-Zero.png', format='png', dpi=700)
#plt.show()
