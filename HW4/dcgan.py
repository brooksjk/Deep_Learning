import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt

BASE_DIR = '/scratch/jkbrook/Deep_Learning/HW4/'

# Create necessary directories
os.makedirs("dcgan", exist_ok=True)

# Set parameters
num_epochs = 500
lr_discrim = 0.0002
beta1_discrim = 0.5
lr_gen = 0.0002
beta1_gen = 0.5
batch_size = 100
z_dim = 100
img_size = 32
num_channels = 3
num_classes = 10

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data processing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 2 * 2, 1),
        )

    def forward(self, x):
        return self.model(x)

# Define the generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * 512),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)

# Function to generate and save images
def generate_and_save_images(generator, epoch, z_dim, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if epoch % 50 == 0:
        # Generate noise
        noise = torch.randn(10, z_dim, device=device)

        # Generate fake images
        generator.eval()
        with torch.no_grad():
            fake_images = generator(noise)
        generator.train()

        # Rescale images to [0, 1] for saving
        fake_images = (fake_images + 1) / 2.0  # Assuming generator outputs in range [-1, 1]

        # Plot and save the images
        fig, axes = plt.subplots(1, 10, figsize=(20, 4))
        for i, ax in enumerate(axes):
            img = fake_images[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch}_generated.png")
        plt.close()

def one_hot(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes, device=device).scatter_(1, labels.unsqueeze(1), 1)

# Initialize models
discriminator = Discriminator().to(device)
generator = Generator(z_dim).to(device)

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_discrim, betas=(beta1_discrim, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr_gen, betas=(beta1_gen, 0.999))

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Track losses
loss_tracker = []

# Training loop
for epoch in range(num_epochs):
    d_losses, g_losses = [], []

    for real_images, _ in train_loader:
        real_images = real_images.to(device)

        # Train Discriminator
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Real images
        d_real_loss = criterion(discriminator(real_images), real_labels)

        # Fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        d_fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_loss = criterion(discriminator(fake_images), real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Record losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    # Log and save progress
    avg_d_loss = np.mean(d_losses)
    avg_g_loss = np.mean(g_losses)
    loss_tracker.append([avg_d_loss, avg_g_loss])

    print(f"Epoch [{epoch + 1}/{num_epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    generate_and_save_images(generator, epoch + 1, z_dim, save_dir=os.path.join(BASE_DIR, 'dcgan', "images"))

    # Save models every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), f"dcgan/generator_epoch_{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"dcgan/discriminator_epoch_{epoch + 1}.pth")

# Save loss tracker
np.save("dcgan/loss_tracker.npy", loss_tracker)
