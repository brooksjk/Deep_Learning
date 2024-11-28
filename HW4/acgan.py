import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import matplotlib.pyplot as plt


BASE_DIR = '/scratch/jkbrook/Deep_Learning/HW4/'

# Create necessary directories
os.makedirs("acgan", exist_ok=True)

# Set parameters
num_epochs = 500
lr_discrim = 0.0002
beta1_discrim = 0.5
lr_gen = 0.0002
beta1_gen = 0.5
batch_size = 8
z_dim = 100
num_classes = 10
img_size = 32
num_channels = 3

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
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.6)
        self.disc_output = nn.Linear(256 * 2 * 2, 1)  # For real/fake
        self.aux_output = nn.Sequential(
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # For class labels
        )
        
    def forward(self, x):
        features = self.conv_blocks(x)
        features = self.flatten(features)
        features = self.dropout(features)
        disc_logits = self.disc_output(features)
        aux_logits = self.aux_output(features)
        return disc_logits, aux_logits

# Define the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim + num_classes, 512 * 4 * 4)
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, num_channels, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        return self.deconv_blocks(x)

def generate_and_save_images(generator, epoch, z_dim, num_classes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    if epoch % 50 == 0:
        # Generate noise and labels explicitly for each class
        noise = torch.randn(num_classes, z_dim, device='cuda')
        labels = torch.arange(num_classes, device='cuda')  # One label for each class
        labels_one_hot = one_hot(labels, num_classes)
        
        # Generate fake images
        generator.eval()
        with torch.no_grad():
            fake_images = generator(noise, labels_one_hot)
        generator.train()
        
        # Rescale images to [0, 1] for saving
        fake_images = (fake_images + 1) / 2.0  # Assuming generator outputs in range [-1, 1]
        
        # Plot and save the images with labels
        fig, axes = plt.subplots(1, num_classes, figsize=(20, 4))
        for i, ax in enumerate(axes):
            img = fake_images[i].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Class {i}", fontsize=10)  # Add class label as title
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch}_labeled.png")
        plt.close()


# Initialize models
discriminator = Discriminator().to(device)
generator = Generator().to(device)

# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Optimizers
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_discrim, betas=(beta1_discrim, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=lr_gen, betas=(beta1_gen, 0.999))

# One-hot encoding for labels
def one_hot(labels, num_classes):
    return torch.zeros(labels.size(0), num_classes, device=device).scatter_(1, labels.unsqueeze(1), 1)

loss_tracker = []

# Training loop
for epoch in range(num_epochs):
    d_loss_list = []
    g_loss_list = []

    for batch_idx, (real_images, real_labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_images.size(0)

        # One-hot encode the real labels for training the generator
        real_labels_one_hot = one_hot(real_labels, num_classes)

        # Train Discriminator
        z_noise = torch.randn(batch_size, z_dim, device=device)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_labels_one_hot = one_hot(fake_labels, num_classes)
        
        fake_images = generator(z_noise, fake_labels_one_hot)
        
        # Real and fake labels for discriminator
        real_target = torch.ones(batch_size, 1, device=device)
        fake_target = torch.zeros(batch_size, 1, device=device)

        # Discriminator loss
        real_logits, real_aux_logits = discriminator(real_images)
        fake_logits, fake_aux_logits = discriminator(fake_images.detach())

        d_real_loss = adversarial_loss(real_logits, real_target) + auxiliary_loss(real_aux_logits, real_labels)
        d_fake_loss = adversarial_loss(fake_logits, fake_target) + auxiliary_loss(fake_aux_logits, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss_list.append(d_loss.item())

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        z_noise = torch.randn(batch_size, z_dim, device=device)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        fake_labels_one_hot = one_hot(fake_labels, num_classes)
        fake_images = generator(z_noise, fake_labels_one_hot)
        
        fake_logits, fake_aux_logits = discriminator(fake_images)

        g_adv_loss = adversarial_loss(fake_logits, real_target)  # Want fake to be classified as real
        g_aux_loss = auxiliary_loss(fake_aux_logits, fake_labels)  # Correct class labels
        g_loss = g_adv_loss + g_aux_loss
        g_loss_list.append(g_loss.item())

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    generate_and_save_images(generator, epoch + 1, z_dim, num_classes, save_dir=os.path.join(BASE_DIR, 'acgan', "images"))

    # Calculate mean losses and append to loss_tracker
    loss_tracker.append((np.mean(d_loss_list), np.mean(g_loss_list)))
    np.save(os.path.join(BASE_DIR, 'acgan', "loss_tracker.npy"), loss_tracker)

    # Save checkpoint
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), os.path.join(BASE_DIR, 'acgan', f"generator_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(BASE_DIR, 'acgan', f"discriminator_epoch_{epoch+1}.pth"))

