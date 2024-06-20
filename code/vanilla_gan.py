#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from data_loader import get_emoji_loader
from models import CycleGenerator, DCDiscriminator
from vanilla_utils import create_dir, create_model, checkpoint, sample_noise, save_samples

# draw loss gragh
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(train_loader, opt, device):
    
    # Create empty lists to store the loss values
    D_real_loss_values = []
    D_fake_loss_values = []
    G_loss_values = []

    G, D = create_model(opts)
    
    G.to(device)
    D.to(device)
    
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])
    
    fixed_noise = sample_noise(16, opts.noise_size).to(device)
    
    iteration = 1
    
    mse_loss = torch.nn.MSELoss()
#     bce_loss = torch.nn.BCELoss()
    total_train_iters = opts.num_epochs * len(train_loader)
    
    for epoch in range(opts.num_epochs):

        for batch in train_loader:

            real_images = batch[0].to(device)


            ## TRAIN THE DISCRIMINATOR
            d_optimizer.zero_grad()

            # 1. Compute the discriminator loss on real images
            D_real_loss = mse_loss(D(real_images), torch.ones(len(real_images)).to(device))

            # 2. Sample noise
            noise = sample_noise(len(real_images), opts.noise_size).to(device)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            D_fake_loss = mse_loss(D(fake_images), torch.zeros(len(real_images)).to(device))

            # 5. Compute the total discriminator loss
            D_total_loss = 0.5 * D_real_loss + 0.5 * D_fake_loss
    
            D_total_loss.backward()
            d_optimizer.step()


            ## TRAIN THE GENERATOR
            g_optimizer.zero_grad()

            # 1. Sample noise
            noise = sample_noise(len(real_images), opts.noise_size).to(device)

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            G_loss = mse_loss(D(fake_images), torch.ones(len(real_images)).to(device)) 

            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            # Append the loss values to the respective lists
            D_real_loss_values.append(D_real_loss.item())
            D_fake_loss_values.append(D_fake_loss.item())
            G_loss_values.append(G_loss.item())

            iteration += 1
    
    # Plot the loss graph
    epochs = range(0, len(D_real_loss_values) * opts.log_step, opts.log_step)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(epochs, D_real_loss_values, 'r', label='D_real_loss')
    plt.plot(epochs, D_fake_loss_values, 'b', label='D_fake_loss')
    plt.plot(epochs, G_loss_values, 'y', label='G_loss')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("Vanilla-GAN.png")

    
    
    
def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    train_loader, _ = get_emoji_loader(opts.emoji, opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)
    
    if torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    train(train_loader, opts, device)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=30) # change here
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./samples_vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)

