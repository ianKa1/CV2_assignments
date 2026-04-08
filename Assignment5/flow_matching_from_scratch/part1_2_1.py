import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.GELU(x)
        return x

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.GELU(x)
        return x
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.GELU(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.flatten = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
    def forward(self, x):
        return self.flatten(x)
    
class Unflatten(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unflatten, self).__init__()
        self.unflatten = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, stride=7, padding=0)
        self.BN = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
    def forward(self, x):
        x = self.unflatten(x)
        x = self.BN(x)
        x = self.GELU(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.downconv = DownConv(in_channels, out_channels)
        self.convblock = ConvBlock(out_channels, out_channels)
    def forward(self, x):
        x = self.downconv(x)
        x = self.convblock(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = UpConv(in_channels, out_channels)
        self.convblock = ConvBlock(out_channels, out_channels)
    def forward(self, x):
        x = self.upconv(x)
        x = self.convblock(x)
        return x

class UnconditionalUNet(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(UnconditionalUNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, num_hiddens)
        self.down1 = DownBlock(num_hiddens, num_hiddens)
        self.down2 = DownBlock(num_hiddens, num_hiddens*2)
        self.down3 = Flatten()
        self.up1 = Unflatten(num_hiddens*2, num_hiddens*2)
        self.up2 = UpBlock(num_hiddens*4, num_hiddens)
        self.up3 = UpBlock(num_hiddens*2, num_hiddens)
        self.conv2 = ConvBlock(num_hiddens*2, num_hiddens)
        self.output = nn.Conv2d(num_hiddens, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        dn1 = self.down1(x)
        dn2 = self.down2(dn1)
        latent = self.down3(dn2)
        up1 = self.up1(latent)
        up2 = self.up2(torch.cat([up1, dn2], dim=1))
        up3 = self.up3(torch.cat([up2, dn1], dim=1))
        x = self.conv2(torch.cat([up3, x], dim=1))
        x = self.output(x)
        return x
    
def add_noise(image, sigma):
    noise = torch.randn_like(image) * sigma
    return image + noise
    
def visualize_noise_sample(image, save_path):
    sigma = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    noisy_images = [add_noise(image, s) for s in sigma]

    fig, axes = plt.subplots(1, len(sigma), figsize=(14, 2))
    for i, (noisy_img, s) in enumerate(zip(noisy_images, sigma)):
        img = noisy_img.cpu().detach().squeeze().numpy()  # squeeze to remove channel dim for grayscale
        img = np.clip(img, 0, 1)
        axes[i].imshow(img, cmap='gray')  # use grayscale colormap
        axes[i].set_title(f'σ={s}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return noisy_images

def train_denosing_model(device):
    model = UnconditionalUNet(in_channels=1, num_hiddens=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # DataLoader for FashionMNIST
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    # Validation set
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Lists to store losses (recorded every 20 steps)
    train_losses = []
    loss_steps = []

    # Get 5 fixed test images for evaluation
    test_images_sample = []
    for i in range(5):
        img, _ = testset[i]
        test_images_sample.append(img)
    test_images_sample = torch.stack(test_images_sample)

    # Train for 5 epochs
    num_epochs = 5
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        pbar = tqdm(trainloader, desc=f"Epoch {epoch + 1}", unit="batch")

        for batch_idx, (images, labels) in enumerate(pbar):
            model.train()
            images = images.to(device)
            noisy_images = add_noise(images, sigma=0.5)
            outputs = model(noisy_images)
            loss = nn.MSELoss()(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Record training loss every 20 steps
            if global_step % 20 == 0:
                train_losses.append(loss.item())
                loss_steps.append(global_step)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Evaluate at the end of each epoch
        model.eval()
        with torch.no_grad():
            eval_imgs = test_images_sample.to(device)
            eval_noisy = add_noise(eval_imgs, sigma=0.5)
            eval_denoised = model(eval_noisy)

            # Save visualization
            fig, axes = plt.subplots(3, 5, figsize=(12, 7))
            for i in range(5):
                # Original
                axes[0, i].imshow(eval_imgs[i].cpu().squeeze().numpy(), cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                # Noisy
                axes[1, i].imshow(eval_noisy[i].cpu().squeeze().numpy().clip(0, 1), cmap='gray')
                axes[1, i].set_title('Noisy')
                axes[1, i].axis('off')
                # Denoised
                axes[2, i].imshow(eval_denoised[i].cpu().squeeze().numpy().clip(0, 1), cmap='gray')
                axes[2, i].set_title('Denoised')
                axes[2, i].axis('off')

            plt.suptitle(f'Evaluation at Epoch {epoch + 1}')
            plt.tight_layout()
            plt.savefig(f'eval_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved evaluation: eval_epoch_{epoch + 1}.png")

        # Save model checkpoint after each epoch
        checkpoint_path = f'denoising_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save the final trained model
    torch.save(model.state_dict(), 'denoising_model_final.pth')
    print("\nFinal model saved to 'denoising_model_final.pth'")

    # Write losses to log file
    with open('training_log.txt', 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Total Steps: {global_step}\n")
        f.write("\nTraining Losses (recorded every 20 steps):\n")
        for step, loss in zip(loss_steps, train_losses):
            f.write(f"Step {step}: {loss:.6f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"\nAverage Loss: {sum(train_losses) / len(train_losses):.6f}\n")
        f.write(f"Final Loss: {train_losses[-1]:.6f}\n")
    print("\nTraining log saved to 'training_log.txt'")

    return model

def test_denoising_model(model, device):
    # Load test dataset and sample 5 images
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    test_images = []
    for i in range(5):
        img, _ = testset[i]
        test_images.append(img)
    test_images = torch.stack(test_images).to(device)

    # Different sigma values to test
    sigma_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    model.eval()
    with torch.no_grad():
        # Create visualization for each sigma value
        for sigma in sigma_values:
            noisy_images = add_noise(test_images, sigma)
            denoised_images = model(noisy_images)

            # Create a grid: 3 rows (original, noisy, denoised) x 5 columns (5 images)
            fig, axes = plt.subplots(3, 5, figsize=(12, 7))

            for i in range(5):
                # Original
                orig_img = test_images[i].cpu().squeeze().numpy()
                axes[0, i].imshow(orig_img, cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')

                # Noisy
                noisy_img = noisy_images[i].cpu().squeeze().numpy().clip(0, 1)
                axes[1, i].imshow(noisy_img, cmap='gray')
                axes[1, i].set_title(f'Noisy (σ={sigma})')
                axes[1, i].axis('off')

                # Denoised
                denoised_img = denoised_images[i].cpu().squeeze().numpy().clip(0, 1)
                axes[2, i].imshow(denoised_img, cmap='gray')
                axes[2, i].set_title('Denoised')
                axes[2, i].axis('off')

            plt.suptitle(f'Denoising Results with σ={sigma}')
            plt.tight_layout()
            plt.savefig(f'test_results_sigma_{sigma}.png', dpi=150, bbox_inches='tight')
            print(f"Saved test results: test_results_sigma_{sigma}.png")
            plt.show()

if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train or test denoising model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    if args.train:
        # Train denoising model
        model = train_denosing_model(device=device)
    else:
        # Load model from disk
        print("Loading model from 'denoising_model_final.pth'...")
        model = UnconditionalUNet(in_channels=1, num_hiddens=128).to(device)
        model.load_state_dict(torch.load('denoising_model_final.pth'))
        model.eval()
        print("Model loaded successfully!")

        test_denoising_model(model, device)

