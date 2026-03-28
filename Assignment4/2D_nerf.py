import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
# x -> [batch_size]
def sinusoidal_position_encoding(x, L, device=None):
    batch_size = x.shape[0]
    arr = torch.arange(L, device=device) # [L]
    arr = 2**arr * torch.pi * x[:, None] # [batch, L]
    sins = torch.sin(arr) 
    coss = torch.cos(arr)
    pe = torch.cat([sins[:, :, None], coss[:, :, None]], dim=-1)
    pe = pe.view(batch_size, -1)
    pe = torch.cat([x[:, None], pe], dim=-1)

    return pe

def make_infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch

class MLP(torch.nn.Module):
    # one pixel (channel + PE(4L+2)))
    def __init__(self, channels=3, L=10, hidden_size=256):
        super().__init__()
        self.L = L
        self.hidden_size = hidden_size
        self.hidden_layers = torch.nn.Sequential(
            torch.nn.Linear(4*L+2, hidden_size),
            torch.nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, coords):
        pe_x = sinusoidal_position_encoding(coords[:, 0], self.L, device=coords.device)
        pe_y = sinusoidal_position_encoding(coords[:, 1], self.L, device=coords.device)
        inputt = torch.cat([pe_x, pe_y], dim=-1)
        hidden = self.hidden_layers(inputt)
        y = self.sigmoid(self.out(hidden))

        return y

class PixelSampleDataset(Dataset):
    def __init__(self, image_path, length=10000):
        self.length = length

        # torch tensor [C, H, W]
        self.image = T.ToTensor()(Image.open(image_path).convert("RGB"))
        self.C, self.H, self.W = self.image.shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randint(0, self.W, ())
        y = torch.randint(0, self.H, ())

        pixel = self.image[ :, y, x]

        coord = torch.Tensor([x, y])
        coord_norm = coord.clone()
        coord_norm[0] /= (self.W - 1)
        coord_norm[1] /= (self.H - 1)

        return {
            "coord": coord,
            "coord_norm": coord_norm,
            "pixel": pixel
        }


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    image_path = "images/fox.jpg"
    image_name = image_path.split("/")[-1].split(".")[0]
    L = 10
    hidden_size = 256
    dataset = PixelSampleDataset(
        image_path=image_path,
        length=10000
    )

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )

    model = MLP(L=L, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Set the number of training steps
    max_steps = 10000

    # Create subdirectory for this training run
    run_name = f"{image_name}_L{L}_width{hidden_size}"
    output_dir = Path("2d_output") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_path = output_dir / "training_log.txt"
    log_file = open(log_path, 'w')
    log_file.write(f"Training {image_name} with L={L}, hidden_size={hidden_size}\n")
    log_file.write(f"Total steps: {max_steps}\n\n")

    model.train()
    train_iter = make_infinite(dataloader)
    for step in range(max_steps):
        batch = next(train_iter)
        coord = batch["coord"].to(device)             # [B, 2]
        coord_norm = batch["coord_norm"].to(device)   # [B, 2]
        pixel = batch["pixel"].to(device)             # [B, 3]

        pred = model(coord_norm)
        mse_loss = F.mse_loss(pred, pixel)
        PSNR = 10 * torch.log10(1.0 / (mse_loss + 1e-6))
        loss = -PSNR

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            log_msg = f"Step {step}/{max_steps}, PSNR: {PSNR.item():.2f}, MSELoss: {mse_loss.item():.4f}"
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()  # Ensure immediate write to disk
        
        if step % (max_steps // 10) == 0:
            output_H = dataset.H
            output_W = dataset.W
            output_channels = dataset.C
            coords_x = torch.linspace(0, 1, output_W)
            coords_y = torch.linspace(0, 1, output_H)
            coords_x = coords_x[:, None].expand(output_W, output_H)
            coords_y = coords_y[None, :].expand(output_W, output_H)
            coords = torch.cat([coords_x[ :, :, None], coords_y[:, :, None]], dim=-1)
            coords = coords.view(-1, 2).to(device)

            with torch.no_grad():
                pred_pixels = model(coords)

            # Convert to numpy and scale to [0, 255]
            pred_pixels = pred_pixels.detach().cpu().view(output_W, output_H, output_channels).transpose(0, 1)
            pred_pixels_np = pred_pixels.numpy()
            pred_pixels_np = (pred_pixels_np * 255).astype('uint8')

            # Create and save image
            output_image = Image.fromarray(pred_pixels_np)
            output_image.save(output_dir / f"step{step}.png")
            save_msg = f"Saved image at step {step}"
            print(save_msg)
            log_file.write(save_msg + "\n")
            log_file.flush()

    # Close log file
    log_file.write(f"\nTraining completed after {max_steps} steps\n")
    log_file.close()
    print(f"Training log saved to {log_path}")

