import torch
import torch.nn as nn
import tyro
from PIL import Image
from pathlib import Path

from src.dataset_3d import load_data, RaysData, pixels_to_rays
from src.rendering import sample_along_rays, predict_rgbs

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

def make_infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch

# x -> [batch, d]
def sinusoidal_position_encoding(x, L):
    arr = torch.arange(L, device=x.device, dtype=x.dtype) # [L]
    arr = 2**arr * torch.pi * x[..., None] # [batch, d, L]
    sins = torch.sin(arr) 
    coss = torch.cos(arr)
    
    pe = torch.cat([sins, coss], dim=-1)
    pe = pe.reshape(*x.shape[:-1], -1)
    pe = torch.cat([x, pe], dim=-1)

    return pe

@dataclass
class Config:
    data_path: str = "data/lego_200x200.npz"
    near: float = 2.0
    far: float = 6.0
    num_samples_along_ray: int = 30
    num_rays: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "mps"
    port: int = 8080


class NerfModel(torch.nn.Module):
    def __init__(self, depth=7, L=4, hidden_size=256, device="cpu"):
        super().__init__()

        self.L = L

        half_depth0 = depth // 2
        half_depth1 = depth - half_depth0

        half_layers0 = [nn.Linear(6*L+3, hidden_size), nn.ReLU()]
        for _ in range(half_depth0 - 1):
            half_layers0 += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.half_net0 = nn.Sequential(*half_layers0)
        
        half_layers1 = [nn.Linear(hidden_size+6*L+3, hidden_size), nn.ReLU()]
        for _ in range(half_depth1 - 2):
            half_layers1 += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        half_layers1 += [nn.Linear(hidden_size, hidden_size)]
        self.half_net1 = nn.Sequential(*half_layers1)

        self.density_output = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

        self.rgb_output_pre = nn.Linear(hidden_size, hidden_size)
        self.rgb_output = nn.Sequential(
            nn.Linear(hidden_size + 6*L+3, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
            nn.Sigmoid()
        )

    # x -> [num_pixels, num_samples, 3] rd -> [num_pixels, 3]
    # density -> [num_pixels, num_samples] rgb->[num_pixels, num_samples, 3]
    def forward(self, x, rd):
        batch_size, num_samples = x.shape[:-1]
        pe_x = sinusoidal_position_encoding(x, self.L)
        pe_rd = sinusoidal_position_encoding(rd, self.L)

        hidden_half_net = self.half_net0(pe_x)
        input_half_net = torch.cat([hidden_half_net, pe_x], dim=-1)
        hidden = self.half_net1(input_half_net)

        density = self.density_output(hidden)

        hidden_rgb = self.rgb_output_pre(hidden)
        pe_rd = pe_rd[:, None, :].repeat(1, num_samples, 1)
        input_rgb_block = torch.cat([hidden_rgb, pe_rd], dim=-1)
        rgb = self.rgb_output(input_rgb_block)

        return rgb, density


if __name__ == '__main__':
    cfg = tyro.cli(Config)

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(data_path=cfg.data_path)

    H, W = images_train.shape[1], images_train.shape[2]
    dataset = RaysData(images_train.to(cfg.device), K.to(cfg.device), c2ws_train.to(cfg.device), device=cfg.device)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    nerf_model = NerfModel(device=cfg.device).to(cfg.device)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=5e-4)

    # Create output directory
    output_dir = Path("3d_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_path = output_dir / "training_log.txt"
    log_file = open(log_path, 'w')
    log_file.write(f"NeRF Training Log\n")
    log_file.write(f"Data path: {cfg.data_path}\n")
    log_file.write(f"Total steps: {10000}\n")
    log_file.write(f"Learning rate: {5e-4}\n")
    log_file.write(f"Batch size: {64}\n\n")

    train_iter = make_infinite(dataloader)
    max_steps = 10000

    nerf_model.train()
    for step in range(max_steps):
        batch = next(train_iter)
        r_os = batch["ray_o"].to(cfg.device)
        r_ds = batch["ray_d"].to(cfg.device)
        gt_rgbs = batch["rgb"].to(cfg.device)
        xs = sample_along_rays(r_os, r_ds, cfg.near, cfg.far, cfg.num_samples_along_ray, device=cfg.device).to(cfg.device)
        pred_rgb = predict_rgbs(nerf_model, xs, r_ds, cfg.near, cfg.far, cfg.num_samples_along_ray)
        
        mse_loss = nn.functional.mse_loss(pred_rgb, gt_rgbs) 
        loss = -10 * torch.log10(1.0 / (mse_loss + 1e-6))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            log_msg = f"steps: {step}/{max_steps}, MSE Loss: {mse_loss.item():.6f}, PSNR: {-loss.item():.2f}"
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()
        
        if step % 500 == 0:
            
            r_os = dataset.rays_o[0:H*W, :]
            r_ds = dataset.rays_d[0:H*W, :]
            gt_rgbs = dataset.gt_rgbs[0:H*W, :]
            
            xs = sample_along_rays(r_os, r_ds, cfg.near, cfg.far, cfg.num_samples_along_ray, device=cfg.device)
            with torch.no_grad():
                pred_rgb = predict_rgbs(nerf_model, xs, r_ds, cfg.near, cfg.far, cfg.num_samples_along_ray)
            pred_rgb = pred_rgb.view(H, W, 3).detach().cpu().numpy()
            
            pred_rgb = (pred_rgb * 255).astype('uint8')

            # Create and save image
            output_image = Image.fromarray(pred_rgb)
            output_image.save(output_dir / f"train_step{step}.png")
            save_msg = f"Saved training image at step {step}"
            print(save_msg)
            log_file.write(save_msg + "\n")
            log_file.flush()

    # Evaluation: Render novel views using c2ws_test
    eval_msg = "\n" + "="*50 + "\nStarting evaluation with test camera poses...\n" + "="*50
    print(eval_msg)
    log_file.write(eval_msg + "\n")
    log_file.flush()

    nerf_model.eval()
    test_dir = output_dir / "test_views"
    test_dir.mkdir(exist_ok=True)

    num_test_views = c2ws_test.shape[0]
    print(f"Rendering {num_test_views} test views...")

    for test_idx in range(num_test_views):
        # Generate rays for all pixels in this test view
        us = torch.arange(W, device=cfg.device).float()
        vs = torch.arange(H, device=cfg.device).float()
        v_grid, u_grid = torch.meshgrid(vs, us, indexing='ij')
        uv_grid = torch.stack([u_grid, v_grid], dim=-1)
        uvs = uv_grid.reshape(-1, 2)  # [H*W, 2]

        # Get camera-to-world matrix for this test view
        c2w_test = c2ws_test[test_idx].to(cfg.device)

        # Convert pixels to rays
        r_os, r_ds = pixels_to_rays(K=K.to(cfg.device), c2w=c2w_test, uvs=uvs, device=cfg.device)

        # Sample points along rays and render
        xs = sample_along_rays(r_os, r_ds, cfg.near, cfg.far, cfg.num_samples_along_ray, device=cfg.device)
        with torch.no_grad():
            pred_rgb = predict_rgbs(nerf_model, xs, r_ds, cfg.near, cfg.far, cfg.num_samples_along_ray)

        # Reshape to image and save
        pred_rgb = pred_rgb.view(H, W, 3).detach().cpu().numpy()
        pred_rgb = (pred_rgb * 255).clip(0, 255).astype('uint8')

        output_image = Image.fromarray(pred_rgb)
        output_image.save(test_dir / f"test_view_{test_idx:03d}.png")

        if test_idx % 10 == 0:
            progress_msg = f"Rendered test view {test_idx}/{num_test_views}"
            print(progress_msg)
            log_file.write(progress_msg + "\n")
            log_file.flush()

    final_msg = f"\nEvaluation complete! Test views saved to {test_dir}"
    print(final_msg)
    log_file.write(final_msg + "\n")

    # Close log file
    log_file.write(f"\nTraining and evaluation completed after {max_steps} steps\n")
    log_file.close()
    print(f"Training log saved to {log_path}")

