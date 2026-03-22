import torch
import torch.nn as nn

from src.dataset_3d import load_data, RaysData
from src.rendering import sample_along_rays

from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

def make_infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch

# x -> [batch, d]
def sinusoidal_position_encoding(x, L, device=None):
    batch_size, d = x.shape
    arr = torch.arange(L, device=device) # [L]
    arr = 2**arr * torch.pi * x[:, :, None] # [batch, d, L]
    sins = torch.sin(arr) 
    coss = torch.cos(arr)
    pe = torch.stack([sins[:, :, :], coss[:, :, :]], dim=-1)
    pe = pe.view(batch_size, d, -1)
    pe = torch.cat([x[:, :, None], pe], dim=-1)

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
            nn.Linear(hidden_size + 6*L+3, hidden_size // 2)
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
            nn.Sigmoid()
        )

    # x -> [batch, 3] rd -> [batch, 3]
    def forward(self, x, rd):
        batch_size = x.shape[0]
        pe_x = sinusoidal_position_encoding(x, self.L).view(batch_size) # [batch_size, 6L+3]
        pe_rd = sinusoidal_position_encoding(rd, self.L).view(batch_size)

        hidden_half_net = self.half_net0(pe_x)
        input_half_net = torch.cat([hidden_half_net, pe_x], dim=-1)
        hidden = self.half_net1(input_half_net)

        density = self.density_output(hidden)

        hidden_rgb = self.rgb_output_pre(hidden)
        input_rgb_block = torch.cat([hidden_rgb, pe_rd], dim=-1)
        rgb = self.rgb_output(input_rgb_block)

        return density, rgb


if __name__ == '__main__':

    # images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(data_path=cfg.data_path)

    # H, W = images_train.shape[1], images_train.shape[2]
    # dataset = RaysData(images_train.to(cfg.device), K.to(cfg.device), c2ws_train.to(cfg.device), device=cfg.device)

    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_worker=0)

    # train_iter = make_infinite(dataloader)
    
    # max_steps = 10000
    
    # for step in range(max_steps):
    #     batch = next(train_iter)