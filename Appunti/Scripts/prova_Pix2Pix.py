import os, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

num_epoche = 100

# --------------------- Dataset ---------------------
class PairedHistologyDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.pairs = self._get_pairs()

    def _get_pairs(self):
        files = os.listdir(self.folder_path)
        prefixes = [f.replace('_label_free.tif', '')
                    for f in files if f.endswith('_label_free.tif')
                    and f.replace('_label_free.tif', '') + '_stained.tif' in files]
        return sorted(prefixes)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prefix = self.pairs[idx]
        lf = Image.open(os.path.join(self.folder_path, prefix + '_label_free.tif')).convert('RGB')
        st = Image.open(os.path.join(self.folder_path, prefix + '_stained.tif')).convert('RGB')
        if self.transform:
            lf = self.transform(lf)
            st = self.transform(st)
        return lf, st

# --------------------- Generator (UNet-like) ---------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --------------------- Discriminator (PatchGAN) ---------------------
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# --------------------- Determinism ---------------------
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------- Training ---------------------
def main():
    set_seed(42) # Imposta il seed per la riproducibilità

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((512, 512)), # potrebbe essere 256 o 512
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PairedHistologyDataset("Materiale/Locale/dataset_split/train", transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True) # prima num_workers era a 0 e pin_memory non c'era e batch_size a 4

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    os.makedirs("Materiale/Locale/output_pix2pix", exist_ok=True)
    for file in os.listdir("Materiale/Locale/output_pix2pix"):
        os.remove(os.path.join("Materiale/Locale/output_pix2pix", file))

    for epoch in range(num_epoche):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # Train discriminator
            fake = G(x).detach()
            D_real = D(x, y)
            D_fake = D(x, fake)
            
            real_label = torch.ones_like(D_real, device=device)
            fake_label = torch.zeros_like(D_fake, device=device)

            loss_D = bce(D_real, real_label) + bce(D_fake, fake_label)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Train generator
            fake = G(x)
            D_fake = D(x, fake)
            loss_G = bce(D_fake, real_label) + l1(fake, y) * 100
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            if i % loader.batch_size == 0:
                save_image((x[0] * 0.5 + 0.5), f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_input.png")
                save_image((fake[0].detach() * 0.5 + 0.5), f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_output.png")
                save_image((y[0] * 0.5 + 0.5), f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_target.png")
                print(f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} loss_D: {loss_D.item():.4f}")

if __name__ == "__main__":
    main()