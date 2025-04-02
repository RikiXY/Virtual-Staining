import os, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
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
class DoubleConv(nn.Module):
    """
    A building block: (Conv -> BatchNorm -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling using transposed convolution, then DoubleConv.
    Skip connection is concatenated with the upsampled feature map.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the current upsampled feature map
        # x2 is the skip connection from the encoder
        x1 = self.up(x1)
        
        # Input is [N, C, H, W].
        # We need to handle differences if x1 and x2 have different shapes
        # (they shouldn't if you started with 512x512, but just in case).
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution for output
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        """
        Args:
            n_channels (int): Number of input channels. For RGB, use 3. 
                              For single-channel (grayscale), use 1, etc.
            n_classes (int): Number of output channels/classes. 
                             For RGB output, use 3.
            bilinear (bool): Whether to use bilinear upsampling or transposed conv.
        """
        super(UNetGenerator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # You can customize the number of filters in each stage.

        # Version from 32 to 512
        # self.inc = DoubleConv(n_channels, 32)         # initial conv
        # self.down1 = Down(32, 64) 
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256)
        # self.down4 = Down(256, 512)

        # Version from 64 to 1024
        self.inc = DoubleConv(n_channels, 64)         # initial conv
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)                  # bottleneck

        
        # if using bilinear, use factor=2 in the below up layers

        # Version from 512 to 32
        # self.up1 = Up(512, 256, bilinear)
        # self.up2 = Up(256, 128, bilinear)
        # self.up3 = Up(128, 64, bilinear)
        # self.up4 = Up(64, 32, bilinear)

        # Version from 1024 to 64
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Version from 32 to 3
        # self.outc = OutConv(32, n_classes)

        # Version from 64 to 3
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # shape: [N, 64, 512, 512]
        x2 = self.down1(x1)    # shape: [N, 128, 256, 256]
        x3 = self.down2(x2)    # shape: [N, 256, 128, 128]
        x4 = self.down3(x3)    # shape: [N, 512, 64, 64]
        x5 = self.down4(x4)    # shape: [N, 1024, 32, 32]

        # Decoder
        x = self.up1(x5, x4)   # shape: [N, 512, 64, 64]
        x = self.up2(x, x3)    # shape: [N, 256, 128, 128]
        x = self.up3(x, x2)    # shape: [N, 128, 256, 256]
        x = self.up4(x, x1)    # shape: [N, 64, 512, 512]
        
        logits = self.outc(x)  # shape: [N, n_classes, 512, 512]
        return logits

# --------------------- Discriminator (PatchGAN) ---------------------
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image tasks (e.g., pix2pix, CycleGAN).
    Produces an N×N map of 'real/fake' predictions for each patch of the input.
    """
    def __init__(self, in_channels=6, ndf=64, use_sigmoid=False):
        """
        Args:
            in_channels (int): Number of channels in the input. In pix2pix, 
                               if we concatenate input + output images, 
                               this might be 6 (e.g. 3+3).
            ndf (int): Base number of discriminator filters. 
                       The number of filters doubles with each layer.
            use_sigmoid (bool): Whether to apply a sigmoid at the end 
                                (common in older versions of pix2pix). 
                                In newer WGAN-GP, we typically don't.
        """
        super(PatchGANDiscriminator, self).__init__()

        # The PatchGAN discriminator is basically several downsampling layers
        # leading to a final 1×1 convolution. Each location in the final
        # feature map corresponds to a "patch" in the input image.

        # Convolution blocks
        # 1) No normalization in the first layer
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # 2) Downsampling layers
        curr_dim = ndf
        next_dim = curr_dim * 2
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        curr_dim = next_dim
        next_dim = curr_dim * 2
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # 3) Stride = 1 for the last convolution(s) so that 
        #    the receptive field is about 70×70 (PatchGAN).
        curr_dim = next_dim
        next_dim = curr_dim * 2
        # This layer keeps stride=1 to reduce the patch size growth
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # 4) Output layer: 1 filter, stride=1
        layers += [
            nn.Conv2d(next_dim, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        # Optionally apply a sigmoid (e.g., for vanilla GAN or BCE loss)
        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Inputs:
            x (Tensor): shape (batch_size, in_channels, H, W)
        Returns:
            Tensor of shape (batch_size, 1, H/2^n, W/2^n) 
            (or similarly downsampled dimensions), where each location 
            is a prediction of real/fake for that patch.
        """
        return self.model(torch.cat([x, y], dim=1))  # Concatenate input and output


# --------------------- Determinism ---------------------
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # prima era True
    torch.backends.cudnn.benchmark = False # prima era False

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
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=12, pin_memory=True) # prima num_workers era a 0 e pin_memory non c'era e batch_size a 4, Shuffle era a Flase
    # andrebbero presi i tempi precisi per ogni configurazione (profiling(?)), ma in linea di massima:
    # con num_workers=12 impiega circa 1.27 minuti, 1.28 e 1.22
    # con num_workers=8 impiega circa 1.26 minuti, 1.26 e 1.34
    # con num_workers=4 impega circa 1.41 minuti, 1.41 e 1.38
    # con num_workers=0 impiega circa 1.27 minuti, 1.23 e 1.29

    G = UNetGenerator(bilinear=False).to(device)
    D = PatchGANDiscriminator().to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    scaler = GradScaler(device='cuda')

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    os.makedirs("Materiale/Locale/output_pix2pix", exist_ok=True)
    for file in os.listdir("Materiale/Locale/output_pix2pix"):
        os.remove(os.path.join("Materiale/Locale/output_pix2pix", file))

    for epoch in range(num_epoche):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # ---------- DISCRIMINATOR ----------
            with autocast("cuda"):
                fake = G(x).detach()
                D_real = D(x, y)
                D_fake = D(x, fake)

                real_label = torch.ones_like(D_real, device=device)
                fake_label = torch.zeros_like(D_fake, device=device)

                loss_D = bce(D_real, real_label) + bce(D_fake, fake_label)

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # ---------- GENERATOR ----------
            with autocast("cuda"):
                fake = G(x)
                D_fake = D(x, fake)
                loss_G = bce(D_fake, real_label) + l1(fake, y) * 100

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ---------- UPDATE SCALER ----------
            scaler.update()

            if i % loader.batch_size == 0:
                save_image((x[0] * 0.5 + 0.5), f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_input.png")
                save_image((fake[0].detach() * 0.5 + 0.5), f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_output.png")
                save_image((y[0] * 0.5 + 0.5), f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_target.png")
                print(f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} loss_D: {loss_D.item():.4f}")

if __name__ == "__main__":
    main()