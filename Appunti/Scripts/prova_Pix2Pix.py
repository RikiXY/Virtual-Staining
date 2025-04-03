import os, random
import numpy as np
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# =========================
# PARAMETRI IMPORTANTI
# ----------------------
num_epoche = 1 # Numero di epoche da eseguire
useCheckPoint = False # Se vuoi riprendere da un checkpoint esistente, metti True
seed = 42 # Seed per la riproducibilità
# -----------------------
BatchSize = 8 # Batch size per il DataLoader (8 è un buon valore, ma dipende dalla GPU)
Shuffle = True # Se vuoi mescolare i dati ad ogni epoca, metti True
NumWorkers = 12 # Numero di worker per il DataLoader (12 è un buon valore, ma dipende dalla GPU)
ImgSize = (512, 512) # Risoluzione delle immagini (512x512 è un buon valore per Pix2Pix), si può pensare anche a 256x256
# =========================

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

# --------------------- Generatore (UNet-like) ---------------------
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

        # Encoders, versione da 64 a 1024
        self.inc = DoubleConv(n_channels, 64) # Convoluzione iniziale
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)          # Bottleneck

        # Decoders, versione da 1024 a 64
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Convoluzione finale, versione da 64 a N (Per RGB è 3)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Per immagini RGB, N vale 3
        # Per immagini grayscale, N vale 1

        # Encoder
        x1 = self.inc(x)       # Shape: [N, 64, 512, 512]
        x2 = self.down1(x1)    # Shape: [N, 128, 256, 256]
        x3 = self.down2(x2)    # Shape: [N, 256, 128, 128]
        x4 = self.down3(x3)    # Shape: [N, 512, 64, 64]
        x5 = self.down4(x4)    # Shape: [N, 1024, 32, 32]

        # Decoder
        x = self.up1(x5, x4)   # Shape: [N, 512, 64, 64]
        x = self.up2(x, x3)    # Shape: [N, 256, 128, 128]
        x = self.up3(x, x2)    # Shape: [N, 128, 256, 256]
        x = self.up4(x, x1)    # Shape: [N, 64, 512, 512]
        
        logits = self.outc(x)  # Shape: [N, n_classes, 512, 512]
        return logits

# --------------------- Discriminatore (PatchGAN) ---------------------
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image-to-image tasks (e.g., pix2pix, CycleGAN).
    Produces an NxN map of 'real/fake' predictions for each patch of the input.
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
        return self.model(torch.cat([x, y], dim=1)) # Concatena input e output


# --------------------- Determinismo ---------------------
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Funzione di caricamento checkpoint
# =========================
def load_checkpoint(checkpoint_path, Generator, Discriminator, opt_G, opt_D, scaler_G, scaler_D, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    Generator.load_state_dict(checkpoint['generator_state_dict'])
    Discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_G.load_state_dict(checkpoint['optimizerG_state_dict'])
    opt_D.load_state_dict(checkpoint['optimizerD_state_dict'])
    scaler_G.load_state_dict(checkpoint['scalerG_state_dict'])
    scaler_D.load_state_dict(checkpoint['scalerD_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Riprendi dalla successiva
    print(f"Checkpoint caricato (epoch {checkpoint['epoch']}). Ripartenza da epoch {start_epoch}.")
    return start_epoch

# =========================
# Validazione del modello
# =========================
def validate(Generator, Discriminator, val_loader, device, bce_loss, l1_loss, epoch):
    # Metti in eval mode
    Generator.eval()
    Discriminator.eval()

    # Crea/assicurati che esista la cartella per gli output di validazione
    os.makedirs("Materiale/Locale/output_val", exist_ok=True)

    total_loss_G = 0.0
    total_loss_D = 0.0
    count = 0

    # Niente gradienti
    with torch.no_grad(), autocast("cuda"):
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)

            # Genera immagini di output
            fake = Generator(x)

            # Predizioni del discriminatore su reale e falso
            D_real = Discriminator(x, y)
            D_fake = Discriminator(x, fake)

            # BCE labels
            real_label = torch.ones_like(D_real, device=device)
            fake_label = torch.zeros_like(D_fake, device=device)

            # Calcolo delle due loss (D e G)
            loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)
            loss_G = bce_loss(D_fake, real_label) + 25 * l1_loss(fake, y)

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            count += 1

            # --- Salvataggio immagini di esempio ---
            # Ad esempio, salvi i primi 5 batch
            if i < 5:
                # Salva la prima immagine del batch (indice 0)
                # Rimettiamo le immagini da [-1,1] a [0,1]
                from torchvision.utils import save_image
                save_image((x[0] * 0.5 + 0.5), 
                           f"Materiale/Locale/output_val/epoch{epoch}_batch{i}_input.png")
                save_image((fake[0].detach() * 0.5 + 0.5), 
                           f"Materiale/Locale/output_val/epoch{epoch}_batch{i}_output.png")
                save_image((y[0] * 0.5 + 0.5), 
                           f"Materiale/Locale/output_val/epoch{epoch}_batch{i}_target.png")

    # Ritorni la media delle loss
    avg_loss_D = total_loss_D / count
    avg_loss_G = total_loss_G / count

    # Rimetti i modelli in training (se vuoi farlo qui)
    Generator.train()
    Discriminator.train()

    return avg_loss_G, avg_loss_D



# --------------------- Training ---------------------
def main():

    os.makedirs("Materiale/Locale/output_pix2pix_logs", exist_ok=True)

    # 1) Genero un nome file log con data/ora attuale (unico per ogni esecuzione).
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Materiale/Locale/output_pix2pix_logs/Log-{timestamp_str}.txt"

    # 2) Tempo iniziale e data/ora di avvio
    start_time = time.time()
    start_dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3) Creazione file di log in scrittura ("w"): cancella o crea ex novo
    with open(log_file, "w") as f:
        f.write(f"{start_dt_str}\nInizio training\n")

    set_seed(seed) # Imposta il seed per la riproducibilità

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize(ImgSize), # Risoluzione delle immagini, potrebbe essere 256 o 512
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PairedHistologyDataset("Materiale/Locale/dataset_split/train", transform)
    loader = DataLoader(dataset, batch_size=BatchSize, shuffle=Shuffle, num_workers=NumWorkers, pin_memory=True) # prima num_workers era a 0 e pin_memory non c'era e batch_size a 4, Shuffle era a Flase
    # andrebbero presi i tempi precisi per ogni configurazione (profiling(?)), ma in linea di massima:
    # con num_workers=12 impiega circa 1.27 minuti, 1.28 e 1.22
    # con num_workers=8 impiega circa 1.26 minuti, 1.26 e 1.34
    # con num_workers=4 impega circa 1.41 minuti, 1.41 e 1.38
    # con num_workers=0 impiega circa 1.27 minuti, 1.23 e 1.29

    val_dataset = PairedHistologyDataset("Materiale/Locale/dataset_split/val", transform)
    val_loader = DataLoader(val_dataset, batch_size=BatchSize, shuffle=Shuffle, num_workers=NumWorkers, pin_memory=True)


    # Inizializzazione del modello
    Generator = UNetGenerator().to(device)
    Discriminator = PatchGANDiscriminator().to(device)
    
    # Inizializzazione degli ottimizzatori
    opt_G = optim.Adam(Generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(Discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Inizializzazione del GradScaler per la precisione mista
    scaler_G = GradScaler(device='cuda')
    scaler_D = GradScaler(device='cuda')

    # Inizializzazione delle funzioni di perdita
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Creazione cartella per le immagini di preview e cancellazione dei file esistenti
    os.makedirs("Materiale/Locale/output_pix2pix", exist_ok=True)
    for file in os.listdir("Materiale/Locale/output_pix2pix"):
        os.remove(os.path.join("Materiale/Locale/output_pix2pix", file))

    if useCheckPoint == 1:
        os.makedirs("Materiale/Locale/checkpoints", exist_ok=True)

    # Se vuoi riprendere da un checkpoint esistente, metti la path qui
    checkpoint_path = "checkpoint_pix2pix_epoch.pth" # <-- Cambia questo con il tuo checkpoint
    if os.path.exists(checkpoint_path) and useCheckPoint:
        print("Caricamento checkpoint...")
        start_epoch = load_checkpoint(checkpoint_path, Generator, Discriminator, 
                                      opt_G, opt_D, scaler_G, scaler_D, device)
    else:
        print("Nessun checkpoint trovato o non richiesto.")
        # Se non esiste un checkpoint, inizia da 0
        start_epoch = 0

    print("Inizio allenamento...")
    for epoch in range(num_epoche):

        # Log inizio epoca
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{now_str}\nStart epoca {epoch}\n")
        
        Generator.train()
        Discriminator.train()

        for i, (x, y) in enumerate(loader):
            # (x, y) sono le immagini di input e target
            # x sono batch_size immagini di input (label-free)
            # y sono batch_size immagini di target (stained)

            # Trasferimento su GPU
            x, y = x.to(device), y.to(device)
            
            # ---------- DISCRIMINATORE ----------
            with autocast("cuda"):
                # Genera immagini false
                fake = Generator(x).detach()

                # Calcola le predizioni del discriminatore
                # D_real è la predizione per le immagini reali
                # D_fake è la predizione per le immagini false
                D_real = Discriminator(x, y)
                D_fake = Discriminator(x, fake)

                # Calcola la perdita del discriminatore
                # real_label e fake_label sono le etichette per BCEWithLogitsLoss
                # real_label = 1 (reale), fake_label = 0 (falso)
                real_label = torch.ones_like(D_real, device=device)
                fake_label = torch.zeros_like(D_fake, device=device)

                # Calcola la perdita combinando le predizioni reali e false
                loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)

            # Ottimizzazione del discriminatore
            opt_D.zero_grad() # zero_grad() azzera i gradienti accumulati
            scaler_D.scale(loss_D).backward() # scaler.scale() calcola i gradienti in modo scalato
            scaler_D.step(opt_D) # scaler.step() applica i gradienti all'ottimizzatore
            scaler_D.update() # Aggiorna lo scaler

            # ---------- GENERATORE ----------
            with autocast("cuda"):
                # Genera immagini false
                fake = Generator(x)

                # Calcola le predizioni del discriminatore per le immagini false
                D_fake = Discriminator(x, fake)

                # Calcola la perdita del generatore
                # La perdita del generatore è la somma della BCE con le predizioni del discriminatore
                # e della L1 loss tra le immagini false e quelle reali
                # La L1 loss penalizza le differenze tra le immagini generate e quelle reali
                # La BCE penalizza le predizioni del discriminatore
                loss_G = bce_loss(D_fake, real_label) + l1_loss(fake, y) * 25 # bisogna provare altri valori (prima era 100)

            # Ottimizzazione del generatore
            opt_G.zero_grad() # zero_grad() azzera i gradienti accumulati
            scaler_G.scale(loss_G).backward() # scaler.scale() calcola i gradienti in modo scalato
            scaler_G.step(opt_G) # scaler.step() applica i gradienti all'ottimizzatore
            scaler_G.update()

            # Stampa e log su file
            if i % 1 == 0:
                msg = f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} loss_D: {loss_D.item():.4f}"
                print(msg)
                # Append su log
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

                save_image((x[0] * 0.5 + 0.5), 
                           f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_input.png")
                save_image((fake[0].detach() * 0.5 + 0.5), 
                           f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_output.png")
                save_image((y[0] * 0.5 + 0.5), 
                           f"Materiale/Locale/output_pix2pix/{epoch:02d}_{i:03d}_target.png")

         # ---------- VALIDATION (a fine epoca) ----------
        val_loss_G, val_loss_D = validate(Generator, Discriminator, val_loader, device, bce_loss, l1_loss, epoch)

        # LOG
        val_msg = f"[Epoca {epoch}] Validation: loss_G={val_loss_G:.4f} loss_D={val_loss_D:.4f}"
        print(val_msg)
        with open(log_file, "a") as f:
            f.write(val_msg + "\n")

        # Log fine epoca
        with open(log_file, "a") as f:
            f.write(f"End epoca {epoch}\n")


        # ------ SALVATAGGIO CHECKPOINT ------
        # Salva ogni epoca (o magari ogni 5 epoche, se preferisci)
        if useCheckPoint == 1:
            if (epoch+1) % 1 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'generator_state_dict': Generator.state_dict(),
                    'discriminator_state_dict': Discriminator.state_dict(),
                    'optimizerG_state_dict': opt_G.state_dict(),
                    'optimizerD_state_dict': opt_D.state_dict(),
                    'scalerG_state_dict': scaler_G.state_dict(),
                    'scalerD_state_dict': scaler_D.state_dict()
                    # volendo potresti salvare anche l'ultimo loss_G, loss_D, ecc.
                }
                checkpoint_name = f"Materiale/Locale/checkpoints/checkpoint_Pix2Pix_epoca{epoch}_{timestamp_str}.pth"
                torch.save(checkpoint, checkpoint_name)
                print(f"Checkpoint salvato all'epoca {epoch}!")

    # Fine allenamento
    end_time = time.time()
    total_seconds = end_time - start_time
    end_dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_msg = f"Fine esecuzione in {end_dt_str}. Tempo impiegato = {total_seconds:.2f} secondi\n"
    print(final_msg)
    with open(log_file, "a") as f:
        f.write(final_msg)
              

if __name__ == "__main__":
    main()