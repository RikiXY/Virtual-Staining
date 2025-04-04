import os, random, time, datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import sys

# =========================
# PARAMETRI IMPORTANTI
# ----------------------
n_epochs = 100 # Numero di epoche da eseguire
log_rate = 15 # Ogni quanto loggare (es. ogni 10 batch)
use_checkpoint = True # Se vuoi riprendere da un checkpoint esistente, metti True       Bisognerebbe creare create_checkpoint e load_checkpoint così si possono distinguere le casiistiche
checkpoint_rate = 10 # Ogni quanto salvare i checkpoint (es. ogni 10 epoche)
restore_checkpoint_path = "Materiale/Locale/checkpoints/checkpoint_Pix2Pix_epoca99_2025-04-04_00-55-08.pth" # Percorso del checkpoint (se esiste), ricordati di cambiare il nome
validate_rate = 10 # Ogni quanto validare (es. ogni 5 epoche), prendiamo per buono al momento come valore standard =checkpoint_rate
seed = 42 # Seed per la riproducibilità
# -----------------------
batch_size = 8 # Batch size per il DataLoader (8 è un buon valore, ma dipende dalla GPU)
training_shuffle = True # Se vuoi mescolare i dati ad ogni epoca del training, metti True
validation_shuffle = False # Se vuoi mescolare i dati ad ogni epoca del validation, metti True
n_workers = 12 # Numero di worker per il DataLoader (12 è un buon valore, ma dipende dalla GPU)
image_size = (512, 512) # Risoluzione delle immagini (512x512 è un buon valore per Pix2Pix), si può pensare anche a 256x256
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

# --------------------- Logging ---------------------
def log_message(message, log_file, show_time=True, use_stdout=True):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if show_time:
        message = f"[{now_str}] {message}"
    with open(log_file, "a+") as f:
        f.write(message + "\n")
    if use_stdout:
        print(message)

class ProgressTracker:
    def __init__(self, total_epochs, total_batches, max_history=50):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.max_history = max_history
        self.start_time = time.time()
        self.times = []
    
    def start(self):
        self.times = [time.time()]

    def calculate_progress(self, epoch, batch):
        self.times.append(time.time())
        if len(self.times) > self.max_history:
            self.times.pop(0)
        total_elapsed_time = self.times[-1] - self.start_time
        elapsed_time = self.times[-1] - self.times[0]
        eta = (elapsed_time / len(self.times)) * (self.total_epochs * self.total_batches - (epoch * self.total_batches + batch))
        expected_time = total_elapsed_time + eta
        progress = (epoch * self.total_batches + batch) / (self.total_epochs * self.total_batches)
        end_time = self.times[-1] + eta
        return progress, total_elapsed_time, expected_time, eta, end_time

# --------------------- Checkpoints ---------------------
def save_checkpoint(checkpoint_path, epoch, G, D, opt_G, opt_D, scaler_G, scaler_D):
    """
    Salva un checkpoint del modello.

    Args:
        checkpoint_path (str): Percorso dove salvare il checkpoint.
        epoch (int): Epoca corrente.
        G (nn.Module): Modello del generatore.
        D (nn.Module): Modello del discriminatore.
        opt_G (torch.optim.Optimizer): Ottimizzatore del generatore.
        opt_D (torch.optim.Optimizer): Ottimizzatore del discriminatore.
        scaler_G (torch.cuda.amp.GradScaler): GradScaler per il generatore.
        scaler_D (torch.cuda.amp.GradScaler): GradScaler per il discriminatore.
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizerG_state_dict': opt_G.state_dict(),
        'optimizerD_state_dict': opt_D.state_dict(),
        'scalerG_state_dict': scaler_G.state_dict(),
        'scalerD_state_dict': scaler_D.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, G, D, opt_G, opt_D, scaler_G, scaler_D, device):
    """
    Carica un checkpoint del modello.

    Args:
        checkpoint_path (str): Percorso del checkpoint da caricare.
        G (nn.Module): Modello del generatore.
        D (nn.Module): Modello del discriminatore.
        opt_G (torch.optim.Optimizer): Ottimizzatore del generatore.
        opt_D (torch.optim.Optimizer): Ottimizzatore del discriminatore.
        scaler_G (torch.cuda.amp.GradScaler): GradScaler per il generatore.
        scaler_D (torch.cuda.amp.GradScaler): GradScaler per il discriminatore.
        device (torch.device): Dispositivo su cui caricare i modelli.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint['generator_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_G.load_state_dict(checkpoint['optimizerG_state_dict'])
    opt_D.load_state_dict(checkpoint['optimizerD_state_dict'])
    scaler_G.load_state_dict(checkpoint['scalerG_state_dict'])
    scaler_D.load_state_dict(checkpoint['scalerD_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Riprendi dalla successiva
    return start_epoch

# --------------------- Funzioni utili ---------------------
def save_images(path, input, output, target, epoch, batch_index):
    """
    Salva le immagini di input, output e target in formato PNG.
    
    Args:
        path (str): Percorso della cartella di salvataggio.
        input (Tensor): Immagine di input.
        output (Tensor): Immagine generata dal modello.
        target (Tensor): Immagine target (stained).
        epoch (int): Numero dell'epoca corrente.
        batch_index (int): Indice del batch corrente.
    """
    # Salva le immagini di input, output e target
    # Rimettiamo le immagini da [-1,1] a [0,1]
    save_image((input * 0.5 + 0.5), os.path.join(path, f"epoch{epoch}_batch{batch_index}_input.tif"))
    save_image((output * 0.5 + 0.5), os.path.join(path, f"epoch{epoch}_batch{batch_index}_output.tif"))
    save_image((target * 0.5 + 0.5), os.path.join(path, f"epoch{epoch}_batch{batch_index}_target.tif"))

# --------------------- Training e Validazione ---------------------
def validate(G, D, validation_loader, device, bce_loss, l1_loss, epoch, log_file):
    """
    Funzione di validazione del modello.
    Questa funzione calcola la loss media del generatore e del discriminatore.

    Args:
        G (nn.Module): Modello del generatore.
        D (nn.Module): Modello del discriminatore.
        validation_loader (DataLoader): DataLoader per il set di validazione.
        device (torch.device): Dispositivo su cui eseguire i calcoli (CPU o GPU).
        bce_loss (nn.Module): Funzione di perdita BCE (Binary Cross Entropy).
        l1_loss (nn.Module): Funzione di perdita L1 (Mean Absolute Error).
        epoch (int): Numero dell'epoca corrente.
        log_file (str): Percorso del file di log.

    Returns:
        tuple: Media delle perdite del generatore e del discriminatore.
    """
    # Metti in eval mode
    G.eval()
    D.eval()

    # Crea/assicurati che esista la cartella per gli output di validazione
    os.makedirs("Materiale/Locale/output_val", exist_ok=True)

    total_loss_G = 0.0
    total_loss_D = 0.0
    count = 0

    # Niente gradienti
    with torch.no_grad(), autocast("cuda"):
        for i, (x, y) in enumerate(validation_loader):
            x, y = x.to(device), y.to(device)

            # Genera immagini di output
            fake = G(x)

            # Predizioni del discriminatore su reale e falso
            D_real = D(x, y)
            D_fake = D(x, fake)

            # BCE labels
            real_label = torch.ones_like(D_real, device=device)
            fake_label = torch.zeros_like(D_fake, device=device)

            # Calcolo delle due loss (D e G)
            loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)
            loss_G = bce_loss(D_fake, real_label) + l1_loss(fake, y) * 25

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            count += 1

            # --- Salvataggio immagini di esempio ---
            # Ad esempio, salvi i primi 5 batch
            if i < 5:
                # Salva la prima immagine del batch (indice 0)
                save_images("Materiale/Locale/output_val", x[0], fake.detach()[0], y[0], epoch, i)

    # Ritorni la media delle loss
    avg_loss_D = total_loss_D / count
    avg_loss_G = total_loss_G / count

    # Log
    log_message(f"[Epoca {epoch}] Validation: loss_G={avg_loss_G:.4f} loss_D={avg_loss_D:.4f}", log_file)

    return avg_loss_G, avg_loss_D

def test_inference(checkpoint_path, test_folder, output_folder="Materiale/Locale/output_test", image_size=(512, 512), device="cuda"):
    """
    Funzione di test per il modello.
    Questa funzione esegue l'inferenza su un set di immagini e salva i risultati.

    Args:
        checkpoint_path (str): Percorso del checkpoint da caricare.
        test_folder (str): Percorso della cartella contenente le immagini di test.
        output_folder (str): Percorso della cartella in cui salvare le immagini di output.
        device (torch.device): Dispositivo su cui eseguire i calcoli (CPU o GPU).
    """

    # Crea cartella per gli output se non esiste già
    os.makedirs(output_folder, exist_ok=True)

    # Carica il checkpoint
    G = UNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()

    # Trasformazioni da applicare alle immagini di test
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Iterazione sui file nella cartella di test
    test_files = sorted([
        f for f in os.listdir(test_folder)
        if f.lower().endswith('_label_free.tif')
    ])
    if not test_files:
        print(f"Nessun file trovato nella cartella di test: {test_folder}")
        return

    with torch.no_grad(), autocast(device_type="cuda"):
        for i, filename in enumerate(test_files):

            # Carico l'immagine di test label-free
            img_path = os.path.join(test_folder, filename)
            img = Image.open(img_path).convert('RGB')

            # Applico la trasfromazione
            img_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

            # Passa dal generatore
            fake_stained = G(img_tensor)  # shape: (1, 3, H, W)

            # Riporta in [0,1] per salvataggio
            fake_stained = (fake_stained * 0.5) + 0.5
            fake_stained = fake_stained.clamp(0,1)

            # Salva l'immagine generata
            out_filename = f"{os.path.splitext(filename)[0]}_generated.tif"
            out_path = os.path.join(output_folder, out_filename)

            # Converte da tensor a PIL e salva
            save_image(fake_stained, out_path)

            # print(f"{out_filename} salvata in {output_folder}")
    print(f"Test completato. Immagini salvate in {output_folder}")



def train_one_epoch(G, D, training_loader, device, opt_G, opt_D, scaler_G, scaler_D, bce_loss, l1_loss, epoch, log_file, progress_tracker):
    """
    Funzione di training per un'epoca.
    Questa funzione esegue il training del generatore e del discriminatore per un'epoca.
    Calcola le perdite e aggiorna i pesi dei modelli.

    Args:
        G (nn.Module): Modello del generatore.
        D (nn.Module): Modello del discriminatore.
        training_loader (DataLoader): DataLoader per il set di addestramento.
        device (torch.device): Dispositivo su cui eseguire i calcoli (CPU o GPU).
        opt_G (torch.optim.Optimizer): Ottimizzatore per il generatore.
        opt_D (torch.optim.Optimizer): Ottimizzatore per il discriminatore.
        scaler_G (torch.cuda.amp.GradScaler): GradScaler per il generatore.
        scaler_D (torch.cuda.amp.GradScaler): GradScaler per il discriminatore.
        bce_loss (nn.Module): Funzione di perdita BCE (Binary Cross Entropy).
        l1_loss (nn.Module): Funzione di perdita L1 (Mean Absolute Error).
        epoch (int): Numero dell'epoca corrente.
        log_file (str): Percorso del file di log.
        progress_tracker (ProgressTracker): Oggetto per monitorare il progresso.
    """
    # Metti in training mode
    G.train()
    D.train()
    
    for i, (x, y) in enumerate(training_loader):
        # (x, y) sono le immagini di input e target
        # x sono batch_size immagini di input (label-free)
        # y sono batch_size immagini di target (stained)

        # Trasferimento su GPU
        x, y = x.to(device), y.to(device)
        
        # ---------- DISCRIMINATORE ----------
        with autocast("cuda"):
            # Genera immagini false
            fake = G(x).detach()
            # .detach() fa sì che quando si utilizza fake, non si calcolino i gradienti per il generatore
            # Di fatto evita che qualsiasi funzione che interagisce con fake possa modificare i gradienti del generatore

            # Calcola le predizioni del discriminatore
            # D_real è la predizione per le immagini reali
            # D_fake è la predizione per le immagini false
            D_real = D(x, y)
            D_fake = D(x, fake)

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
            fake = G(x)

            # Calcola le predizioni del discriminatore per le immagini false
            D_fake = D(x, fake)

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

        progress, total_elapsed_time, expected_time, eta, end_time = progress_tracker.calculate_progress(epoch, i)
        # progress_str = f"{progress:.2%} | Durata esecuzione: {total_elapsed_time:.2f}s | Durata stimata: {expected_time:.2f}s | ETA: {eta:.2f}s | Fine: {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"
        progress_str = f"{progress:.2%} | Fine stimata: {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"


        # Stampa e log su file
        if i % log_rate == 0:
            log_message(f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} loss_D: {loss_D.item():.4f} - {progress_str}", log_file)
            # Non serve salvare le immagini di training dato che le salviamo in validate
            # save_images("Materiale/Locale/output_pix2pix", x[0], fake.detach()[0], y[0], epoch, i)

# --------------------- Main ---------------------
def main():
    start_time = time.time()

    # Crea la cartella per i log (se non esiste già)
    os.makedirs("Materiale/Locale/output_pix2pix_logs", exist_ok=True)

    # Genero un nome file log con data/ora attuale (unico per ogni esecuzione).
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Materiale/Locale/output_pix2pix_logs/Log-{timestamp_str}.txt"
    # Se esiste già un file con lo stesso nome, lo cancello (per evitare conflitti)
    if os.path.exists(log_file):
        os.remove(log_file)
    log_message("Script avviato", log_file)

    # Imposta il seed per la riproducibilità
    set_seed(seed)
    log_message(f"Seed impostato a {seed}", log_file)

    # Imposta il dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Dispositivo: {device}", log_file)

    # Trasformazioni per il dataset
    # Normalizzazione delle immagini tra -1 e 1 (per il generatore)
    transform = transforms.Compose([
        transforms.Resize(image_size), # Risoluzione delle immagini, potrebbe essere 256 o 512
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    training_dataset = PairedHistologyDataset("Materiale/Locale/dataset_split/train", transform)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=training_shuffle, num_workers=n_workers, pin_memory=True) # prima num_workers era a 0 e pin_memory non c'era e batch_size a 4, Shuffle era a Flase
    # andrebbero presi i tempi precisi per ogni configurazione (profiling(?)), ma in linea di massima:
    # con num_workers=12 impiega circa 1.27 minuti, 1.28 e 1.22
    # con num_workers=8 impiega circa 1.26 minuti, 1.26 e 1.34
    # con num_workers=4 impega circa 1.41 minuti, 1.41 e 1.38
    # con num_workers=0 impiega circa 1.27 minuti, 1.23 e 1.29

    validation_dataset = PairedHistologyDataset("Materiale/Locale/dataset_split/val", transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=validation_shuffle, num_workers=n_workers, pin_memory=True)

    # Inizializzazione del modello
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # Inizializzazione degli ottimizzatori
    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

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

    # Checkpoint
    start_epoch = 0
    if use_checkpoint:
        os.makedirs("Materiale/Locale/checkpoints", exist_ok=True)
        # Se vuoi usare i checkpoint, carica il checkpoint esistente
        if os.path.exists(restore_checkpoint_path):
            start_epoch = load_checkpoint(restore_checkpoint_path, generator, discriminator, 
                                        opt_G, opt_D, scaler_G, scaler_D, device)
            log_message(f"Checkpoint caricato da {restore_checkpoint_path}, epoca {start_epoch}", log_file)
        else:
            log_message("ATTENZIONE - Checkpoint non trovato", log_file)
    else:
        log_message("Nessun checkpoint caricato", log_file)

    # Crea il progress tracker
    progress_tracker = ProgressTracker(n_epochs, len(training_loader))
    progress_tracker.start()

    # Inizio allenamento
    log_message("Inizio allenamento", log_file)
    for epoch in range(start_epoch, n_epochs):
        log_message(f"Inizio epoca {epoch}", log_file)

        # ---------- ALLENAMENTO (una epoca) ----------
        train_one_epoch(generator, discriminator, training_loader, device, opt_G, opt_D, scaler_G, scaler_D,
                        bce_loss, l1_loss, epoch, log_file, progress_tracker)

        # Log fine epoca
        log_message(f"Fine epoca {epoch}", log_file)

        # ------ SALVATAGGIO CHECKPOINT ------
        # Salva ogni epoca (o magari ogni 5 epoche, se preferisci)
        if use_checkpoint:
            if (epoch + 1) % checkpoint_rate == 0:
                checkpoint_path = f"Materiale/Locale/checkpoints/checkpoint_Pix2Pix_epoca{epoch}_{timestamp_str}.pth"
                save_checkpoint(checkpoint_path, epoch, generator, discriminator, opt_G, opt_D, scaler_G, scaler_D)
                log_message(f"Checkpoint salvato in {checkpoint_path} all'epoca {epoch}", log_file)

        # ---------- VALIDATION (in corrispondenza con checkpoint_rate) ----------
        if(epoch + 1) % validate_rate == 0:
            validate(generator, discriminator, validation_loader, device, bce_loss, l1_loss, epoch, log_file)

    # Fine allenamento
    end_time = time.time()
    total_seconds = end_time - start_time
    log_message(f"Fine esecuzione. Tempo impiegato = {total_seconds:.2f} secondi", log_file)
              

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        # Esegui il test con un checkpoint esistente
        print(f"Parametro 'test' fornito. Inizio test con il checkpoint in: {restore_checkpoint_path}")
        test_inference(restore_checkpoint_path, test_folder="Materiale/Locale/dataset_split/test", output_folder="Materiale/Locale/output_test", image_size=image_size, device="cuda")
    else:
        # Esegui il training
        print("Parametro 'test' non fornito. Inizio allenamento.")
        main()