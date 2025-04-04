import os, random, time, datetime, math
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
n_epochs = 5 # Numero di epoche da eseguire
log_rate = 15 # Ogni quanto loggare (es. ogni 10 batch)
use_checkpoint = True # Se vuoi riprendere da un checkpoint esistente, metti True       Bisognerebbe creare create_checkpoint e load_checkpoint così si possono distinguere le casiistiche
checkpoint_rate = 15 # Ogni quanto salvare i checkpoint (es. ogni 10 epoche)
restore_checkpoint_path = "Materiale/Locale/Pix2Pix+/checkpoints/PASTE_HERE_THE_CHECKPOINT_NAME" # Percorso del checkpoint (se esiste), ricordati di cambiare il nome
validate_rate = 1 # Ogni quanto validare (es. ogni 5 epoche), prendiamo per buono al momento come valore standard =checkpoint_rate
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


# --------------------- Generatore (UNet3+ Semplificato) ---------------------
class UNet3PlusGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNet3PlusGenerator, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_filters * 8, base_filters * 16)

        # Decoder con full-scale skip
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

        self.up4 = up_block(base_filters * (1 + 2 + 4 + 8 + 16), base_filters * 8)
        self.up3 = up_block(base_filters * (1 + 2 + 4 + 8), base_filters * 4)
        self.up2 = up_block(base_filters * (1 + 2 + 4), base_filters * 2)
        self.up1 = up_block(base_filters * (1 + 2), base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)  # 512x512
        e2 = self.enc2(self.pool1(e1))  # 256x256
        e3 = self.enc3(self.pool2(e2))  # 128x128
        e4 = self.enc4(self.pool3(e3))  # 64x64
        b  = self.bottleneck(self.pool4(e4))  # 32x32

        # Upsample tutti per il decoder
        u4_in = torch.cat([
            nn.functional.interpolate(e1, size=b.shape[2:]),
            nn.functional.interpolate(e2, size=b.shape[2:]),
            nn.functional.interpolate(e3, size=b.shape[2:]),
            nn.functional.interpolate(e4, size=b.shape[2:]),
            b
        ], dim=1)
        d4 = self.up4(u4_in)

        u3_in = torch.cat([
            nn.functional.interpolate(e1, size=d4.shape[2:]),
            nn.functional.interpolate(e2, size=d4.shape[2:]),
            nn.functional.interpolate(e3, size=d4.shape[2:]),
            d4
        ], dim=1)
        d3 = self.up3(u3_in)

        u2_in = torch.cat([
            nn.functional.interpolate(e1, size=d3.shape[2:]),
            nn.functional.interpolate(e2, size=d3.shape[2:]),
            d3
        ], dim=1)
        d2 = self.up2(u2_in)

        u1_in = torch.cat([
            nn.functional.interpolate(e1, size=d2.shape[2:]),
            d2
        ], dim=1)
        d1 = self.up1(u1_in)

        return self.final(d1)


# --------------------- Discriminatore (PatchGAN Migliorato) ---------------------
class AdvancedPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6, base_filters=64, use_sigmoid=False):
        super(AdvancedPatchGANDiscriminator, self).__init__()
        layers = []

        # Questo è un blocco di convoluzione con normalizzazione e ReLU
        # Usiamo questo invece di DoubleConv perchè rischia di essere instabile 
        # ed è meno usato nel contesto di GANs
        def conv_block(in_ch, out_ch, stride=2, use_norm=False):
            block = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
            if use_norm:
                block.append(nn.InstanceNorm2d(out_ch))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        # Primo blocco (senza normalization)
        layers += conv_block(in_channels, base_filters, use_norm=False)

        # Blocchi intermedi con stride 2 (nel discriminatore si usa stride 2 per ridurre la risoluzione)
        filters = base_filters
        for _ in range(3):  # più profondo di PatchGAN classico
            layers += conv_block(filters, filters * 2)
            filters *= 2

        # Blocchi finali con stride 1 (aumentano receptive field senza ridurre risoluzione)
        for _ in range(2):
            layers += conv_block(filters, filters, stride=1)

        # Layer finale
        layers.append(nn.Conv2d(filters, 1, kernel_size=4, stride=1, padding=1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

def compute_gradient_penalty(D, real_samples, fake_samples, x_input, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(x_input, interpolates)
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


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
    def __init__(self, total_epochs, total_batches, max_history=500):
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
def validate(G, D, validation_loader, device, l1_loss, epoch, log_file):
    """
    Funzione di validazione del modello.
    Questa funzione calcola la loss media del generatore e del discriminatore.

    Args:
        G (nn.Module): Modello del generatore.
        D (nn.Module): Modello del discriminatore.
        validation_loader (DataLoader): DataLoader per il set di validazione.
        device (torch.device): Dispositivo su cui eseguire i calcoli (CPU o GPU).
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
    os.makedirs("Materiale/Locale/Pix2Pix+/output_val", exist_ok=True)

    total_loss_G = 0.0
    total_loss_D = 0.0
    count = 0

    # Niente gradienti
    with autocast("cuda"):
        for i, (x, y) in enumerate(validation_loader):
            x, y = x.to(device), y.to(device)

            # Genera immagini di output
            fake = G(x)

            # Predizioni del discriminatore su reale e falso
            D_real = D(x, y)
            D_fake = D(x, fake)

            loss_D_real = -torch.mean(D_real)
            loss_D_fake = torch.mean(D_fake)
            gp = compute_gradient_penalty(D, y, fake, x, device)
            lambda_gp = 10

            loss_D = loss_D_real + loss_D_fake + lambda_gp * gp
            loss_G = -torch.mean(D_fake) + l1_loss(fake, y) * 25
            
            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            count += 1

            # --- Salvataggio immagini di esempio ---
            # Ad esempio, salvi i primi 5 batch
            if i < 5:
                # Salva la prima immagine del batch (indice 0)
                save_images("Materiale/Locale/Pix2Pix+/output_val", x[0], fake.detach()[0], y[0], epoch, i)

    # Ritorni la media delle loss
    avg_loss_D = total_loss_D / count
    avg_loss_G = total_loss_G / count

    # Log
    log_message(f"[Epoca {epoch}] Validation: loss_G={avg_loss_G:.4f} loss_D={avg_loss_D:.4f}", log_file)

    return avg_loss_G, avg_loss_D

def test_inference(checkpoint_path, test_folder, output_folder="Materiale/Locale/Pix2Pix+/output_test", image_size=(512, 512), device="cuda"):
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
    G = UNet3PlusGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    log_message(f"Checkpoint caricato.")

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

def train_one_epoch(G, D, training_loader, device, opt_G, opt_D, scaler_G, scaler_D, l1_loss, epoch, log_file, progress_tracker):
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

            # WGAN-GP losses
            loss_D_real = -torch.mean(D_real)
            loss_D_fake = torch.mean(D_fake)
            gp = compute_gradient_penalty(D, y, fake, x, device)
            lambda_gp = 10  # Costante consigliata nel paper

            loss_D = loss_D_real + loss_D_fake + lambda_gp * gp

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

            # Loss del generatore in WGAN
            loss_G = -torch.mean(D_fake) + l1_loss(fake, y) * 25  # puoi tenere il tuo L1

        # Ottimizzazione del generatore
        opt_G.zero_grad() # zero_grad() azzera i gradienti accumulati
        scaler_G.scale(loss_G).backward() # scaler.scale() calcola i gradienti in modo scalato
        scaler_G.step(opt_G) # scaler.step() applica i gradienti all'ottimizzatore
        scaler_G.update()

        progress, total_elapsed_time, expected_time, eta, end_time = progress_tracker.calculate_progress(epoch, i)
        progress_str = f"{progress:.2%} | {total_elapsed_time/3600:.1f}h/{expected_time/3600:.1f}h {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"
        # progress_str = f"{progress:.2%} | Fine stimata: {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"


        # Stampa e log su file
        if i % log_rate == 0:
            log_message(f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} loss_D: {loss_D.item():.4f} - {progress_str}", log_file)
            # Non serve salvare le immagini di training dato che le salviamo in validate
            # save_images("Materiale/Locale/Pix2Pix+/output_train", x[0], fake.detach()[0], y[0], epoch, i)


# --------------------- Main ---------------------
def main():
    start_time = time.time()

    # Crea la cartella per i log (se non esiste già)
    os.makedirs("Materiale/Locale/Pix2Pix+/logs", exist_ok=True)

    # Genero un nome file log con data/ora attuale (unico per ogni esecuzione).
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Materiale/Locale/Pix2Pix+/logs/Log-{timestamp_str}.txt"
    # Se esiste già un file con lo stesso nome, lo cancello (per evitare conflitti)
    if os.path.exists(log_file):
        os.remove(log_file)

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
    generator = UNet3PlusGenerator().to(device)
    discriminator = AdvancedPatchGANDiscriminator(use_sigmoid=False).to(device)
    
    # Inizializzazione degli ottimizzatori
    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Inizializzazione del GradScaler per la precisione mista
    scaler_G = GradScaler(device='cuda')
    scaler_D = GradScaler(device='cuda')

    # Inizializzazione delle funzioni di perdita
    l1_loss = nn.L1Loss()

    # Creazione cartella per le immagini di preview e cancellazione dei file esistenti
    os.makedirs("Materiale/Locale/Pix2Pix+/output_train", exist_ok=True)
    for file in os.listdir("Materiale/Locale/Pix2Pix+/output_train"):
        os.remove(os.path.join("Materiale/Locale/Pix2Pix+/output_train", file))

    # Checkpoint
    start_epoch = 0
    if use_checkpoint:
        os.makedirs("Materiale/Locale/Pix2Pix+/checkpoints", exist_ok=True)
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
                        l1_loss, epoch, log_file, progress_tracker)

        # Log fine epoca
        log_message(f"Fine epoca {epoch}", log_file)

        # ------ SALVATAGGIO CHECKPOINT ------
        # Salva ogni epoca (o magari ogni 5 epoche, se preferisci)
        if use_checkpoint:
            if (epoch + 1) % checkpoint_rate == 0:
                checkpoint_path = f"Materiale/Locale/Pix2Pix+/checkpoints/checkpoint_Pix2Pix_epoca{epoch}_{timestamp_str}.pth"
                save_checkpoint(checkpoint_path, epoch, generator, discriminator, opt_G, opt_D, scaler_G, scaler_D)
                log_message(f"Checkpoint salvato in {checkpoint_path} all'epoca {epoch}", log_file)

        # ---------- VALIDATION (in corrispondenza con checkpoint_rate) ----------
        if(epoch + 1) % validate_rate == 0:
            validate(generator, discriminator, validation_loader, device, l1_loss, epoch, log_file)

    # Fine allenamento
    end_time = time.time()
    total_seconds = end_time - start_time
    log_message(f"Fine esecuzione. Tempo impiegato = {total_seconds:.2f} secondi", log_file)
              

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        # Esegui il test con un checkpoint esistente
        print(f"Inizio test con il checkpoint in: {restore_checkpoint_path}")
        test_inference(restore_checkpoint_path, test_folder="Materiale/Locale/dataset_split/test", output_folder="Materiale/Locale/Pix2Pix+/output_test", image_size=image_size, device="cuda")
    elif len(sys.argv) >= 2 and sys.argv[1] == "train":
        print(f"Inizio allenamento.")
        main()
    else:
        print("Uso: python Pix2Pix.py [train/test]")
        print("Esegui 'train' per allenare il modello o 'test' per eseguire il test con un checkpoint esistente.")

"""
Epoca	loss_D tipica	        loss_G tipica
0	    1 → 0 → -1 → ±0.1	    Alta (10+) → ↓
5+	    -0.5 <→> 0.5	        3-5, poi ↓
"""