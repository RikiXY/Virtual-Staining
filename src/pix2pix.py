import os
import random
import time
import datetime
import json
import argparse
from pathlib import Path

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.utils import save_image


# --------------------- Percorsi di lavoro ---------------------
# Questa funzione costruisce tutti i percorsi principali del progetto
# partendo da una sola root. Evita di spargere stringhe hardcoded nel
#  resto del file.
def build_workspace_paths(run_root: str | Path) -> dict:
    root = Path(run_root)

    return {
        "run_root": root,
        "logs_dir": root / "logs",
        "checkpoints_dir": root / "checkpoints",
        "output_val_dir": root / "output_val",
        "output_test_dir": root / "output_test",
        "output_train_dir": root / "output_train",
    }


# --------------------- Argomenti da riga di comando ---------------------
# Separiamo parser e logica. 
# Lo script supporta due modalità distinte: addestramento e test.
def build_parser():
    parser = argparse.ArgumentParser(
    prog="python src/pix2pix.py",
    description="Train or test the Pix2Pix model on a paired histology dataset.",
    epilog=(
        "Examples:\n"
        "  python src/pix2pix.py train "
        "--dataset-root local_workspace/datasets/inverted_512 "
        "--run-name L1-25 "
        "--epochs 100\n"
        "\n"
        "  python src/pix2pix.py test "
        "--dataset-root local_workspace/datasets/inverted_512 "
        "--run-path local_workspace/results/L1-25 "
        "--checkpoint local_workspace/results/L1-25/checkpoints/model.pth\n"
        "\n"
        "Use 'python src/pix2pix.py <command> --help' "
        "to see the options for a specific command."
    ),
    formatter_class=argparse.RawTextHelpFormatter
)

    # I subparser permettono di avere sottocomandi diversi all'interno
    # dello stesso script, ciascuno con i propri argomenti.
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the Pix2Pix model",
        formatter_class=argparse.RawTextHelpFormatter
    )
    train_parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root containing dataset_train/ and dataset_val/"
    )
    train_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the output run directory to create"
    )
    train_parser.add_argument(
        "--results-path",
        type=str,
        default="local_workspace/results",
        help="Base directory where the new run folder will be created (default: local_workspace/results)"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, a random seed is generated."
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the DataLoader (default: 8)"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of DataLoader workers (default: 12)"
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(512, 512),
        help="Resize images before training, e.g. --image-size 512 512"
    )
    train_parser.add_argument(
        "--log-rate",
        type=int,
        default=15,
        help="Log every N batches (default: 15)"
    )
    train_parser.add_argument(
        "--checkpoint-rate",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (default: 10)"
    )
    train_parser.add_argument(
        "--validate-rate",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1)"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from"
    )
    train_parser.add_argument(
        "--l1-lambda",
        type=float,
        default=25.0,
        help="Weight of the L1 reconstruction loss (default: 25.0)"
    )
    train_parser.add_argument(
        "--lr-g",
        type=float,
        default=2e-4,
        help="Learning rate for the generator (default: 2e-4)"
    )
    train_parser.add_argument(
        "--lr-d",
        type=float,
        default=2e-4,
        help="Learning rate for the discriminator (default: 2e-4)"
    )
    train_parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5)"
    )
    train_parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999)"
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Run inference on the test set",
        formatter_class=argparse.RawTextHelpFormatter
    )
    test_parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root containing dataset_test/"
    )
    test_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to use for inference"
    )
    test_parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="Path to an existing training run containing checkpoints/ and output folders"
    )
    test_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(512, 512),
        help="Resize images before inference, e.g. --image-size 512 512"
    )

    return parser

# ====================[DATASET]====================
# Questo dataset contiene coppie di immagini source e target.
# A ogni immagine source deve corrispondere la rispettiva immagine target.
#
# Implementare `__len__` e `__getitem__` è il requisito minimo richiesto
# da PyTorch per una classe `Dataset`: il DataLoader usa questi metodi per
# sapere quanti campioni esistono e come recuperarli quando costruisce i batch.
class PairedHistologyDataset(Dataset):
    VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}
        
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # Le coppie vengono costruite una volta sola all'inizio, così
        # durante training e validation non dobbiamo esplorare la cartella
        # ogni volta.
        self.pairs = self._discover_pairs()

    def _discover_pairs(self):
        grouped = {}

        for filename in sorted(os.listdir(self.folder_path)):
            file_path = os.path.join(self.folder_path, filename)

            if not os.path.isfile(file_path):
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in self.VALID_IMAGE_EXTENSIONS:
                continue

            stem = Path(filename).stem

            if stem.startswith("mask_") or "_mask_" in stem:
                continue

            parts = stem.split("_")
            if len(parts) < 3:
                continue

            key = (parts[0], parts[1])
            grouped.setdefault(key, []).append(file_path)

        samples = []
        for key in sorted(grouped):
            files = grouped[key]

            source_path = None
            target_path = None

            for file_path in files:
                stem = Path(file_path).stem.lower()

                if stem.endswith("_source"):
                    source_path = file_path
                elif stem.endswith("_target"):
                    target_path = file_path

            if source_path is None or target_path is None:
                continue

            samples.append((source_path, target_path))

        return samples

    def __len__(self):
        # Restituiamo quante coppie valide sono state trovate.
        return len(self.pairs)

    def __getitem__(self, idx):
        # Dato un indice, recuperiamo il prefisso comune e apriamo entrambe le immagini della coppia.
        source_path, target_path = self.pairs[idx]

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            # La stessa trasformazione viene applicata a source e target.
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image
# =================================================

def is_amp_enabled(device):
    # La mixed precision con `autocast` e `GradScaler` ha senso soprattutto
    # su GPU CUDA. Su CPU lasciamo tutto disattivato per evitare complessità
    # inutile e mantenere il comportamento più lineare possibile.
    return isinstance(device, torch.device) and device.type == "cuda"

# ====================[CONFIGURAZIONE]====================
# Alcuni parametri dei blocchi convoluzionali vengono letti da JSON.
# è una scelta pratica: i dettagli architetturali restano centralizzati
# e modificabili senza dover ritoccare il codice del modello.
script_dir = Path(__file__).resolve().parent
settings_path = script_dir / "json" / "p2p_settings.json"

with settings_path.open("r", encoding="utf-8") as s:
    SETTINGS = json.load(s)
# ========================================================

# ====================[GENERATOR (U-NET)]====================
# Il generatore segue la struttura classica della U-Net:
# encoder -> bottleneck -> decoder, con skip connections tra livelli simmetrici.
#
# L'idea è questa:
# - l'encoder comprime e amplia il contesto;
# - il bottleneck conserva una rappresentazione più astratta;
# - il decoder ricostruisce l'immagine;
# - le skip connections riportano indietro dettagli fini che altrimenti
#   andrebbero persi nella discesa.
class DoubleConv(nn.Module):
    """
    Blocco composto da due convoluzioni consecutive, ciascuna seguita
    da batch normalization e attivazione ReLU.

    Args:
        in_channels (int): Numero di canali in ingresso.
        out_channels (int): Numero di canali in uscita.
    """
    def __init__(self, in_channels, out_channels):
        conv_params = SETTINGS["double_conv"]
        super(DoubleConv, self).__init__()
        # Due convoluzioni di fila sono un blocco molto usato nelle U-Net:
        # la prima inizia a trasformare le feature, la seconda le rifinisce.
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_params['kernel_size'], stride=conv_params["stride"], padding=conv_params['padding'], bias=conv_params['bias']),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(conv_params['inplace']),
            nn.Conv2d(out_channels, out_channels, kernel_size=conv_params['kernel_size'], stride=conv_params["stride"], padding=conv_params['padding'], bias=conv_params['bias']),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(conv_params['inplace'])
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Blocco di downsampling per architetture in stile U-Net.

    Riduce la risoluzione spaziale tramite max pooling e poi applica
    un blocco `DoubleConv` per estrarre feature più ricche.
    """
    def __init__(self, in_channels, out_channels):
        down_params = SETTINGS["down"]
        super(Down, self).__init__()
        # Qui dimezziamo la risoluzione spaziale e poi aumentiamo il numero
        # di canali. è il compromesso tipico dell'encoder: meno dettaglio
        # pixel per pixel, ma più capacità di rappresentare pattern complessi.
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=down_params["kernel_size"], stride=down_params["stride"]),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#TODO RIPRENDI A IMPLEMENTARE IL JSON DA QUI
class Up(nn.Module):
    """
    Blocco di upsampling seguito da `DoubleConv`.

    La feature map risalita viene concatenata con la skip connection
    dell'encoder prima della convoluzione finale.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Questo blocco appartiene al decoder. Prima si risale di risoluzione,
        # poi si combinano le feature del decoder con la skip connection
        # proveniente dall'encoder.
        
        # Il codice supporta due strategie di upsampling:
        # - bilinear: semplice e senza parametri addestrabili;
        # - transposed convolution: più flessibile, ma con pesi da imparare.
        # In bilinear usiamo un semplice upsampling seguito da una convoluzione per ridurre i canali.
        # In transposed convolution invece la convoluzione stessa fa anche da upsampling, dimezzando i canali in uscita.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # `x1` è la feature map corrente del decoder, che stiamo riallargando.
        # `x2` è la skip connection corrispondente arrivata dall'encoder.
        x1 = self.up(x1)
        
        # Concateniamo i canali, ovvero li uniamo nello stesso punto della rete,
        # con l'obiettivo di combinare insieme due tipi di informazione:
        # - il contesto appreso in profondità dal decoder, che ha visto l'immagine in modo più globale;
        # - i dettagli locali conservati dal ramo encoder.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Convoluzione finale 1x1 che porta le feature
    al numero di canali richiesto in output.
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
            n_channels (int): Numero di canali in input.
            n_classes (int): Numero di canali in output.
            bilinear (bool): Se usare upsampling bilineare o transposed convolution.
        """
        super(UNetGenerator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder: man mano che scendiamo, la risoluzione cala e i canali
        # aumentano. è il modo con cui la rete guadagna contesto senza
        # dover mantenere tutto il dettaglio spaziale a ogni livello.
        # Encoders, versione da 64 a 1024
        self.inc = DoubleConv(n_channels, 64) # Convoluzione iniziale
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)          # Bottleneck

        # Decoder: qui il processo si inverte e l'informazione viene riportata
        # gradualmente verso una forma di nuovo "immagine-like".
        # Decoders, versione da 1024 a 64
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Convoluzione finale, versione da 64 a N (Per RGB è 3)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Il forward della U-Net ha una logica molto regolare:
        # prima si scende nell'encoder salvando le feature intermedie,
        # poi si risale nel decoder riusando quelle stesse feature
        # tramite le skip connections.

        # Per immagini RGB, N vale 3
        # Per immagini grayscale, N vale 1

        # Encoder
        x1 = self.inc(x)       # Shape: [N, 64, 512, 512]
        x2 = self.down1(x1)    # Shape: [N, 128, 256, 256]
        x3 = self.down2(x2)    # Shape: [N, 256, 128, 128]
        x4 = self.down3(x3)    # Shape: [N, 512, 64, 64]
        x5 = self.down4(x4)    # Shape: [N, 1024, 32, 32]

        # `x5` è il bottleneck: la rappresentazione più compressa e più astratta.
        # Qui la rete conserva il contesto globale dell'immagine.
        # Decoder
        x = self.up1(x5, x4)   # Shape: [N, 512, 64, 64]
        x = self.up2(x, x3)    # Shape: [N, 256, 128, 128]
        x = self.up3(x, x2)    # Shape: [N, 128, 256, 256]
        x = self.up4(x, x1)    # Shape: [N, 64, 512, 512]
        
        # Il layer finale trasforma le feature ricostruite nell'output vero e proprio.
        logits = self.outc(x)  # Shape: [N, n_classes, 512, 512]
        return logits

# --------------------- Discriminatore (PatchGAN) ---------------------
# Il discriminatore valuta la coerenza della coppia condizionale:
# immagine di partenza + target reale o generato.
class PatchGANDiscriminator(nn.Module):
    """
    Discriminatore PatchGAN per task image-to-image.

    Produce una mappa NxN di predizioni real/fake, una per ogni patch.
    """
    def __init__(self, in_channels=6, ndf=64, use_sigmoid=False):
        """
        Args:
            in_channels (int): Numero di canali in ingresso. In pix2pix,
                               concatenando input e output RGB si arriva a 6.
            ndf (int): Numero base di filtri del discriminatore.
            use_sigmoid (bool): Se applicare una sigmoid finale.
        """
        super(PatchGANDiscriminator, self).__init__()

        # PatchGAN non produce un singolo scalare finale, ma bensì
        # una feature map, una per ogni patch osservata dall'uscita.
        # è una scelta molto pratica: costringe il modello a curare le
        # texture locali, non soltanto l'aspetto globale dell'immagine.

        # Blocchi della convoluzione:
        # 1) Primo blocco senza normalizzazione
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # 2) Blocchi di downsampling
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

        # 3) Stride = 1 per l’ultima/e convoluzione/i, in modo che il campo ricettivo sia circa 70×70 (PatchGAN).
        curr_dim = next_dim
        next_dim = curr_dim * 2
        # Questo layer mantie stride=1 per evitare che la dimensione delle patch cresca troppo
        layers += [
            nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # 4) Layer di output
        # Il canale finale è uno, ma distribuito su una griglia spaziale.
        # Per questo l'output non è uno scalare singolo.
        layers += [
            nn.Conv2d(next_dim, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        # Quando si usa BCEWithLogitsLoss la sigmoid finale non serve,
        # perchè è già incorporata nella loss.
        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Immagine di input.
            y (Tensor): Target reale o output generato.

        Returns:
            Tensor: Mappa di predizioni real/fake per patch.
        """
        # Qui la concatenazione lungo i canali è il passaggio chiave:
        # il discriminatore giudica la coerenza della coppia condizionale,
        # non solo la qualità del target reale o generato preso da solo.
        return self.model(torch.cat([x, y], dim=1)) # Concatena source e target/output

# --------------------- Determinismo ---------------------
def set_seed(seed):
    # Fissare il seed riduce molto la variabilità e aiuta quando si confrontano esperimenti.
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
    # Loggiamo su file, oltre che a schermo, perchè così
    # dopo ore o giorni si puo' capire cosa è successo.
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if show_time:
        message = f"[{now_str}] {message}"
    with open(log_file, "a+") as f:
        f.write(message + "\n")
    if use_stdout:
        print(message)

def get_first_pair_size(dataset):
    """
    Restituisce la dimensione reale su disco della prima coppia del dataset.
    Serve per distinguere tra patch native e resize applicato dal training.
    """
    if len(dataset) == 0:
        return None

    source_path, target_path = dataset.pairs[0]

    with Image.open(source_path) as src_img:
        source_size = src_img.size  # (width, height)

    with Image.open(target_path) as tgt_img:
        target_size = tgt_img.size  # (width, height)

    return {
        "source": source_size,
        "target": target_size,
        "source_path": source_path,
        "target_path": target_path,
    }


def save_run_config(run_config, run_root):
    config_path = os.path.join(run_root, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
    return config_path


def log_run_header(log_file, run_config):
    """
    Scrive nel log un riepilogo iniziale ordinato della run.
    """
    log_message("=" * 80, log_file, show_time=False)
    log_message("RUN CONFIGURATION", log_file, show_time=False)
    log_message("=" * 80, log_file, show_time=False)

    for key, value in run_config.items():
        log_message(f"{key}: {value}", log_file, show_time=False)

    log_message("=" * 80, log_file, show_time=False)

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
            # Manteniamo una finestra temporale limitata, così la stima dell'ETA
            # resta reattiva anche se la velocità del training cambia nel tempo.
            self.times.pop(0)
        total_elapsed_time = self.times[-1] - self.start_time
        elapsed_time = self.times[-1] - self.times[0]
        eta = (elapsed_time / len(self.times)) * (self.total_epochs * self.total_batches - (epoch * self.total_batches + batch))
        expected_time = total_elapsed_time + eta
        progress = (epoch * self.total_batches + batch) / (self.total_epochs * self.total_batches)
        end_time = self.times[-1] + eta
        return progress, total_elapsed_time, expected_time, eta, end_time

# --------------------- Checkpoints ---------------------
def save_checkpoint(
    checkpoint_path, 
    epoch, 
    G, 
    D, 
    opt_G, 
    opt_D, 
    scaler_G, 
    scaler_D,
    l1_lambda,
    lr_g,
    lr_d,
    beta1,
    beta2,
    image_size=None,
    batch_size=None,
    num_workers=None,
    dataset_root=None
):
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
        "epoch": epoch,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "optimizerG_state_dict": opt_G.state_dict(),
        "optimizerD_state_dict": opt_D.state_dict(),
        "scalerG_state_dict": scaler_G.state_dict(),
        "scalerD_state_dict": scaler_D.state_dict(),
        "l1_lambda": l1_lambda,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "beta1": beta1,
        "beta2": beta2,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
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
    # Qui non carichiamo solo i pesi dei modelli, ma anche optimizer e scaler.
    # Questo rende il resume davvero coerente con il punto in cui il training
    # era stato interrotto.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint['generator_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_G.load_state_dict(checkpoint['optimizerG_state_dict'])
    opt_D.load_state_dict(checkpoint['optimizerD_state_dict'])
    scaler_G.load_state_dict(checkpoint['scalerG_state_dict'])
    scaler_D.load_state_dict(checkpoint['scalerD_state_dict'])
    # Riprendiamo dall'epoca successiva per non ripetere quella già completata.
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

# --------------------- Funzioni utili ---------------------
def save_images(path, input, output, target, epoch, batch_index):
    """
    Salva le immagini di input, output e target in formato TIF.

    Args:
        path (str): Percorso della cartella di salvataggio.
        input (Tensor): Immagine di input.
        output (Tensor): Immagine generata dal modello.
        target (Tensor): Immagine target corrispondente all'input corrente.
        epoch (int): Numero dell'epoca corrente.
        batch_index (int): Indice del batch corrente.
    """
    # Le immagini sono normalizzate in [-1, 1]; prima di salvarle per uso umano
    # dobbiamo riportarle nell'intervallo [0, 1].
    save_image((input * 0.5 + 0.5), os.path.join(path, f"epoch{epoch}_batch{batch_index}_input.tif"))
    save_image((output * 0.5 + 0.5), os.path.join(path, f"epoch{epoch}_batch{batch_index}_output.tif"))
    save_image((target * 0.5 + 0.5), os.path.join(path, f"epoch{epoch}_batch{batch_index}_target.tif"))

# --------------------- Training e Validazione ---------------------
# Training e validation hanno due ruoli diversi:
# - nel training aggiorniamo i pesi del modello;
# - nella validation misuriamo come si comporta su dati non usati
#   per gli aggiornamenti, quindi senza backpropagation.
def validate(
    G, 
    D, 
    validation_loader, 
    device, 
    bce_loss, 
    l1_loss, epoch, 
    log_file, 
    output_val_dir, 
    l1_lambda
):
    """
    Esegue un pass di validazione e calcola le loss medie
    di generatore e discriminatore.
    """
    G.eval()
    D.eval()

    os.makedirs(output_val_dir, exist_ok=True)

    total_loss_G = 0.0
    total_loss_D = 0.0
    count = 0
    amp_enabled = is_amp_enabled(device)

    # In validazione disattiviamo i gradienti: risparmiamo memoria e tempo,
    # e soprattutto evitiamo qualunque aggiornamento accidentale dei pesi.
    with torch.no_grad():
        for i, (x, y) in enumerate(validation_loader):
            x, y = x.to(device), y.to(device)

            # `autocast` abilita la mixed precision quando il device lo supporta.
            # In pratica alcune operazioni vengono svolte in precisione ridotta
            # per guadagnare velocità e memoria.
            with autocast(device_type=device.type, enabled=amp_enabled):
                fake = G(x)

                D_real = D(x, y)
                D_fake = D(x, fake)

                real_label = torch.ones_like(D_real, device=device)
                fake_label = torch.zeros_like(D_fake, device=device)

                # BCE misura la parte adversarial: quanto bene il discriminatore
                # distingue coppie reali e coppie false.
                loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)

                # La loss del generatore combina due obiettivi:
                # 1) ingannare il discriminatore;
                # 2) restare vicino al target corretto.
                
                # La componente L1 è utile per evitare output plausibili ma
                # scollegati dal target specifico del campione attuale.
                loss_G = bce_loss(D_fake, real_label) + l1_loss(fake, y) * l1_lambda

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            count += 1

            # Salviamo qualche preview.
            if i < 5:
                save_images(output_val_dir, x[0], fake[0], y[0], epoch, i)

    avg_loss_D = total_loss_D / count if count > 0 else 0.0
    avg_loss_G = total_loss_G / count if count > 0 else 0.0

    log_message(
        f"[Epoch {epoch}] Validation: loss_G={avg_loss_G:.4f} loss_D={avg_loss_D:.4f}",
        log_file
    )

    return avg_loss_G, avg_loss_D

def test_inference(
    checkpoint_path,
    test_folder,
    output_folder,
    image_size=(512, 512),
    device=None
):
    """
    Esegue inferenza sul test set e salva le immagini generate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)

    valid_exts = {".tif", ".tiff", ".png"}

    # In inferenza ci serve soltanto il generatore.
    # Il discriminatore serve solo in training.
    G = UNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint["generator_state_dict"])
    G.eval()

    # Il preprocessing deve restare coerente con quello del training,
    # altrimenti il modello riceverebbe input distribuiti in modo diverso.
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    test_files = sorted(
        f for f in os.listdir(test_folder)
        if (
            Path(f).suffix.lower() in valid_exts
            and Path(f).stem.lower().endswith("_source")
        )
    )

    if not test_files:
        print(f"No test source files found in: {test_folder}")
        return

    amp_enabled = is_amp_enabled(device)

    with torch.no_grad():
        for filename in test_files:
            img_path = os.path.join(test_folder, filename)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=amp_enabled):
                fake_target = G(img_tensor)

            # Riportiamo l'output nell'intervallo adatto al salvataggio.
            fake_target = (fake_target * 0.5) + 0.5
            fake_target = fake_target.clamp(0, 1)

            source_stem = Path(filename).stem
            source_ext = Path(filename).suffix.lower()
            prefix = source_stem[:-len("_source")]
            out_filename = f"{prefix}_target_generated{source_ext}"

            out_path = os.path.join(output_folder, out_filename)
            save_image(fake_target, out_path)

    print(f"Test completed. Images saved in {output_folder}")
    
def train_one_epoch(
    G,
    D,
    training_loader,
    device,
    opt_G,
    opt_D,
    scaler_G,
    scaler_D,
    bce_loss,
    l1_loss,
    epoch,
    log_file,
    progress_tracker,
    log_rate,
    l1_lambda
):
    """
    Addestra generatore e discriminatore per una singola epoca.
    """
    G.train()
    D.train()
    amp_enabled = is_amp_enabled(device)

    for i, (x, y) in enumerate(training_loader):
        x, y = x.to(device), y.to(device)

        # ---------- DISCRIMINATOR ----------
        # Prima aggiorniamo il discriminatore. In questo step il suo compito
        # è distinguere le coppie reali `(x, y)` da quelle sintetiche `(x, fake)`.
        with autocast(device_type=device.type, enabled=amp_enabled):
            # `.detach()` è fondamentale: vogliamo usare l'output del generatore
            # come esempio falso per D, ma senza far tornare il gradiente dentro G.
            fake = G(x).detach()

            D_real = D(x, y)
            D_fake = D(x, fake)

            real_label = torch.ones_like(D_real, device=device)
            fake_label = torch.zeros_like(D_fake, device=device)

            loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)

        opt_D.zero_grad()
        # Con mixed precision il GradScaler aiuta a evitare problemi numerici
        # durante il backward in precisione ridotta.
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # ---------- GENERATOR ----------
        # Ora aggiorniamo il generatore. Qui NON facciamo detach, perchè
        # adesso vogliamo che il gradiente attraversi davvero G e lo corregga.
        with autocast(device_type=device.type, enabled=amp_enabled):
            fake = G(x)
            D_fake = D(x, fake)

            # La loss del generatore combina due spinte complementari:
            # - BCE adversarial: far sembrare l'output abbastanza realistico
            #   da "convincere" il discriminatore;
            # - L1: mantenere fedeltà verso il target reale.
            loss_G = bce_loss(D_fake, real_label) + l1_loss(fake, y) * l1_lambda

        opt_G.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        progress, total_elapsed_time, expected_time, eta, end_time = progress_tracker.calculate_progress(epoch, i)
        progress_str = (
            f"{progress:.2%} | "
            f"{total_elapsed_time / 3600:.1f}h/{expected_time / 3600:.1f}h | "
            f"{datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if i % log_rate == 0:
            log_message(
                f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} "
                f"loss_D: {loss_D.item():.4f} - {progress_str}",
                log_file
            )

# --------------------- Main ---------------------
def main(
    dataset_root,
    run_root,
    logs_dir,
    checkpoints_dir,
    output_val_dir,
    output_train_dir,
    n_epochs,
    seed=None,
    batch_size=8,
    n_workers=12,
    image_size=(512, 512),
    log_rate=15,
    checkpoint_rate=10,
    validate_rate=1,
    resume_checkpoint=None,
    l1_lambda=25.0,
    lr_g=2e-4,
    lr_d=2e-4,
    beta1=0.5,
    beta2=0.999
):
    start_time = time.time()

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"Log-{timestamp_str}.txt")

    if os.path.exists(log_file):
        os.remove(log_file)

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    set_seed(seed)
    log_message(f"Seed set to {seed}", log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Device: {device}", log_file)

    # La normalizzazione con media e std pari a 0.5 porta i valori
    # da [0, 1] a [-1, 1].
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dir = os.path.join(dataset_root, "dataset_train")
    val_dir = os.path.join(dataset_root, "dataset_val")

    training_dataset = PairedHistologyDataset(
        train_dir,
        transform=transform
    )

    validation_dataset = PairedHistologyDataset(
        val_dir,
        transform=transform
    )
    
    first_train_pair_info = get_first_pair_size(training_dataset)
    first_val_pair_info = get_first_pair_size(validation_dataset)

    run_config = {
        "timestamp": timestamp_str,
        "dataset_root": str(dataset_root),
        "run_root": str(run_root),
        "logs_dir": str(logs_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "output_train_dir": str(output_train_dir),
        "output_val_dir": str(output_val_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "seed": seed,
        "device": str(device),
        "epochs": n_epochs,
        "batch_size": batch_size,
        "num_workers": n_workers,
        "image_size_resize": list(image_size),
        "log_rate": log_rate,
        "checkpoint_rate": checkpoint_rate,
        "validate_rate": validate_rate,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "l1_lambda": l1_lambda,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "beta1": beta1,
        "beta2": beta2,
        "train_samples": len(training_dataset),
        "val_samples": len(validation_dataset),
        "first_train_pair_info": first_train_pair_info,
        "first_val_pair_info": first_val_pair_info,
    }

    # In training facciamo shuffle; in validation no, perchè lì non stiamo
    # imparando ma solo misurando il comportamento del modello.
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda")
    )
    
    run_config["train_batches"] = len(training_loader)
    run_config["val_batches"] = len(validation_loader)

    config_path = save_run_config(run_config, run_root)
    log_run_header(log_file, run_config)
    log_message(f"Run config saved to {config_path}", log_file)

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    amp_enabled = is_amp_enabled(device)
    scaler_G = GradScaler(enabled=amp_enabled)
    scaler_D = GradScaler(enabled=amp_enabled)

    # BCEWithLogitsLoss lavora direttamente sui logits del discriminatore,
    # mentre L1Loss misura la distanza diretta tra output generato e target.
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Puliamo la cartella di output del training per non mischiare materiale di run diverse.
    for file in os.listdir(output_train_dir):
        file_path = os.path.join(output_train_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    start_epoch = 0
    if resume_checkpoint is not None:
        if os.path.exists(resume_checkpoint):
            start_epoch = load_checkpoint(
                resume_checkpoint,
                generator,
                discriminator,
                opt_G,
                opt_D,
                scaler_G,
                scaler_D,
                device
            )
            log_message(f"Checkpoint loaded from {resume_checkpoint}, epoch {start_epoch}", log_file)
        else:
            log_message(f"WARNING - Checkpoint not found: {resume_checkpoint}", log_file)
    else:
        log_message("Training started from scratch", log_file)

    progress_tracker = ProgressTracker(n_epochs, len(training_loader))
    progress_tracker.start()

    log_message("Training started", log_file)
    log_message(
        f"Hyperparameters | l1_lambda={l1_lambda} | lr_g={lr_g} | lr_d={lr_d} | "
        f"beta1={beta1} | beta2={beta2}",
        log_file
    )

    for epoch in range(start_epoch, n_epochs):
        log_message(f"Starting epoch {epoch}", log_file)

        train_one_epoch(
            generator,
            discriminator,
            training_loader,
            device,
            opt_G,
            opt_D,
            scaler_G,
            scaler_D,
            bce_loss,
            l1_loss,
            epoch,
            log_file,
            progress_tracker,
            log_rate,
            l1_lambda
        )

        log_message(f"Finished epoch {epoch}", log_file)

        # Salviamo checkpoint periodici: se il training si interrompe, non si riparte da zero.
        if (epoch + 1) % checkpoint_rate == 0:
            checkpoint_path = os.path.join(
                checkpoints_dir,
                f"ep{epoch:03d}.pth"
            )
            save_checkpoint(
                checkpoint_path,
                epoch,
                generator,
                discriminator,
                opt_G,
                opt_D,
                scaler_G,
                scaler_D,
                l1_lambda,
                lr_g,
                lr_d,
                beta1,
                beta2,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=n_workers,
                dataset_root=dataset_root
            )
            log_message(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}", log_file)

        # La validation periodica misura se il modello sta migliorando
        # anche fuori dal training set.
        if (epoch + 1) % validate_rate == 0:
            validate(
                generator,
                discriminator,
                validation_loader,
                device,
                bce_loss,
                l1_loss,
                epoch,
                log_file,
                output_val_dir,
                l1_lambda
            )

    total_seconds = time.time() - start_time
    log_message(f"Execution completed. Total time = {total_seconds:.2f} seconds", log_file)
              

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_root = Path(args.results_path) / args.run_name
        paths = build_workspace_paths(run_root)
        
        print(f"Starting training run '{args.run_name}' in: {run_root}")
        
        main(
            dataset_root=args.dataset_root,
            run_root=run_root,
            logs_dir=paths["logs_dir"],
            checkpoints_dir=paths["checkpoints_dir"],
            output_val_dir=paths["output_val_dir"],
            output_train_dir=paths["output_train_dir"],
            n_epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            n_workers=args.num_workers,
            image_size=tuple(args.image_size),
            log_rate=args.log_rate,
            checkpoint_rate=args.checkpoint_rate,
            validate_rate=args.validate_rate,
            resume_checkpoint=args.resume,
            l1_lambda=args.l1_lambda,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            beta1=args.beta1,
            beta2=args.beta2
        )

    elif args.mode == "test":
        test_dir = Path(args.dataset_root) / "dataset_test"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        run_root = Path(args.run_path)
        paths = build_workspace_paths(run_root)

        print(f"Starting test for run: {run_root}")
        print(f"Using checkpoint: {args.checkpoint}")
        test_inference(
            checkpoint_path=args.checkpoint,
            test_folder=str(test_dir),
            output_folder=str(paths["output_test_dir"]),
            image_size=tuple(args.image_size),
            device=device
        )
