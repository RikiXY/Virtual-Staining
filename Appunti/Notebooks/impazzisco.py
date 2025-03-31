import os
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# -------------------------------------------------------
# Dataset personalizzato
# -------------------------------------------------------
class PairedHistologyDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.pairs = self._get_pairs()

    def _get_pairs(self):
        """Cerca tutti i file che finiscono con '_label_free.tif'
           e costruisce la lista dei prefissi."""
        all_files = os.listdir(self.folder_path)
        prefixes = []
        for f in all_files:
            if f.endswith("_label_free.tif"):
                prefix = f.replace("_label_free.tif", "")
                # Controllo che esista anche '_stained.tif'
                stained_file = prefix + "_stained.tif"
                if stained_file in all_files:
                    prefixes.append(prefix)
        return sorted(prefixes)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prefix = self.pairs[idx]
        lf_path = os.path.join(self.folder_path, prefix + "_label_free.tif")
        st_path = os.path.join(self.folder_path, prefix + "_stained.tif")

        # Carico entrambe le immagini
        lf_img = Image.open(lf_path).convert("RGB")
        st_img = Image.open(st_path).convert("RGB")

        # Applico le eventuali trasformazioni
        if self.transform:
            lf_img = self.transform(lf_img)
            st_img = self.transform(st_img)

        return lf_img, st_img

# -------------------------------------------------------
# Trasformazioni
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),   # semplifica
    transforms.ToTensor(),           # converte in tensor [C,H,W] in [0,1]
])

# -------------------------------------------------------
# Semplice modello di test
# -------------------------------------------------------
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

def main():
    # -------------------------------------------------------
    # Creazione dataset e dataloader
    # -------------------------------------------------------
    train_folder = "Materiale/Locale/dataset_split/train"  # <-- METTI il tuo path
    train_dataset = PairedHistologyDataset(train_folder, transform=transform)

    # Imposta batch_size e num_workers=0 per evitare errori su Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,   # 0 => nessun worker parallelo (profiling più leggibile)
        pin_memory=True if torch.cuda.is_available() else False
    )

    # -------------------------------------------------------
    # Inizializzazione
    # -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in uso:", device)

    model = SimpleConvNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------------------------------------
    # Training loop + mini-profiler con time.time()
    # -------------------------------------------------------
    print("Inizio training di prova...\n")
    model.train()

    num_epochs = 2  # due epoche di test
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_epoch = time.time()

        for i, (input_img, target_img) in enumerate(train_loader):
            # Misura tempo di caricamento + transfer su GPU
            t0 = time.time()
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)
            t1 = time.time()

            # Forward + backward pass
            output = model(input_img)
            loss = criterion(output, target_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t2 = time.time()

            running_loss += loss.item()

            # Stampa ogni 10 batch i tempi
            if i % 10 == 0:
                print(f"[Ep {epoch+1:02d} | Batch {i:03d}] "
                    f"Load+Transfer: {(t1 - t0)*1e3:.1f} ms, "
                    f"Fwd+Bwd: {(t2 - t1)*1e3:.1f} ms, "
                    f"Loss: {loss.item():.4f}")

        end_epoch = time.time()
        epoch_time = end_epoch - start_epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss media: {avg_loss:.4f}, "
            f"Tempo epoch: {epoch_time:.2f} s\n")

    print("Training completato! ✅")

if __name__ == "__main__":
    main()
