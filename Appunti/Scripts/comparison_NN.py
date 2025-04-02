# ==================================================================
# Comparison of Neural Networks
# ==================================================================

import os
import re
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIGURAZIONE ===
input_folder = "Materiale/Locale/output_pix2pix"
output_folder = "Materiale/Locale/graphs"
num_img = "000"         # <-- cambia questo per visualizzare altre immagini
filtro_epoche = 5       # <-- epoche da mostrare: solo quelle per cui (epoca % filtro_epoche == 0)

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Regex per trovare i file relativi a una stessa immagine
pattern = re.compile(rf"(\d+)_({num_img})_(input|output|target)\.(png|jpg|jpeg)")

# Raccolta immagini organizzate per epoca
images_by_epoch = {}

for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        epoch, img_id, kind, ext = match.groups()
        if epoch not in images_by_epoch:
            images_by_epoch[epoch] = {}
        images_by_epoch[epoch][kind] = os.path.join(input_folder, filename)

# Ordina le epoche numericamente
sorted_epochs = sorted(images_by_epoch.keys(), key=lambda x: int(x))

# Applica il filtro
filtered_epochs = [e for e in sorted_epochs if int(e) % filtro_epoche == 0]

# Se il filtro esclude tutto, mostra almeno la prima epoca
if not filtered_epochs and sorted_epochs:
    filtered_epochs = [sorted_epochs[0]]

# Plotting
fig, axs = plt.subplots(len(filtered_epochs), 3, figsize=(12, 4 * len(filtered_epochs)))
fig.suptitle(f"Confronto immagini per numImg = {num_img}", fontsize=16)

if len(filtered_epochs) == 1:
    axs = [axs]  # compatibilità per una sola epoca

for i, epoch in enumerate(filtered_epochs):
    for j, kind in enumerate(["input", "output", "target"]):
        ax = axs[i][j]
        img_path = images_by_epoch[epoch].get(kind)
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title(kind.upper())
        axs[i][0].set_title(f"Epoca {epoch}", loc='left', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # lascia spazio per il titolo

# === SALVATAGGIO ===
output_path = os.path.join(output_folder, f"confronto_{num_img}_ogni{filtro_epoche}.png")
plt.savefig(output_path)
plt.show()

print(f"Grafico salvato in: {output_path}")
