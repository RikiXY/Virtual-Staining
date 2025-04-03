# ==================================================================
# Comparison of Neural Networks
# ==================================================================
import os
import re
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIGURAZIONE ===
input_folder = "Materiale/Locale/output_val"
output_folder = "Materiale/Locale/graphs"
filtro_epoche = 1  # <-- epoche da mostrare: solo quelle per cui (epoca % filtro_epoche == 0)
num_batch = 1  # <-- batch da mostrare: solo quelli per cui (batch == num_batch)

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Nuovo pattern: cerca "epoch<numero>_batch<numero>_tipo.(png|jpg|jpeg)"
# Esempio: "epoch10_batch3_output.png"
pattern = re.compile(rf"epoch(\d+)_batch({num_batch})_(input|output|target)\.(png|jpg|jpeg)", re.IGNORECASE)

# Dizionario per immagazzinare le immagini indicizzate da (epoch, batch)
images_by_epoch_batch = {}

for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        epoch_str, batch_str, kind, _ = match.groups()
        # Inizializza la struttura se non presente
        if (epoch_str, batch_str) not in images_by_epoch_batch:
            images_by_epoch_batch[(epoch_str, batch_str)] = {}
        # Salva il percorso del file in base a input / output / target
        images_by_epoch_batch[(epoch_str, batch_str)][kind] = os.path.join(input_folder, filename)

# Ordina (epoch, batch) prima per epoca (intero) e poi per batch (intero)
sorted_epoch_batches = sorted(images_by_epoch_batch.keys(),
                              key=lambda x: (int(x[0]), int(x[1])))

# Applica il filtro sulle epoche (ignorando il batch, ma lasciandolo poi in fase di visualizzazione)
filtered_epoch_batches = [
    (ep, ba) for (ep, ba) in sorted_epoch_batches if int(ep) % filtro_epoche == 0
]

# Se il filtro esclude tutto, mostra almeno la prima epoca (se presente)
if not filtered_epoch_batches and sorted_epoch_batches:
    filtered_epoch_batches = [sorted_epoch_batches[0]]

# Plotting
fig, axs = plt.subplots(len(filtered_epoch_batches), 3, figsize=(12, 4 * len(filtered_epoch_batches)))
fig.suptitle("Confronto immagini (senza num_img fisso)", fontsize=16)

# Se c'è solo un blocco (una riga) axs è singolo array, uniformiamo la logica
if len(filtered_epoch_batches) == 1:
    axs = [axs]

for i, (epoch_str, batch_str) in enumerate(filtered_epoch_batches):
    # Recupero dizionario con i path delle immagini (input/output/target)
    img_dict = images_by_epoch_batch[(epoch_str, batch_str)]
    for j, kind in enumerate(["input", "output", "target"]):
        ax = axs[i][j]
        img_path = img_dict.get(kind)
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title(kind.upper())
    # Titolo a sinistra, utile per distinguere epoca e batch
    axs[i][0].set_title(f"Epoca {epoch_str}, batch {batch_str}", loc='left', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # lascia spazio per il titolo

# === SALVATAGGIO ===
output_path = os.path.join(output_folder, f"confronto_epoch_batch_ogni{filtro_epoche}.png")
plt.savefig(output_path)
# plt.show()

print(f"Grafico salvato in: {output_path}")
