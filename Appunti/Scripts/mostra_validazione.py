import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Lista dei percorsi delle immagini (in ordine da sinistra a destra, riga per riga)
image_paths = [
    "Appunti/Scripts/img/e0.tif", "Appunti/Scripts/img/e1.tif", "Appunti/Scripts/img/e3.tif",
    "Appunti/Scripts/img/e4.tif", "Appunti/Scripts/img/e149.tif", "Appunti/Scripts/img/output.tif"
]

# Titolo generale del plot
titoli = [
    "Epoca 0", "Epoca 1", "Epoca 3",
    "Epoca 4", "Epoca 149", "Immagine Reale"
]

# Crea figura
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Inserisce le immagini
for i, path in enumerate(image_paths):
    row = i // 3
    col = i % 3
    img = mpimg.imread(path)
    axs[row, col].imshow(img)
    axs[row, col].axis('off')
    axs[row, col].set_title(titoli[i], fontsize=10, fontweight='bold')

    # Aggiunge una cornice attorno all'immagine
    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=axs[row, col].transAxes,
        linewidth=1.5, edgecolor='black', facecolor='none'
    )
    axs[row, col].add_patch(rect)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # lascia spazio per il titolo
plt.savefig("Appunti/Scripts/validazione_completa.png", dpi=300, bbox_inches='tight')
plt.show()
