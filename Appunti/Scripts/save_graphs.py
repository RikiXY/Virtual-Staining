import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carica le immagini (assicurati che i percorsi e i nomi dei file siano corretti)
img_input =  mpimg.imread('Materiale/Locale/output_pix2pix/86_264_input.png')
img_output = mpimg.imread('Materiale/Locale/output_pix2pix/86_264_output.png')
img_target = mpimg.imread('Materiale/Locale/output_pix2pix/86_264_target.png')

# Crea la figura con tre subplot affiancati
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Primo subplot: Input
axes[0].imshow(img_input)
axes[0].set_title('Input')
axes[0].axis('off')  # Nasconde gli assi

# Secondo subplot: Output
axes[1].imshow(img_output)
axes[1].set_title('Output')
axes[1].axis('off')

# Terzo subplot: Target
axes[2].imshow(img_target)
axes[2].set_title('Target')
axes[2].axis('off')

# Disposizione più pulita e mostra la figura
plt.tight_layout()
plt.show()

# Se vuoi salvare la figura come file PNG, aggiungi:
# plt.savefig('Materiale/Locale/graphs/confronto_immagini.png', dpi=300)
