import cv2
import matplotlib.pyplot as plt

# === PARAMETRI ===
path_img = "Appunti/Scripts/fullsize_label_free.tif"
crop_start = (10500, 8500)  # (x, y) inizio ritaglio
crop_size = (4096, 4096)   # (width, height) della porzione da visualizzare
patch_size = (256, 256)    # dimensione griglia
highlight_patch = (7, 4)   # (col, row) patch da evidenziare (facoltativo)
save_path = "immagine_con_griglia.png"

# === CARICAMENTO E CROP ===
img = cv2.imread(path_img)
if img is None:
    raise FileNotFoundError(f"Immagine non trovata a: {path_img}")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x_start, y_start = crop_start
w_crop, h_crop = crop_size
img_crop = img_gray[y_start:y_start + h_crop, x_start:x_start + w_crop]

# === DISEGNA GRIGLIA ===
img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
n_cols = w_crop // patch_size[0]
n_rows = h_crop // patch_size[1]

for i in range(1, n_cols):
    x = i * patch_size[0]
    cv2.line(img_rgb, (x, 2), (x, h_crop), (255, 0, 0), 1)

for j in range(1, n_rows):
    y = j * patch_size[1]
    cv2.line(img_rgb, (2, y), (w_crop, y), (255, 0, 0), 1)

# === EVIDENZIA UNA PATCH ===
if highlight_patch:
    col, row = highlight_patch
    x0 = col * patch_size[0]
    y0 = row * patch_size[1]
    x1 = x0 + patch_size[0]
    y1 = y0 + patch_size[1]
    cv2.rectangle(img_rgb, (x0, y0), (x1, y1), (0, 0, 255), 4)

# === VISUALIZZA O SALVA ===
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.title("Griglia su porzione Label-Free")
plt.axis('off')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
