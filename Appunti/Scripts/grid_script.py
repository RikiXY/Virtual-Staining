import cv2, os

def extract_image(img, x, y, w, h):
    """
    Estrae una regione dell'immagine
    Arguments:
    img -- immagine di input
    x -- coordinata x dell'angolo in alto a sinistra
    y -- coordinata y dell'angolo in alto a sinistra
    w -- larghezza della regione
    h -- altezza della regione
    """
    return img[y:y+h, x:x+w]

def main():
    # Apertura delle immagini
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    mask_lf = cv2.imread("Materiale/Immagini/mask_label_free.tif", cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread("Materiale/Locale/fullsize_stained.tif")
    mask_st = cv2.imread("Materiale/Immagini/mask_stained.tif", cv2.IMREAD_GRAYSCALE)

    # Controllo dell'esistenza della cartella Materiale/Locale/grid
    if not os.path.exists("Materiale/Locale/grid"):
        os.makedirs("Materiale/Locale/grid")

    image_size = (1000, 1000)
    # Estrazione delle immagini da label_free
    indices = []
    for i in range(0, label_free.shape[1]//image_size[1]):
        for j in range(0, label_free.shape[0]//image_size[0]):
            roi_mask = extract_image(mask_lf, i*image_size[0], j*image_size[1], image_size[0], image_size[1])
            # Se tutta la maschera è nera salto il quadrato in quanto è sfondo
            if cv2.countNonZero(roi_mask) == 0:
                print(f"Skipping {i}_{j}")
                continue
            roi_image = extract_image(label_free, i*image_size[0], j*image_size[1], image_size[0], image_size[1])
            cv2.imwrite(f"Materiale/Locale/grid/{i}_{j}_label_free.tif", roi_image)
            cv2.imwrite(f"Materiale/Locale/grid/mask_{i}_{j}_label_free.tif", roi_mask)
            indices.append((i, j))
    # Estrazione delle immagini da stained
    for i, j in indices:
        roi_mask = extract_image(mask_st, i*image_size[0], j*image_size[1], image_size[0], image_size[1])
        # Se tutta la maschera è nera salto il quadrato in quanto è sfondo
        if cv2.countNonZero(roi_mask) == 0:
            print(f"Skipping {i}_{j}")
            continue
        roi_image = extract_image(stained, i*image_size[0], j*image_size[1], image_size[0], image_size[1])
        cv2.imwrite(f"Materiale/Locale/grid/{i}_{j}_stained.tif", roi_image)
        cv2.imwrite(f"Materiale/Locale/grid/mask_{i}_{j}_mask_stained.tif", roi_mask)

if __name__ == "__main__":
    main()