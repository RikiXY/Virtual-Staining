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
    # Controllo dell'esistenza della cartella Materiale/Locale/grid
    if not os.path.exists("Materiale/Locale/grid"):
        os.makedirs("Materiale/Locale/grid")
        print("Cartella creata")
    # Cancella le immagini presenti nella cartella
    for file in os.listdir("Materiale/Locale/grid"):
        os.remove(f"Materiale/Locale/grid/{file}")
    print("Immagini rimosse")

    # Apertura delle immagini
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    mask_lf = cv2.imread("Materiale/Immagini/mask_label_free.tif", cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread("Materiale/Locale/fullsize_stained.tif")
    mask_st = cv2.imread("Materiale/Immagini/mask_stained.tif", cv2.IMREAD_GRAYSCALE)
    print("Immagini caricate")

    image_size = (1000, 1000)
    margin = 200
    # Estrazione delle immagini da label_free
    indices = []
    for x in range(0, label_free.shape[1], image_size[0]):
        for y in range(0, label_free.shape[0], image_size[1]):
            i = x // image_size[0]
            j = y // image_size[1]
            # Estrazione della regione di interesse della maschera
            roi_mask = extract_image(mask_lf, i*image_size[0], j*image_size[1], image_size[0]+margin*2, image_size[1]+margin*2)
            # Se l'immagine è troppo piccola salto il quadrato
            if roi_mask.shape[0] < image_size[1]+margin*2 or roi_mask.shape[1] < image_size[0]+margin*2:
                print(f"Salto {i}_{j} per dimensioni insufficienti")
                continue
            # Se tutta la maschera è al 60% nera salto il quadrato in quanto è sfondo
            if cv2.countNonZero(roi_mask) < 0.4 * roi_mask.size:
                print(f"Salto {i}_{j} per maschera insufficiente")
                continue
            # Estrazione della regione di interesse dell'immagine
            roi_image = extract_image(label_free, i*image_size[0], j*image_size[1], image_size[0]+margin*2, image_size[1]+margin*2)
            cv2.imwrite(f"Materiale/Locale/grid/{i:>02}_{j:>02}_label_free.tif", roi_image)
            cv2.imwrite(f"Materiale/Locale/grid/mask_{i:>02}_{j:>02}_label_free.tif", roi_mask)
            indices.append((i, j))
    print(f"{len(indices)} immagini estratte da label_free")

    # Estrazione delle immagini da stained
    for i, j in indices:
        # Estrazione della regione di interesse della maschera
        roi_mask = extract_image(mask_st, i*image_size[0], j*image_size[1], image_size[0]+margin*2, image_size[1]+margin*2)
        # Estrazione della regione di interesse dell'immagine
        roi_image = extract_image(stained, i*image_size[0], j*image_size[1], image_size[0]+margin*2, image_size[1]+margin*2)
        cv2.imwrite(f"Materiale/Locale/grid/{i:>02}_{j:>02}_stained.tif", roi_image)
        cv2.imwrite(f"Materiale/Locale/grid/mask_{i:>02}_{j:>02}_stained.tif", roi_mask)
    print(f"{len(indices)} immagini estratte da stained")

if __name__ == "__main__":
    main()