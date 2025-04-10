# =============================================================
# Il file crea una griglia di immagini allineate a partire dalle immagini label_free e aligned_stained
# =============================================================

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
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    mask_lf = cv2.imread("Materiale/Immagini/mask_label_free.tif", cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread("Materiale/Locale/aligned_stained.tif")

    if label_free is None or mask_lf is None or stained is None:
        print("Errore nel caricamento delle immagini")
        return
    print("Immagini caricate")

    # Ritaglio del bordo
    margin = 200
    label_free = label_free[margin:-margin, margin:-margin]
    mask_lf = mask_lf[margin:-margin, margin:-margin]
    stained = stained[margin:-margin, margin:-margin]

    # Controllo dell'esistenza della cartella Materiale/Locale/aligned
    if not os.path.exists("Materiale/Locale/aligned"):
        os.makedirs("Materiale/Locale/aligned")
        print("Cartella creata")
    # Cancella le immagini presenti nella cartella
    for file in os.listdir("Materiale/Locale/aligned"):
        os.remove(f"Materiale/Locale/aligned/{file}")
    print("Immagini rimosse")

    image_size = (512, 512)
    # Estrazione delle immagini
    grid_movement = (300, 300)
    count = 0
    for x in range(0, label_free.shape[1], grid_movement[0]):
        for y in range(0, label_free.shape[0], grid_movement[1]):
            # Estrazione della regione di interesse della maschera
            roi_mask = extract_image(mask_lf, x, y, image_size[0], image_size[1])
            # Se l'immagine è troppo piccola salto il quadrato
            if roi_mask.shape[0] < image_size[1] or roi_mask.shape[1] < image_size[0]:
                print(f"Salto {x:>05}_{y:>05} per dimensioni insufficienti")
                continue
            # Se tutta la maschera è al 60% nera salto il quadrato in quanto è sfondo
            if cv2.countNonZero(roi_mask) < 0.4 * roi_mask.size:
                print(f"Salto {x:>05}_{y:>05} per maschera insufficiente")
                continue
            # Estrazione della regione di interesse dell'immagine
            roi_lf = extract_image(label_free, x, y, image_size[0], image_size[1])
            roi_st = extract_image(stained, x, y, image_size[0], image_size[1])
            cv2.imwrite(f"Materiale/Locale/aligned/{x:>05}_{y:>05}_label_free.tif", roi_lf)
            cv2.imwrite(f"Materiale/Locale/aligned/{x:>05}_{y:>05}_stained.tif", roi_st)
            count += 1
    print(f"{count} coppie di immagini estratte")

if __name__ == "__main__":
    main()