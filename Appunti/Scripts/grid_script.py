# ======================================================================
# Il file crea una griglia di immagini non allineate a partire dalle immagini label_free e stained
# =====================================================================

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
    if label_free is None or mask_lf is None or stained is None or mask_st is None:
        print("Errore nel caricamento delle immagini")
        return
    print("Immagini caricate")

    image_size = (512, 512) # prima era 1000x1000
    margin = 200
    image_size = (image_size[0]+margin*2, image_size[1]+margin*2)
    # Estrazione delle immagini da label_free
    positions = []
    grid_movement = (300, 300) # prima il passo era 500x500
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
            roi_image = extract_image(label_free, x, y, image_size[0], image_size[1])
            cv2.imwrite(f"Materiale/Locale/grid/{x:>05}_{y:>05}_label_free.tif", roi_image)
            cv2.imwrite(f"Materiale/Locale/grid/mask_{x:>05}_{y:>05}_label_free.tif", roi_mask)
            positions.append((x, y))
    print(f"{len(positions)} immagini estratte da label_free")

    # Estrazione delle immagini da stained
    for x, y in positions:
        # Estrazione della regione di interesse della maschera
        roi_mask = extract_image(mask_st, x, y, image_size[0], image_size[1])
        # Estrazione della regione di interesse dell'immagine
        roi_image = extract_image(stained, x, y, image_size[0], image_size[1])
        cv2.imwrite(f"Materiale/Locale/grid/{x:>05}_{y:>05}_stained.tif", roi_image)
        cv2.imwrite(f"Materiale/Locale/grid/mask_{x:>05}_{y:>05}_stained.tif", roi_mask)
    print(f"{len(positions)} immagini estratte da stained")

if __name__ == "__main__":
    main()