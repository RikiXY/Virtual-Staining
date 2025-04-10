# =============================================================
# Il file crea una griglia di immagini allineate a partire dalle immagini label_free e aligned_stained
# =============================================================

import cv2, os, sys
import numpy as np

def extract_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Estrae una regione dell'immagine
    
    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    x : int
        Coordinata x dell'angolo in alto a sinistra
    y : int
        Coordinata y dell'angolo in alto a sinistra
    w : int
        Larghezza della regione
    h : int
        Altezza della regione

    Returns
    -------
    roi : np.ndarray
        Regione dell'immagine
    """
    return img[y:y+h, x:x+w]

def main(path: str, save_masks: bool = False) -> None:
    margin = 200
    # Caricamento delle immagini senza margine
    print(f"Caricamento delle immagini da {path}")
    label_free = cv2.imread(os.path.join(path, "label_free.tif"))[margin:-margin, margin:-margin]
    mask_lf = cv2.imread(os.path.join(path, "mask_lf.tif"), cv2.IMREAD_GRAYSCALE)[margin:-margin, margin:-margin]
    stained = cv2.imread(os.path.join(path, "aligned_stained.tif"))[margin:-margin, margin:-margin]
    mask_st = cv2.imread(os.path.join(path, "aligned_mask_st.tif"), cv2.IMREAD_GRAYSCALE)[margin:-margin, margin:-margin]
    if label_free is None or mask_lf is None or stained is None or mask_st is None:
        print("Impossibile caricare le immagini. Devono essere chiamate label_free.tif, mask_lf.tif, aligned_stained.tif, aligned_mask_st.tif")
        sys.exit(1)
    if label_free.shape != stained.shape or label_free.shape != mask_lf.shape or label_free.shape != mask_st.shape:
        print("Le immagini non hanno la stessa dimensione")
        sys.exit(1)
    print(f"Immagini caricate")

    # Controllo dell'esistenza della cartella {path}/subimages
    os.makedirs(f"{path}/subimages", exist_ok=True)
    # Cancella le immagini presenti nella cartella
    for file in os.listdir(f"{path}/subimages"):
        os.remove(os.path.join(f"{path}/subimages", file))
    print("Immagini rimosse")
    
    img_size = (512, 512)
    # Estrazione delle sottoimmagini
    print("Estrazione delle sottoimmagini")
    grid_movement = (300, 300)
    max_mask_percentage = 0.4
    count = 0
    for x in range(0, label_free.shape[1], grid_movement[0]):
        for y in range(0, label_free.shape[0], grid_movement[1]):
            # Estrazione della regione di interesse dell'immagine
            mlf_roi = extract_image(mask_lf, x, y, img_size[0], img_size[1])
            # Se l'immagine è troppo piccola salto la salto
            if mlf_roi.shape[0] < img_size[1] or mlf_roi.shape[1] < img_size[0]:
                continue
            # Se la maschera è al troppo nera salto l'immagine in quanto è per la maggior parte sfondo
            if cv2.countNonZero(mlf_roi) < max_mask_percentage * mlf_roi.size:
                continue
            lf_roi = extract_image(label_free, x, y, img_size[0], img_size[1])
            st_roi = extract_image(stained, x, y, img_size[0], img_size[1])
            mst_roi = extract_image(mask_st, x, y, img_size[0], img_size[1])
            # Salvataggio delle immagini
            cv2.imwrite(f"{path}/subimages/{x:>05}_{y:>05}_label_free.tif", lf_roi)
            cv2.imwrite(f"{path}/subimages/{x:>05}_{y:>05}_stained.tif", st_roi)
            if save_masks:
                cv2.imwrite(f"{path}/subimages/{x:>05}_{y:>05}_mask_lf.tif", mlf_roi)
                cv2.imwrite(f"{path}/subimages/{x:>05}_{y:>05}_mask_st.tif", mst_roi)
            count += 1
    print(f"{count} coppie di immagini estratte")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python create_grid.py <path> [--save_masks]")
        print("Esempio: python create_grid.py /Materiale/Locale/liver --save_masks")
        sys.exit(1)
    path = sys.argv[1]
    save_masks = True if "--save_masks" in sys.argv else False
    main(path, save_masks)