# =============================================================
# Il file allinea l'immagine stained sulla label_free attraverso la maschera omografica
# calcolata sull'immagine fullsize scalata
# =============================================================

import cv2, os, sys
import numpy as np
from typing import Optional

def align_images(img1: np.ndarray, img2: np.ndarray, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, nfeatures: int = 10000, ed_distance: int = 200) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Allinea due immagini

    Parameters
    ----------
    img1 : np.ndarray
        La prima immagine
    img2 : np.ndarray
        La seconda immagine
    mask1 : np.ndarray, optional
        La maschera della prima immagine. None di base.
    mask2 : np.ndarray, optional
        La maschera della seconda immagine. None di base.
    nfeatures : int, optional
        Numero di features per il calcolo di SIFT. 10000 di base.
    ed_distance : int, optional
        Distanza per il filtro euclideo inclusiva. 200 di base.
    
    Returns
    -------
    img2_aligned : np.ndarray
        L'immagine 2 allineata
    mask2_aligned : np.ndarray, optional
        La maschera dell'immagine 2 allineata
    warp_matrix : np.ndarray
        La matrice di trasformazione
    """
    # Applicazione CLAHE
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    img1_clahe = img1
    img2_clahe = img2
    # Se l'immagine è a colori la converto in scala di grigi
    if len(img1_clahe.shape) == 3:
        img1_clahe = cv2.cvtColor(img1_clahe, cv2.COLOR_BGR2GRAY)
    if len(img2_clahe.shape) == 3:
        img2_clahe = cv2.cvtColor(img2_clahe, cv2.COLOR_BGR2GRAY)
    img1_clahe = clahe.apply(img1_clahe)
    img2_clahe = clahe.apply(img2_clahe)

    # Calcolo delle features con SIFT
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_clahe, mask1)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_clahe, mask2)

    # Controllo se ci sono abbastanza features
    if len(keypoints_1) < 4 or len(keypoints_2) < 4:
        raise ValueError("Non ci sono abbastanza features")

    # Matching delle features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    # Filtraggio dei match con distanza euclidea
    filtered_matches = []
    for match in matches:
        distance = np.linalg.norm(np.array(keypoints_1[match.queryIdx].pt) - np.array(keypoints_2[match.trainIdx].pt))
        if distance <= ed_distance:
            filtered_matches.append(match)
    
    # Controllo se ci sono abbastanza match
    if len(filtered_matches) < 4:
        raise ValueError("Non ci sono abbastanza match")
    
    # Estrazione dei punti
    points_1 = np.float32([keypoints_1[match.queryIdx].pt for match in filtered_matches]).reshape(-1, 1, 2)
    points_2 = np.float32([keypoints_2[match.trainIdx].pt for match in filtered_matches]).reshape(-1, 1, 2)

    # Calcolo della matrice di trasformazione
    warp_matrix, mask = cv2.estimateAffinePartial2D(points_2, points_1)

    # Allineamento dell'immagine e della maschera
    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix

def main(path: str):
    # Caricamento delle immagini
    print(f"Caricamento delle immagini da {path}")
    label_free = cv2.imread(os.path.join(path, "label_free.tif"))
    mask_lf = cv2.imread(os.path.join(path, "mask_lf.tif"), cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread(os.path.join(path, "stained.tif"))
    mask_st = cv2.imread(os.path.join(path, "mask_st.tif"), cv2.IMREAD_GRAYSCALE)
    if label_free is None or mask_lf is None or stained is None or mask_st is None:
        print("Impossibile caricare le immagini. Devono essere chiamate label_free.tif, mask_lf.tif, stained.tif, mask_st.tif")
        sys.exit(1)
    if label_free.shape != mask_lf.shape or stained.shape != mask_st.shape:
        print("Le immagini e corrispettive maschere non hanno la stessa dimensione")
        sys.exit(1)
    print(f"Immagini caricate")

    print("Calcolo della matrice omografica")
    scale = 0.5
    # Ridimensionamento delle immagini
    lf_scaled = cv2.resize(label_free, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mlf_scaled = cv2.resize(mask_lf, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    lf_scaled = cv2.resize(stained, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mlf_scaled = cv2.resize(mask_st, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Allineamento della immagine ritagliata
    _, _, warp_matrix = align_images(lf_scaled, lf_scaled, mlf_scaled, mlf_scaled)

    # Riscalamento della matrice omografica
    warp_matrix[0, 2] = warp_matrix[0, 2] / scale
    warp_matrix[1, 2] = warp_matrix[1, 2] / scale
    print(f"Matrice calcolata:\n{warp_matrix}")

    # Allineamento dell'immagine e della maschera
    print("Allineamento delle immagini")
    aligned_stained = cv2.warpAffine(stained, warp_matrix, (label_free.shape[1], label_free.shape[0]))
    aligned_mask_st = cv2.warpAffine(mask_st, warp_matrix, (label_free.shape[1], label_free.shape[0]))
    print("Immagini allineate")

    # Salvataggio dell'immagine allineata
    cv2.imwrite(f"{path}/aligned_stained.tif", aligned_stained)
    cv2.imwrite(f"{path}/aligned_mask_st.tif", aligned_mask_st)
    print("Immagini allineate salvate")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ollie_wan_kenobi.py <path> [seed] [--save_masks]")
        print("Esempio: python ollie_wan_kenobi.py /Materiale/Locale/liver --save_masks")
        sys.exit(1)
    path = sys.argv[1]
    main(path)