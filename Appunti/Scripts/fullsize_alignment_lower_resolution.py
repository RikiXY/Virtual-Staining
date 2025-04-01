import cv2
import numpy as np

def align(img_1, img_2, img_1_mask=None, img_2_mask=None, nfeatures=10000, ed_distance=200):
    """Allinea due immagini

    Args:
        img_1 (numpy.ndarray): La prima immagine
        img_2 (numpy.ndarray): La seconda immagine
        img_1_mask (numpy.ndarray, optional): La maschera della prima immagine. None di base.
        img_2_mask (numpy.ndarray, optional): La maschera della seconda immagine. None di base.
        nfeatures (int, optional): Numero di features per il calcolo di SIFT. 10000 di base.
        ed_distance (int, optional): Distanza per il filtro euclideo inclusiva. 200 di base.

    Returns:
        numpy.ndarray: L'immagine 2 allineata
        numpy.ndarray: La maschera dell'immagine 2 allineata
        numpy.ndarray: La matrice di trasformazione
    """
    # Applicazione CLAHE
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    img_1_clahe = img_1
    img_2_clahe = img_2
    # Se l'immagine è a colori la converto in scala di grigi
    if len(img_1_clahe.shape) == 3:
        img_1_clahe = cv2.cvtColor(img_1_clahe, cv2.COLOR_BGR2GRAY)
    if len(img_2_clahe.shape) == 3:
        img_2_clahe = cv2.cvtColor(img_2_clahe, cv2.COLOR_BGR2GRAY)
    img_1_clahe = clahe.apply(img_1_clahe)
    img_2_clahe = clahe.apply(img_2_clahe)

    # Calcolo delle features con SIFT
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_1_clahe, img_1_mask)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2_clahe, img_2_mask)

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
    img_2_aligned = cv2.warpAffine(img_2, warp_matrix, (img_1.shape[1], img_1.shape[0]))
    img_2_mask_aligned = None
    if img_2_mask is not None:
        img_2_mask_aligned = cv2.warpAffine(img_2_mask, warp_matrix, (img_1.shape[1], img_1.shape[0]))

    return img_2_aligned, img_2_mask_aligned, warp_matrix

def main():
    # Carica delle immagini
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    mask_lf = cv2.imread("Materiale/Immagini/mask_label_free.tif", cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread("Materiale/Locale/fullsize_stained.tif")
    mask_st = cv2.imread("Materiale/Immagini/mask_stained.tif", cv2.IMREAD_GRAYSCALE)
    if label_free is None or mask_lf is None or stained is None or mask_st is None:
        print("Errore nel caricamento delle immagini")
        return
    print("Immagini caricate")

    scale = 0.5
    # Ridimensionamento delle immagini
    scaled_label_free = cv2.resize(label_free, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_mask_lf = cv2.resize(mask_lf, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_stained = cv2.resize(stained, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_mask_st = cv2.resize(mask_st, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Allineamento della immagine ritagliata
    aligned, mask_aligned, warp_matrix = align(
        scaled_label_free, scaled_stained,
        img_1_mask=scaled_mask_lf, img_2_mask=scaled_mask_st)
    
    # Matrice di trasformazione
    dx, dy = warp_matrix[:, 2]
    print(f"Traslazione: dx={dx}, dy={dy}")
    print(warp_matrix)

    warp_matrix[0, 2] = warp_matrix[0, 2] / scale
    warp_matrix[1, 2] = warp_matrix[1, 2] / scale
    print(f"Traslazione: dx={warp_matrix[0, 2]}, dy={warp_matrix[1, 2]}")
    print(warp_matrix)

    # Allineamento dell'immagine e della maschera
    fullsize_aligned = cv2.warpAffine(stained, warp_matrix, (label_free.shape[1], label_free.shape[0]))
    fullsize_mask_aligned = cv2.warpAffine(mask_st, warp_matrix, (label_free.shape[1], label_free.shape[0]))

    # Salvataggio dell'immagine allineata
    cv2.imwrite("Materiale/Locale/aligned_stained.tif", fullsize_aligned)
    cv2.imwrite("Materiale/Immagini/aligned_mask_stained.tif", fullsize_mask_aligned)
    print("Immagine allineata salvata")

if __name__ == "__main__":
    main()