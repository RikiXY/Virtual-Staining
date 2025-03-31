import cv2, os
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
    # Creazione di GpuMat per le immagini e le maschere
    g_img1 = cv2.cuda_GpuMat()
    g_img2 = cv2.cuda_GpuMat()
    g_img1_mask = cv2.cuda_GpuMat()
    g_img2_mask = cv2.cuda_GpuMat()

    # Caricamento delle immagini e delle maschere nella GPU
    g_img1.upload(img_1)
    g_img2.upload(img_2)
    if img_1_mask is not None:
        g_img1_mask.upload(img_1_mask)
    if img_2_mask is not None:
        g_img2_mask.upload(img_2_mask)

    # Applicazione CLAHE
    clahe = cv2.cuda.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    g_img1_clahe = g_img1
    g_img2_clahe = g_img2
    # Se l'immagine è a colori la converto in scala di grigi
    if len(g_img1_clahe.shape) == 3:
        g_img1_clahe = cv2.cuda.cvtColor(g_img1_clahe, cv2.COLOR_BGR2GRAY)
    if len(g_img2_clahe.shape) == 3:
        g_img2_clahe = cv2.cuda.cvtColor(g_img2_clahe, cv2.COLOR_BGR2GRAY)
    g_img1_clahe = clahe.apply(g_img1_clahe)
    g_img2_clahe = clahe.apply(g_img2_clahe)

    # Calcolo delle features con SIFT
    sift = cv2.cuda.SIFT_create(nfeatures=nfeatures)
    keypoints_1, descriptors_1 = sift.detectAndCompute(g_img1_clahe, img_1_mask)
    keypoints_2, descriptors_2 = sift.detectAndCompute(g_img2_clahe, img_2_mask)

    # Controllo se ci sono abbastanza features
    if len(keypoints_1) < 4 or len(keypoints_2) < 4:
        raise ValueError("Non ci sono abbastanza features")

    # Matching delle features
    bf = cv2.cuda.BFMatcher(cv2.NORM_L2, crossCheck=True)
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
    g_img2_aligned = cv2.cuda.warpAffine(g_img2, warp_matrix, (img_2.shape[1], img_2.shape[0]))
    img_2_aligned = g_img2_aligned.download()
    img_2_mask_aligned = None
    if g_img2_mask is not None:
        g_img2_mask_aligned = cv2.cuda.warpAffine(g_img2_mask, warp_matrix, (img_2.shape[1], img_2.shape[0]))
        img_2_mask_aligned = g_img2_mask_aligned.download()


    return img_2_aligned, img_2_mask_aligned, warp_matrix

def main():
    # Controllo della disponibilità di una GPU CUDA
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise EnvironmentError("Nessuna GPU CUDA trovata o OpenCV non compilato con supporto CUDA.")

    # Carica delle immagini
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    mask_lf = cv2.imread("Materiale/Immagini/mask_label_free.tif", cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread("Materiale/Locale/fullsize_stained.tif")
    mask_st = cv2.imread("Materiale/Immagini/mask_stained.tif", cv2.IMREAD_GRAYSCALE)
    if label_free is None or mask_lf is None or stained is None or mask_st is None:
        print("Errore nel caricamento delle immagini")
        return
    print("Immagini caricate")
    
    # Allineamento della immagine ritagliata
    aligned, mask_aligned, warp_matrix = align(
        label_free, stained,
        img_1_mask=mask_lf, img_2_mask=mask_st)
    
    # Matrice di trasformazione
    dx, dy = warp_matrix[:, 2]
    print(f"Traslazione: dx={dx}, dy={dy}")
    print(warp_matrix)

    # Salvataggio dell'immagine allineata
    cv2.imwrite("Materiale/Locale/fullsize_stained_aligned.tif", aligned)
    cv2.imwrite("Materiale/Locale/fullsize_stained_mask_aligned.tif", mask_aligned)
    print("Immagine allineata salvata")

            
if __name__ == "__main__":
    main()