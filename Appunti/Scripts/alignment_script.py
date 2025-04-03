# ==========================================================================
# Il file allinea una serie di immagini 
# ==========================================================================

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
    img_2_aligned = cv2.warpAffine(img_2, warp_matrix, (img_2.shape[1], img_2.shape[0]))
    img_2_mask_aligned = None
    if img_2_mask is not None:
        img_2_mask_aligned = cv2.warpAffine(img_2_mask, warp_matrix, (img_2.shape[1], img_2.shape[0]))

    return img_2_aligned, img_2_mask_aligned, warp_matrix

def main():
    # Controllo dell'esistenza della cartella Materiale/Locale/grid
    if not os.path.exists("Materiale/Locale/grid"):
        print("Eseguire prima il grid script")
        return
    # Carica le immagini presenti nella cartella
    images = {}
    for file in os.listdir("Materiale/Locale/grid"):
        # Formato nome: (mask_){i}_{j}_{label_free/stained}.tif
        if file.endswith(".tif"):
            image = cv2.imread(f"Materiale/Locale/grid/{file}")

            # Controllo se è una maschera
            mask = False
            if "mask" in file:
                mask = True
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                file = file.replace("mask_", "")
            
            # Estrazione delle coordinate
            i, j, type = file.replace(".tif", "").split("_", 2)
            i, j = int(i), int(j)
            if (i, j) not in images:
                images[(i, j)] = {}
            
            # Salvataggio dell'immagine
            if mask:
                type = f"mask_{type}"
            images[(i, j)][type] = image
    print(f"{len(images)} coppie di immagini caricate")

    # Controllo dell'esistenza della cartella Materiale/Locale/aligned
    if not os.path.exists("Materiale/Locale/aligned"):
        os.makedirs("Materiale/Locale/aligned")
        print("Cartella creata")
    # Cancella le immagini presenti nella cartella
    for file in os.listdir("Materiale/Locale/aligned"):
        os.remove(f"Materiale/Locale/aligned/{file}")
    print("Immagini rimosse")

    # Controllo dell'esistenza della cartella Materiale/Locale/bad_alignment
    if not os.path.exists("Materiale/Locale/bad_alignment"):
        os.makedirs("Materiale/Locale/bad_alignment")
        print("Cartella creata")
    
    margin = 200
    count = 0
    count_bad = 0
    # Allineamento delle immagini
    for (i, j), data in images.items():
        if "label_free" in data and "stained" in data:
            try:
                aligned, mask_aligned, warp_matrix = align(
                    data["label_free"], data["stained"],
                    img_1_mask=data["mask_label_free"], img_2_mask=data["mask_stained"])
            except ValueError:
                cv2.imwrite(f"Materiale/Locale/bad_alignment/MATCHES_{i:>05}_{j:>05}_label_free.tif", data["label_free"])
                cv2.imwrite(f"Materiale/Locale/bad_alignment/MATCHES_{i:>05}_{j:>05}_stained.tif", aligned)
                count_bad += 1
                print(f"Non ci sono abbastanza match per {i:>05}_{j:>05}")
                continue

            # Controllo della traslazione
            dx, dy = warp_matrix[:, 2]
            if dx > margin or dy > margin:
                cv2.imwrite(f"Materiale/Locale/bad_alignment/MARGIN_{i:>05}_{j:>05}_label_free.tif", data["label_free"])
                cv2.imwrite(f"Materiale/Locale/bad_alignment/MARGIN_{i:>05}_{j:>05}_stained.tif", aligned)
                count_bad += 1
                print(f"Traslazione troppo grande per {i:>05}_{j:>05}")
                continue
            
            # Immagini ritagliate senza margine
            label_free = data["label_free"][margin:-margin, margin:-margin]
            label_free_mask = data["mask_label_free"][margin:-margin, margin:-margin]
            aligned = aligned[margin:-margin, margin:-margin]
            mask_aligned = mask_aligned[margin:-margin, margin:-margin]

            # Salvataggio delle immagini
            cv2.imwrite(f"Materiale/Locale/aligned/{i:>05}_{j:>05}_label_free.tif", label_free)
            cv2.imwrite(f"Materiale/Locale/aligned/mask_{i:>05}_{j:>05}_label_free.tif", label_free_mask)
            cv2.imwrite(f"Materiale/Locale/aligned/{i:>05}_{j:>05}_stained.tif", aligned)
            cv2.imwrite(f"Materiale/Locale/aligned/mask_{i:>05}_{j:>05}_stained.tif", mask_aligned)
            count += 1
            print(f"Immagine allineata per {i:>05}_{j:>05} - {(count+count_bad)/len(images)*100:.2f}%")
        else:
            print(f"Manca un'immagine per {i:>05}_{j:>05}", data)
    print(f"{count} immagini allineate, {count_bad} immagini scartate")

if __name__ == "__main__":
    main()