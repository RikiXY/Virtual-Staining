# ==========================================================================
# Il file crea le maschere delle immagini di label-free e stained,
# allinea le immagini, crea una griglia, e le suddivide in training, validation e test.
# Ollie Wan Kenobi è il nome del file perché sembra di dire "All in one", Kenobi ci stava bene.
# (non abbiamo mai visto Star Wars)
# ==========================================================================

import cv2, os, random, sys
import numpy as np
from typing import Optional

def pad_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Espande l'immagine con un bordo bianco
    
    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    x : int
        Coordinata x del bordo
    y : int
        Coordinata y del bordo
    w : int
        Larghezza dell'immagine
    h : int
        Altezza dell'immagine
    
    Returns
    -------
    padded_image : np.ndarray
        Immagine espansa
    """
    top = y
    bottom = h - y - img.shape[0]
    left = x
    right = w - x - img.shape[1]
    padded_image = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=255)
    return padded_image

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

def calculate_mask(img: np.ndarray) -> np.ndarray:
    """
    Trova la maschera per i componenti connessi dell'immagine

    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    
    Returns
    -------
    mask : np.ndarray
        Maschera dell'immagine
    """
    # Binarizza l'immagine con una soglia
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)
    # Trova i componenti connessi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Ordina in modo decrescente i componenti per area
    n_filtered = 10
    sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    # Crea una maschera vuota per filtrare i componenti
    mask = np.zeros_like(binary).astype(np.uint8)

    # Per ogni componente in ordine decrescente di area
    for i in sorted_indices[:n_filtered]:
        x, y, w, h, area = stats[i]

        # Filtra i componenti troppo piccoli
        if w < 100 and h < 100:
            continue

        # Riempe la maschera
        component_mask = (labels == i).astype(np.uint8) * 255
        countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
        
        # Estrae dalla regione di interesse
        roi = img[y:y+h, x:x+w]
        roi_mask = component_mask[y:y+h, x:x+w]

        # Calcola la deviazione standard della regione di interesse
        std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]

        # Filtra i componenti con una deviazione standard troppo alta
        if std_dev < 10:
            mask[component_mask == 255] = 255
    
    # Inverte la maschera per ottenere il foreground
    # La maschera vale 255 per il foreground e 0 per lo sfondo
    mask = cv2.bitwise_not(mask)
    return mask

def calculate_mask_with_grid(img: np.ndarray, sub_shape: tuple[int, int], grid: int) -> np.ndarray:
    """
    Trova la maschera per i componenti connessi dell'immagine in una griglia

    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    sub_shape : tuple[int, int]
        Dimensioni della regione di interesse
    grid : int
        Numero di regioni per lato della griglia
    
    Returns
    -------
    mask : np.ndarray
        Maschera dell'immagine
    """
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255 # Maschera totale

    # Dividendo l'immagine in una griglia grid*grid, trova la maschera per ogni regione
    for y in range(0, img.shape[0], img.shape[0]//grid):
        for x in range(0, img.shape[1], img.shape[1]//grid):
            # Trova la maschera per la regione di interesse grande sub_shape
            roi = img[y:y+sub_shape[0], x:x+sub_shape[1]]
            roi_mask = calculate_mask(roi)
            # Espande la maschera per mantenere le dimensioni originali
            roi_mask = pad_image(roi_mask, x, y, img.shape[1], img.shape[0])
            # Aggiorna la maschera totale
            mask = cv2.bitwise_and(mask, roi_mask)
    return mask

def calculate_mask_with_mutliple_parameters(img: np.ndarray, divisors: list[int], grids: list[int]) -> np.ndarray:
    """
    Calcola la maschera per l'immagine di input
    
    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    divisors : list[int]
        Lista di divisori per l'immagine
    grids : list[int]
        Lista di griglie per l'immagine
    
    Returns
    -------
    mask : np.ndarray
        Maschera dell'immagine
    """
    # Unisce i divisori e le griglie in un'unica lista di tuple
    parameters = zip(divisors, grids)
    # Crea una maschera vuota
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    # Si usano diversi parametri sub_shape (2=metà del lato) e grid per trovare la maschera (3=3 quadri per lato)
    for divisor, grid in parameters:
        # Si trova la maschera per l'immagine
        sub_shape = (img.shape[0]//divisor, img.shape[1]//divisor)
        # Si trova la maschera con i parametri specificati
        _mask = calculate_mask_with_grid(img, sub_shape, grid)
        mask = cv2.bitwise_and(mask, _mask)
    return mask

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

def align_from_scaled(img1: np.ndarray, img2: np.ndarray, scale: int = 0.5, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, nfeatures: int = 10000, ed_distance: int = 200) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Allinea due immagini usando una matrice di omografia calcolata sulle immagini scalate

    Parameters
    ----------
    img1 : np.ndarray
        La prima immagine
    img2 : np.ndarray
        La seconda immagine
    scale : int, optional
        Fattore di scala per il calcolo della matrice omografica. 0.5 di base.
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
    # Scala le immagini
    img1_scaled = cv2.resize(img1, None, fx=scale, fy=scale)
    if mask1 is not None:
        mask1_scaled = cv2.resize(mask1, None, fx=scale, fy=scale)
    img2_scaled = cv2.resize(img2, None, fx=scale, fy=scale)
    if mask2 is not None:
        mask2_scaled = cv2.resize(mask2, None, fx=scale, fy=scale)
    
    # Allinea le immagini scalate con la funzione standard
    _, _, warp_matrix = align_images(img1_scaled, img2_scaled, mask1_scaled if mask1 is not None else None, mask2_scaled if mask2 is not None else None, nfeatures, ed_distance)

    # Adatta la matrice di omografia alla dimensione originale
    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale

    # Allinea l'immagine originale con la matrice di omografia calcolata
    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix

def divide_image_with_grid(img: np.ndarray, img_size: tuple[int, int], grid_movement: tuple[int, int], mask: Optional[np.ndarray] = None, max_mask_percentage = 0.4) -> list[np.ndarray]:
    """
    Divide l'immagine in una griglia di immagini di dimensioni img_size
    
    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    img_size : tuple[int, int]
        Dimensioni della regione di interesse
    grid_movement : tuple[int, int]
        Spostamento della griglia
    mask : np.ndarray, optional
        Maschera dell'immagine per il filtraggio. None di base.
    
    Returns
    -------
    images : list[np.ndarray]
        Lista delle immagini divise
    masks : list[np.ndarray], optional
        Lista delle maschere divise. None di base.
    positions : list[tuple[int, int]]
        Lista delle posizioni delle immagini divise
    """
    images = []
    masks = []
    positions = []

    # Estrazione delle immagini
    for x in range(0, img.shape[1], grid_movement[0]):
        for y in range(0, img.shape[0], grid_movement[1]):
            # Estrazione della regione di interesse dell'immagine
            roi_img = extract_image(img, x, y, img_size[0], img_size[1])
            # Se l'immagine è troppo piccola salto il quadrato
            if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
                continue
            # Se tutta la maschera è al 60% nera salto il quadrato in quanto è sfondo
            roi_mask = None
            if mask is not None:
                roi_mask = extract_image(mask, x, y, img_size[0], img_size[1])
                if cv2.countNonZero(roi_mask) < max_mask_percentage * roi_mask.size:
                    continue
            images.append(roi_img)
            if mask is not None:
                masks.append(roi_mask)
            positions.append((x, y))
    
    # Se la maschera è None, non la restituisco
    if mask is None:
        masks = None
    return images, masks, positions

def divide_image_with_positions(img: np.ndarray, img_size: tuple[int, int], positions: list[tuple[int, int]]) -> list[np.ndarray]:
    """
    Divide l'immagine in una griglia di immagini di dimensioni img_size usando le posizioni specificate
    
    Parameters
    ----------
    img : np.ndarray
        Immagine di input
    img_size : tuple[int, int]
        Dimensioni della regione di interesse
    positions : list[tuple[int, int]]
        Lista delle posizioni delle immagini divise
    
    Returns
    -------
    images : list[np.ndarray]
        Lista delle immagini divise
    """
    images = []
    for x, y in positions:
        # Estrazione della regione di interesse dell'immagine
        roi_img = extract_image(img, x, y, img_size[0], img_size[1])
        # Se l'immagine è troppo piccola salto il quadrato
        if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
            continue
        images.append(roi_img)
    
    return images

def split_input(input: list, ratios: list[int]) -> list[list]:
    """
    Suddivide l'input in N liste in base ai rapporti specificati
    
    Parameters
    ----------
    input : list
        Lista di input da suddividere
    ratios : list[int]
        Rapporti di suddivisione (es. [0.7, 0.15, 0.15])
    
    Returns
    -------
    split_input : list[list]
        Lista delle liste suddivise
    """
    if len(ratios) < 2:
        raise ValueError("Devi specificare almeno 2 rapporti")
    if sum(ratios) > 1:
        raise ValueError("La somma dei rapporti deve essere <= 1")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("I rapporti devono essere >= 0")
    
    # Mescolamento casuale dell'input
    shuffled = input.copy()
    random.shuffle(shuffled)

    output = []
    start = 0
    for ratio in ratios:
        end = start + int(len(shuffled) * ratio)
        output.append(shuffled[start:end])
        start = end
    return output

def main(path: str, save_masks: bool = False, seed: Optional[int] = None) -> None:
    # Imposta il seed per ottenere sempre la stessa suddivisione, se necessario
    if seed is not None:
        random.seed(seed)
        print(f"Seed impostato a: {seed}")

    # Caricamento delle immagini
    label_free = cv2.imread(os.path.join(path, "label_free.tif"))
    stained = cv2.imread(os.path.join(path, "stained.tif"))
    if label_free is None or stained is None:
        print("Impossibile caricare le immagini. Devono esssere chiamate label_free.tif e stained.tif")
        sys.exit(1)
    print(f"Immagini caricate: {label_free.shape}, {stained.shape}")

    # Calcolo delle maschere
    mask_lf = calculate_mask_with_mutliple_parameters(label_free, [2, 4, 6, 8], [3, 6, 9, 15])
    mask_st = calculate_mask_with_mutliple_parameters(stained, [2, 4, 6, 8], [3, 6, 9, 15])
    print("Maschere calcolate")

    # Salvataggio delle maschere
    cv2.imwrite(os.path.join(path, "mask_lf.tif"), mask_lf)
    cv2.imwrite(os.path.join(path, "mask_st.tif"), mask_st)
    print("Maschere salvate")

    # Allineameanto delle immagini
    aligned_stained, aligned_mask_st, warp_matrix = align_from_scaled(label_free, stained, mask1=mask_lf, mask2=mask_st, scale=0.5)
    print("Immagini allineate")

    # Salvataggio delle immagini allineate
    cv2.imwrite(os.path.join(path, "aligned_stained.tif"), aligned_stained)
    cv2.imwrite(os.path.join(path, "aligned_mask_st.tif"), aligned_mask_st)
    print("Immagini allineate salvate")

    # Estrazione delle sottoimmagini
    image_size = (512, 512)
    grid_movement = (300, 300)
    lf_images, lf_masks, positions = divide_image_with_grid(label_free, image_size, grid_movement, mask_lf)
    st_images = divide_image_with_positions(aligned_stained, image_size, positions)
    st_masks = divide_image_with_positions(aligned_mask_st, image_size, positions)
    print(f"Totale coppie estratte: {len(lf_images)}")

    # Combinazione delle immagini con il nome
    named_lf_images = []
    named_st_images = []
    named_lf_masks = []
    named_st_masks = []
    for (x, y), lf_img, lf_mask, st_img, st_mask in zip(positions, lf_images, lf_masks, st_images, st_masks):
        named_lf_images.append((lf_img, f"{x:>05}_{y:>05}_label_free"))
        named_st_images.append((st_img, f"{x:>05}_{y:>05}_stained"))
        named_lf_masks.append((lf_mask, f"{x:>05}_{y:>05}_mask_label_free"))
        named_st_masks.append((st_mask, f"{x:>05}_{y:>05}_mask_stained"))
    print("Coppie rinominate")

    # Salvataggio delle sottoimmagini
    os.makedirs(os.path.join(path, "subimages"), exist_ok=True)
    for lf_img, lf_mask, st_img, st_mask in zip(named_lf_images, named_lf_masks, named_st_images, named_st_masks):
        cv2.imwrite(os.path.join(path, "subimages", f"{lf_img[1]}.tif"), lf_img[0])
        cv2.imwrite(os.path.join(path, "subimages", f"{st_img[1]}.tif"), st_img[0])
        if save_masks:
            cv2.imwrite(os.path.join(path, "subimages", f"{lf_mask[1]}.tif"), lf_mask[0])
            cv2.imwrite(os.path.join(path, "subimages", f"{st_mask[1]}.tif"), st_mask[0])
    print("Coppie salvate")

    # Suddivisione del dataset in training, validation e testing
    images = list(zip(named_lf_images, named_st_images))
    split = split_input(images, [0.7, 0.15, 0.15])
    print(f"Coppie suddivise in: {len(split[0])} train, {len(split[1])} val, {len(split[2])} test")

    # Salvataggio delle immagini suddivise
    for i, subset in enumerate(split):
        subset_name = ["train", "val", "test"][i]
        os.makedirs(os.path.join(path, subset_name), exist_ok=True)
        for lf_img, st_img in subset:
            cv2.imwrite(os.path.join(path, subset_name, f"{lf_img[1]}.tif"), lf_img[0])
            cv2.imwrite(os.path.join(path, subset_name, f"{st_img[1]}.tif"), st_img[0])
    print("Dataset suddiviso e salvato")
        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ollie_wan_kenobi.py <path> [seed] [--save_masks]")
        print("Esempio: python ollie_wan_kenobi.py /Materiale/Locale/liver --save_masks")
        sys.exit(1)
    path = sys.argv[1]
    seed = None
    if len(sys.argv) > 2 and sys.argv[2] != "--save_masks":
        # Se il secondo argomento è un numero intero lo uso come seed
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print("Il seed deve essere un numero intero")
            sys.exit(1)
    save_masks = True if "--save_masks" in sys.argv else False
    main(path, save_masks, seed)