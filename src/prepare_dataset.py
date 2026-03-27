"""
Integrated preprocessing pipeline for paired histopathology samples.

This script loads two paired full-size images (a source image and a target
image), computes tissue masks, aligns the target image to the source reference,
extracts paired patches, and creates the `dataset_train`, `dataset_val`, and
`dataset_test` splits.
"""

# The original filename was `ollie_wan_kenobi`, born from an internal joke:
# we needed an "all-in-one" script, which gradually turned into "Ollie Wan",
# and from there "Kenobi" felt like the only possible ending.
# Ironically, none of the collaborators had even seen Star Wars.

from os import path

import cv2, os, random
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

# [ITA] Importo il file messages.json che contiene i messaggi in italiano e inglese
# [EN] Import the messages.json file that contains messages in Italian and English
script_dir = Path(__file__).resolve()
messages_path = script_dir.parent / "json" / "messages.json"
with messages_path.open("r", encoding="utf-8") as m:
    MESSAGES = json.load(m)

def pad_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Expands the image with a white border.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    x : int
        X coordinate of the border.
    y : int
        Y coordinate of the border.
    w : int
        Width of the output image.
    h : int
        Height of the output image.

    Returns
    -------
    padded_image : np.ndarray
        Expanded (padded) image.
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

def calculate_mask(img: np.ndarray) -> np.ndarray:
    """
    Finds the mask for the connected components in the image.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    mask : np.ndarray
        Image mask.
    """
    
    # [ITA] Binarizza l'immagine con una soglia
    # [EN] Binarizes the image with a threshold
    # ----- DA FAR VEDERE AD ANDREA ----- 
    # Vogliamo rendere la soglia un parametro? Sì
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)

    # [ITA] Trova i componenti connessi
    # [EN] Finds the connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # [ITA] Ordina i componenti per area in modo decrescente per area
    # [EN] Sorts the components by area in descending order
    n_filtered = 10
    sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    # [ITA] Crea una maschera vuota per filtrare i componenti
    # [EN] Creates an empty mask to filter the components
    mask = np.zeros_like(binary).astype(np.uint8)

    # [ITA] Per ogni componente in ordine decrescente di area
    # [EN] For each component in descending order of area
    for i in sorted_indices[:n_filtered]:
        x, y, w, h, area = stats[i]

        # [ITA] Se il componente è troppo piccolo lo salto
        # [EN] If the component is too small, skip it
        if w < 100 and h < 100:
            continue
        
        # [ITA] Riempe la maschera 
        # [EN] Fills the mask
        component_mask = (labels == i).astype(np.uint8) * 255
        countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
        
        # [ITA] Estrae dalla regione di interesse
        # [EN] Extracts from the region of interest
        roi = img[y:y+h, x:x+w]
        roi_mask = component_mask[y:y+h, x:x+w]

        # [ITA] Calcola la deviazione standard della regione di interesse
        # [EN] Calculates the standard deviation of the region of interest
        std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]

        # [ITA] Filtra i componenti con una deviazione standard troppo alta
        # [EN] Filters components with too high standard deviation
        if std_dev < 10:
            mask[component_mask == 255] = 255
    
    # [ITA] Inverte la maschera per ottenere il foreground. La maschera vale 255 per il foreground e 0 per lo sfondo
    # [EN] Inverts the mask to get the foreground. The mask is 255 for the foreground and 0 for the background
    mask = cv2.bitwise_not(mask)
    return mask

def calculate_mask_with_grid(img: np.ndarray, sub_shape: tuple[int, int], grid: int) -> np.ndarray:
    """
    Finds the mask for the connected components of the image using a grid.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    sub_shape : tuple[int, int]
        Size of the region of interest.
    grid : int
        Number of regions per side of the grid.

    Returns
    -------
    mask : np.ndarray
        Mask of the image.
    """
    
    # [ITA] Maschera totale
    # [EN] Total mask
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255 

    # [ITA] Dividendo l'immagine in una griglia grid*grid, trova la maschera per ogni regione
    # [EN] Dividing the image into a grid of grid*grid, finds the mask for each region
    for y in range(0, img.shape[0], img.shape[0]//grid):
        for x in range(0, img.shape[1], img.shape[1]//grid):
            
            # [ITA] Trova la maschera per la regione di interesse grande sub_shape
            # [EN] Find the mask for the region of interest of size sub_shape
            roi = img[y:y+sub_shape[0], x:x+sub_shape[1]]
            roi_mask = calculate_mask(roi)
            
            # [ITA] Espande la maschera per mantenere le dimensioni originali
            # [EN] Pads the mask to keep the original size
            roi_mask = pad_image(roi_mask, x, y, img.shape[1], img.shape[0])
            
            # [ITA] Aggiorna la maschera totale
            # [EN] Updates the total mask
            mask = cv2.bitwise_and(mask, roi_mask)
    return mask

def calculate_mask_with_multiple_parameters(img: np.ndarray, parameters: list[tuple[int, int]]) -> np.ndarray:
    """
    Calculates the mask for the input image using multiple parameter pairs.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    parameters : list[tuple[int, int]]
        List of (divisor, grid) pairs used to calculate the masks.

    Returns
    -------
    mask : np.ndarray
        Mask of the image.
    """
    
    # [ITA] Crea una maschera vuota
    # [EN] Create an empty mask
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    
    # [ITA] Si usano diversi parametri: sub_shape (2=metà del lato) e grid per trovare la maschera (3=3 quadri per lato)
    # [EN] We use different parameters: sub_shape (2=half side) and grid to find the mask (3=3 squares per side)
    for divisor, grid in parameters:
        
        # [ITA] Si trova la maschera per l'immagine
        # [EN] We find the mask for the image
        sub_shape = (img.shape[0]//divisor, img.shape[1]//divisor)
        
        # [ITA] Si trova la maschera con i parametri specificati
        # [EN] We find the mask with the specified parameters
        _mask = calculate_mask_with_grid(img, sub_shape, grid)

        # [ITA] Si aggiorna la maschera totale
        # [EN] We update the total mask
        mask = cv2.bitwise_and(mask, _mask)
        
    return mask

def align_images(img1: np.ndarray, img2: np.ndarray, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, nfeatures: int = 10000, ed_distance: int = 200) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Aligns a moving image to a reference image.

    Parameters
    ----------
    img1 : np.ndarray
        Reference image.
    img2 : np.ndarray
        Image to align to the reference.
    mask1 : np.ndarray, optional
        The mask for the first image. Default is None.
    mask2 : np.ndarray, optional
        The mask for the second image. Default is None.
    nfeatures : int, optional
        Number of features for SIFT computation. Default is 10000.
    ed_distance : int, optional
        Inclusive Euclidean distance threshold for filtering matches. Default is 200.

    Returns
    -------
    img2_aligned : np.ndarray
        The aligned second image.
    mask2_aligned : np.ndarray, optional
        The aligned mask for the second image.
    warp_matrix : np.ndarray
        The transformation matrix.
    """
    
    # [ITA] Applicazione CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # [EN] Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
    img1_clahe = img1
    img2_clahe = img2

    # [ITA] Se le immagini sono a colori, le converto in scala di grigi
    # [EN] If the images are in color, convert them to grayscale
    if len(img1_clahe.shape) == 3:
        img1_clahe = cv2.cvtColor(img1_clahe, cv2.COLOR_BGR2GRAY)
    if len(img2_clahe.shape) == 3:
        img2_clahe = cv2.cvtColor(img2_clahe, cv2.COLOR_BGR2GRAY)

    # [ITA] Applico CLAHE alle immagini
    # [EN] Apply CLAHE to the images
    img1_clahe = clahe.apply(img1_clahe)
    img2_clahe = clahe.apply(img2_clahe)

    # [ITA] Calcolo delle features con SIFT (Scale-Invariant Feature Transform)
    # [EN] Calculate features with SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_clahe, mask1)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_clahe, mask2)

    # [ITA] Controllo se ci sono abbastanza features
    # [EN] Check if there are enough features
    if len(keypoints_1) < 4 or len(keypoints_2) < 4:
        raise ValueError(MESSAGES["not_enough_features"][lang])

    # [ITA] Matching delle features con BFMatcher (Brute Force Matcher)
    # [EN] Feature matching with BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    # [ITA] Filtraggio dei match in base alla distanza euclidea
    # [EN] Filtering matches based on Euclidean distance
    filtered_matches = []
    for match in matches:
        distance = np.linalg.norm(np.array(keypoints_1[match.queryIdx].pt) - np.array(keypoints_2[match.trainIdx].pt))
        if distance <= ed_distance:
            filtered_matches.append(match)
    
    # [ITA] Controllo se ci sono abbastanza match
    # [EN] Check if there are enough matches
    if len(filtered_matches) < 4:
        raise ValueError(MESSAGES["not_enough_matches"][lang])
    
    # [ITA] Estrazione dei punti chiave dai match filtrati
    # [EN] Extracting keypoints from filtered matches
    points_1 = np.float32([keypoints_1[match.queryIdx].pt for match in filtered_matches]).reshape(-1, 1, 2)
    points_2 = np.float32([keypoints_2[match.trainIdx].pt for match in filtered_matches]).reshape(-1, 1, 2)

    # [ITA] Calcolo della matrice di trasformazione
    # [EN] Calculating the transformation matrix
    warp_matrix, mask = cv2.estimateAffinePartial2D(points_2, points_1)

    # [ITA] Allineamento dell'immagine e della maschera
    # [EN] Aligning the image and the mask
    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix

def align_from_scaled(img1: np.ndarray, img2: np.ndarray, scale: float = 0.5, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, nfeatures: int = 10000, ed_distance: int = 200) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    # ----- DA FAR VEDERE AD ANDREA -----
    # vogliamo mettere il fattore di scala come parametro? Sì
    # vogliamo mettere la soglia di distanza euclidea come parametro? Sì
    # vogliamo mettere il numero di features come parametro? Sì
    """
    Aligns two images by first scaling them, estimating the transformation on the scaled images, and then applying the transformation to the original images.
    
    Parameters
    ----------
    img1 : np.ndarray
        Source image used as reference.
    img2 : np.ndarray
        Target image to be aligned to the source image.
    scale : float, optional
        Scaling factor to resize images before alignment (default is 0.5).
    mask1 : Optional[np.ndarray], optional
        Optional mask for the first image.
    mask2 : Optional[np.ndarray], optional
        Optional mask for the second image.
    nfeatures : int, optional
        Number of features to use for alignment (default is 10000).
    ed_distance : int, optional
        Euclidean distance threshold for feature matching (default is 200).

    Returns
    -------
    img2_aligned : np.ndarray
        The second image aligned to the first.
    mask2_aligned : Optional[np.ndarray]
        The aligned mask for the second image, if provided.
    warp_matrix : np.ndarray
        The affine transformation matrix used for alignment.
    """
    
    # [ITA] Scala le immagini per l'allineamento con un fattore di scala di 0.5 (ovvero dimezza le dimensioni)
    # [EN] Scale the images for alignment with a scale factor of 0.5 (i.e., halve the dimensions)
    img1_scaled = cv2.resize(img1, None, fx=scale, fy=scale)
    img2_scaled = cv2.resize(img2, None, fx=scale, fy=scale)

    if mask1 is None or mask2 is None:
        print(MESSAGES["scaled_mask_error"][lang])
        raise ValueError(MESSAGES["scaled_mask_error"][lang])
    else:
        mask1_scaled = cv2.resize(mask1, None, fx=scale, fy=scale)
        mask2_scaled = cv2.resize(mask2, None, fx=scale, fy=scale)
    
    # [ITA] Allinea le immagini scalate con la funzione standard
    # [EN] Aligns the scaled images using the standard function
    _, _, warp_matrix = align_images(img1_scaled, img2_scaled, mask1_scaled if mask1 is not None else None, mask2_scaled if mask2 is not None else None, nfeatures, ed_distance)

    # [ITA] Adatta la matrice di omografia alla dimensione originale
    # [EN] Adjusts the homography matrix to the original size
    warp_matrix[0, 2] /= scale
    warp_matrix[1, 2] /= scale

    # [ITA] Allinea l'immagine originale con la matrice di omografia calcolata
    # [EN] Aligns the original image with the calculated homography matrix
    img2_aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))
    mask2_aligned = None
    if mask2 is not None:
        mask2_aligned = cv2.warpAffine(mask2, warp_matrix, (img1.shape[1], img1.shape[0]))

    return img2_aligned, mask2_aligned, warp_matrix

def extract_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Extracts a region from the image.

    Parameters
    ----------
    img : np.ndarray
        Input image
    x : int
        x-coordinate of the top-left corner
    y : int
        y-coordinate of the top-left corner
    w : int
        Width of the region
    h : int
        Height of the region

    Returns
    -------
    roi : np.ndarray
        Region of the image
    """
    
    return img[y:y+h, x:x+w]

def divide_image_with_grid(img: np.ndarray, img_size: tuple[int, int], grid_movement: tuple[int, int], mask: Optional[np.ndarray] = None, max_mask_percentage = 0.4) -> tuple[list[np.ndarray], Optional[list[np.ndarray]], list[tuple[int, int]]]:
    """
    Divides the input image into a grid of sub-images of size `img_size`.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    img_size : tuple[int, int]
        Size of the region of interest (width, height).
    grid_movement : tuple[int, int]
        Step size for moving the grid (x, y).
    mask : np.ndarray, optional
        Image mask for filtering. Default is None.
    max_mask_percentage : float, optional
        Maximum allowed percentage of masked (non-zero) pixels for a region to be included. Default is 0.4.
    
    Returns
    -------
    images : list[np.ndarray]
        List of extracted sub-images.
    masks : list[np.ndarray] or None
        List of extracted mask regions, or None if no mask is provided.
    positions : list[tuple[int, int]]
        List of positions (x, y) of the top-left corner of each extracted sub-image.
    """
    
    images = []
    masks = []
    positions = []

    # [ITA] Estrazione delle immagini 
    # [EN] Extracting images
    for x in range(0, img.shape[1], grid_movement[0]):
        for y in range(0, img.shape[0], grid_movement[1]):

            # [ITA] Estrazione della regione di interesse dell'immagine
            # [EN] Extracting the region of interest from the image
            roi_img = extract_image(img, x, y, img_size[0], img_size[1])

            # [ITA] Se l'immagine è troppo piccola salto la salto
            # [EN] If the image is too small, skip it
            if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
                continue

            # [ITA] Se la maschera è troppo nera, salto l'immagine in quanto è per la maggior parte sfondo
            # [EN] If the mask is too black, skip the image as it is mostly background
            roi_mask = None
            if mask is not None:
                roi_mask = extract_image(mask, x, y, img_size[0], img_size[1])
                if cv2.countNonZero(roi_mask) < max_mask_percentage * roi_mask.size:
                    continue
            images.append(roi_img)
            if mask is not None:
                masks.append(roi_mask)
            positions.append((x, y))
    
    # [ITA] Se la maschera è None, non la restituisco
    # [EN] If the mask is None, do not return it
    if mask is None:
        masks = None
    return images, masks, positions

def divide_image_with_positions(img: np.ndarray, img_size: tuple[int, int], positions: list[tuple[int, int]]) -> list[np.ndarray]:
    """
    Splits the image into a grid of images of size img_size using the specified positions.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    img_size : tuple[int, int]
        Size of the region of interest.
    positions : list[tuple[int, int]]
        List of positions for the split images.

    Returns
    -------
    images : list[np.ndarray]
        List of the split images.
    """
    
    images = []
    for x, y in positions:
        # [ITA] Estrazione della regione di interesse dell'immagine
        # [EN] Extracting the region of interest from the image
        roi_img = extract_image(img, x, y, img_size[0], img_size[1])

        # [ITA] Se l'immagine è troppo piccola salto il quadrato
        # [EN] If the image is too small, skip the square
        if roi_img.shape[0] < img_size[1] or roi_img.shape[1] < img_size[0]:
            continue
        images.append(roi_img)
    
    return images

def split_items(items: list, ratios: list[int]) -> list[list]:
    """
    Splits the input list into N sublists according to the specified ratios.
    
    Parameters
    ----------
    items : list
        List to split.
    ratios : list[int]
        Split ratios (e.g. [0.7, 0.15, 0.15]).
    
    Returns
    -------
    output : list[list]
        List of generated sublists.
    """
    if len(ratios) < 2:
        raise ValueError("Devi specificare almeno 2 rapporti")
    if sum(ratios) > 1:
        raise ValueError("La somma dei rapporti deve essere <= 1")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("I rapporti devono essere >= 0")
    
    # Mescolamento casuale della lista
    shuffled = items.copy()
    random.shuffle(shuffled)

    output = []
    start = 0
    for ratio in ratios:
        end = start + int(len(shuffled) * ratio)
        output.append(shuffled[start:end])
        start = end
    return output

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png"}

def validate_image_filename(filename: str, role: str) -> Path:
    file_path = Path(filename)
    suffix = file_path.suffix.lower()

    if not file_path.name:
        raise ValueError(f"{role} filename is empty.")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"{role} must use one of these extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}. "
            f"Received: {filename}"
        )
    return file_path

def main(
    path: str,
    source_name: str,
    target_name: str,
    seed: Optional[int] = None,
    save_masks: bool = False,
    image_size: tuple[int, int] = (256, 256),
    grid_movement: tuple[int, int] = (256, 256),
    margin: int = 200,
) -> None:

    # ====================[SET SEED]====================
    # [ITA] Imposta il seed per ottenere sempre la stessa suddivisione, se necessario
    # [EN] Set seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    print(MESSAGES["seed_set"][lang].format(seed=seed))
    # ==================================================


    # ====================[LOAD IMAGES]====================
    # [ITA]  Caricamento delle immagini
    # [EN] Loading images
    print(MESSAGES["loading_images"][lang].format(path=path))
    if not os.path.exists(path):
        raise FileNotFoundError(MESSAGES["check_path"][lang].format(path=path)) 
    source_file = validate_image_filename(source_name, "Source")
    target_file = validate_image_filename(target_name, "Target")
    
    source_stem = source_file.stem
    target_stem = target_file.stem
    source_suffix = source_file.suffix.lower()
    target_suffix = target_file.suffix.lower()

    source_image = cv2.imread(os.path.join(path, source_file.name))
    target_image = cv2.imread(os.path.join(path, target_file.name))
    if source_image is None or target_image is None:
        raise FileNotFoundError(
            f"Missing paired files. Expected '{source_name}' and '{target_name}' inside: {path}"
        )
    print(
        MESSAGES["images_loaded"][lang].format(
            source_image_shape=source_image.shape,
            target_image_shape=target_image.shape,
        )
    )
    # =====================================================


    # ====================[MASK PROCESSING]====================
    # [ITA] Calcolo delle maschere per le immagini source e target
    # [EN] Calculate masks for source and target images
    print(MESSAGES["calculate_masks"][lang])
    source_mask = calculate_mask_with_multiple_parameters(source_image, [(2, 3), (4, 6), (6, 9), (8, 15)])
    target_mask = calculate_mask_with_multiple_parameters(target_image, [(2, 3), (4, 6), (6, 9), (8, 15)])
    print(MESSAGES["masks_calculated"][lang])
    # [ITA] Salvataggio delle maschere
    # [EN] Saving masks
    cv2.imwrite(os.path.join(path, f"mask_{source_stem}{source_suffix}"), source_mask)
    cv2.imwrite(os.path.join(path, f"mask_{target_stem}{target_suffix}"), target_mask)
    print(MESSAGES["mask_saved"][lang])
    # =========================================================


    # ====================[IMAGE ALIGNMENT]==================== 
    # [ITA] Allineamento dell'immagine target sull'immagine source
    # [EN] Aligning the target image to the source image
    print(MESSAGES["aligning_images"][lang])
    aligned_target, aligned_target_mask, warp_matrix = align_from_scaled(
        source_image,
        target_image,
        mask1=source_mask,
        mask2=target_mask,
        scale=0.5,
    )
    print(MESSAGES["images_aligned"][lang])
    # [ITA] Salvataggio delle immagini allineate
    # [EN] Saving aligned images
    cv2.imwrite(os.path.join(path, f"aligned_{target_stem}{target_suffix}"), aligned_target)
    cv2.imwrite(os.path.join(path, f"aligned_mask_{target_stem}{target_suffix}"), aligned_target_mask)
    print(MESSAGES["images_aligned_saved"][lang])
    # =========================================================


    # ====================[DATASET CREATION]==================== 
    # [ITA] Estrazione delle sottoimmagini
    # [EN] Extracting sub-images
    print(MESSAGES["extracting_subimages"][lang])
    # ----- DA FAR VEDERE AD ANDREA -----
    # Vogliamo mettere il margine come parametro? Sì
    # Vogliamo mettere le dimensioni delle sottoimmagini come parametro? Sì
    # Vogliamo mettere lo spostamento della griglia come parametro? Sì
    
    source_images, source_masks, positions = divide_image_with_grid(
        source_image[margin:-margin, margin:-margin],
        image_size,
        grid_movement,
        source_mask[margin:-margin, margin:-margin],
    )
    target_images = divide_image_with_positions(
        aligned_target[margin:-margin, margin:-margin],
        image_size,
        positions,
    )
    target_masks = divide_image_with_positions(
        aligned_target_mask[margin:-margin, margin:-margin],
        image_size,
        positions,
    )
    print(MESSAGES["total_subimages"][lang].format(count=len(source_images)))
    # [ITA] Combinazione delle immagini con le maschere
    # [EN] Combining images with masks
    named_source_images = []
    named_target_images = []
    named_source_masks = []
    named_target_masks = []
    for (x, y), source_img, source_patch_mask, target_img, target_patch_mask in zip(
        positions,
        source_images,
        source_masks,
        target_images,
        target_masks,
    ):
        named_source_images.append((source_img, f"{x:05}_{y:05}_source{source_suffix}"))
        named_target_images.append((target_img, f"{x:05}_{y:05}_target{target_suffix}"))
        named_source_masks.append((source_patch_mask, f"{x:05}_{y:05}_mask_source{source_suffix}"))
        named_target_masks.append((target_patch_mask, f"{x:05}_{y:05}_mask_target{target_suffix}"))
    print(MESSAGES["pair_renamed"][lang])
    # [ITA] Salvataggio delle sottoimmagini
    # [EN] Saving sub-images
    print(MESSAGES["pair_saving"][lang])
    os.makedirs(os.path.join(path, "subimages"), exist_ok=True)
    for source_img, source_patch_mask, target_img, target_patch_mask in zip(
        named_source_images,
        named_source_masks,
        named_target_images,
        named_target_masks,
    ):
        cv2.imwrite(os.path.join(path, "subimages", source_img[1]), source_img[0])
        cv2.imwrite(os.path.join(path, "subimages", target_img[1]), target_img[0])
        if save_masks:
            cv2.imwrite(os.path.join(path, "subimages", source_patch_mask[1]), source_patch_mask[0])
            cv2.imwrite(os.path.join(path, "subimages", target_patch_mask[1]), target_patch_mask[0])
    print(MESSAGES["pair_saved"][lang])
    # [ITA] Suddivisione del dataset in training, validation e testing
    # [EN] Splitting the dataset into training, validation, and testing
    print(MESSAGES["dataset_subdivision"][lang])
    images = list(zip(named_source_images, named_target_images))
    split = split_items(images, [0.8, 0.05, 0.15])
    print(MESSAGES["pair_number_division"][lang].format(train=len(split[0]), val=len(split[1]), test=len(split[2])))
    # [ITA] Salvataggio delle immagini suddivise
    # [EN] Saving the split images
    for i, subset in enumerate(split):
        subset_name = ["dataset_train", "dataset_val", "dataset_test"][i]
        os.makedirs(os.path.join(path, subset_name), exist_ok=True)
        for source_img, target_img in subset:
            cv2.imwrite(os.path.join(path, subset_name, source_img[1]), source_img[0])
            cv2.imwrite(os.path.join(path, subset_name, target_img[1]), target_img[0])
    print(MESSAGES["dataset_saved"][lang])
    # ==========================================================

if __name__ == "__main__":
    # ====================[ARGUMENT PARSING]====================
    # [ITA] Parsing degli argomenti della riga di comando. Il primo parser serve per la lingua, il secondo per gli argomenti principali
    # [EN] Parsing command line arguments. The first parser is for the language, the second for the main arguments
    lang_parser = argparse.ArgumentParser(add_help=False)
    lang_parser.add_argument(
        "--lang",
        type=str,
        choices=["it", "en"],
        default="en",
        help="Language for messages (default: en)"
    )
    lang_args, _ = lang_parser.parse_known_args()
    # [ITA] Caricamento dei messaggi in base alla lingua scelta
    # [EN] Loading messages based on the chosen language
    help_path = script_dir.parent / "json" / "help.json"
    with help_path.open("r", encoding="utf-8") as h:
        HELP = json.load(h)
    # [ITA] Parser principale per gli argomenti
    # [EN] Main parser for the arguments
    parser = argparse.ArgumentParser(
    usage=(
        "python src/prepare_dataset.py <path> "
        "--source-name SOURCE_NAME --target-name TARGET_NAME\n"
        "       [--seed SEED] [--save-masks] [--image-size WIDTH HEIGHT]\n"
        "       [--grid-movement STEP_X STEP_Y] [--margin MARGIN] [--lang {en,it}]"
    ),
        description=HELP["description"][lang_args.lang],
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[lang_parser]
    )
    parser.add_argument(
        "path",
        type=str,
        help=HELP["path"][lang_args.lang]
    )
    parser.add_argument(
        "--source-name",
        type=str,
        required=True,
        help="Source image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--target-name",
        type=str,
        required=True,
        help="Target image filename with extension (.tif, .tiff, .png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=HELP["seed"][lang_args.lang]
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help=HELP["save_masks"][lang_args.lang]
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(512, 512),
        help="Patch size used for extraction (default: 512 512)"
    )
    parser.add_argument(
        "--grid-movement",
        type=int,
        nargs=2,
        metavar=("STEP_X", "STEP_Y"),
        default=(512, 512),
        help="Grid step used for patch extraction (default: 512 512)"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=200,
        help="Margin cropped from each border before patch extraction (default: 200)"
    )
    args = parser.parse_args()
    lang = args.lang
    # ==========================================================


    # [ITA] Esecuzione della funzione principale con gli argomenti specificati
    # [EN] Running the main function with the specified arguments
    main(
        path=args.path,
        source_name=args.source_name,
        target_name=args.target_name,
        seed=args.seed,
        save_masks=args.save_masks,
        image_size=tuple(args.image_size),
        grid_movement=tuple(args.grid_movement),
        margin=args.margin,
    )

