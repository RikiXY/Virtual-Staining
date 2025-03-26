import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(img, title=None, cmap=None):
    cmap = cmap or ('gray' if len(img.shape) == 2 else None)
    img = img
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    figure, axes = plt.subplots(figsize=(20, 20))
    axes.imshow(img, cmap=cmap)
    axes.set_title(title)
    axes.axis('off')
    plt.show()

def show_images(images, cmap=None):
    # Se l'immagine è un array monodimensionale, la trasformo in un array bidimensionale
    if isinstance(images[0], tuple):
        images = [images]
    figure, subplots = plt.subplots(len(images), len(images[0]), figsize=(20, 20))
    figure.tight_layout()
    for row in range(len(images)):
        for column in range(len(images[row])):
            # Se l'immagine è un array di tre elementi, allora il terzo elemento è il cmap
            if len(images[row][column]) == 3:
                img, title, _cmap = images[row][column]
            else:
                img, title = images[row][column]
                _cmap = cmap or ('gray' if len(img.shape) == 2 else None)

            # Se l'immagine è a colori, la converto in RGB per poterla visualizzare correttamente
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Se subplots è un array monodimensionale, allora non c'è bisogno di usare la notazione a due dimensioni
            if len(images) == 1:
                subplots[column].imshow(img, cmap=_cmap)
                subplots[column].set_title(title)
                subplots[column].axis('off')
            else:
                subplots[row, column].imshow(img, cmap=_cmap)
                subplots[row, column].set_title(title)
                subplots[row, column].axis('off')
    plt.show()

def find_mask(img):
    """Trova la maschera per i componenti connessi dell'immagine

    Args:
        img (MatLike): Immagine di input

    Returns: Maschera
    """
    # Binarizza l'immagine con una soglia
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)
    # Trova i componenti connessi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"Number of components: {num_labels}")

    # Ordina in modo decrescente i componenti per area
    n_filtered = 10
    sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    # Crea una maschera vuota per filtrare i componenti
    mask = np.zeros_like(binary).astype(np.uint8)

    images = []
    # Per ogni componente in ordine decrescente di area
    for i in sorted_indices[:n_filtered]:
        x, y, w, h, area = stats[i]

        # Riempe la maschera
        component_mask = (labels == i).astype(np.uint8) * 255
        countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
        
        # Estrae dalla regione di interesse
        roi = img[y:y+h, x:x+w]
        roi_mask = component_mask[y:y+h, x:x+w]

        # Calcola la deviazione standard della regione di interesse
        std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]
        images.append([(roi, f"Componente {i}"), (roi_mask, f"Maschera σ {std_dev:.2f}")])

        # Filtra i componenti con una deviazione standard troppo alta
        if std_dev < 10:
            mask[labels == i] = 255
    
    # Applica la maschera all'immagine
    mask = cv2.bitwise_not(mask)
    return mask

def pad_image(img, x, y, w, h):
    """
    Espande l'immagine con un bordo bianco per mantenere le dimensioni originali
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

def find_mask_with_grid(img, sub_shape, grid):
    """
    Trova la maschera per ogni regione dell'immagine divisa in una griglia
    Arguments:
    img -- immagine di input
    sub_shape -- dimensione delle sotto-immagini
    grid -- numero di suddivisioni della griglia
    """
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255 # Maschera totale

    # Dividendo l'immagine in una griglia grid*grid, trova la maschera per ogni regione
    for y in range(0, img.shape[0], img.shape[0]//grid):
        for x in range(0, img.shape[1], img.shape[1]//grid):
            # Trova la maschera per la regione di interesse grande sub_shape
            roi = img[y:y+sub_shape[0], x:x+sub_shape[1]]
            roi_mask = find_mask(roi)
            # Espande la maschera per mantenere le dimensioni originali
            roi_mask = pad_image(roi_mask, x, y, img.shape[1], img.shape[0])
            # Aggiorna la maschera totale
            mask = cv2.bitwise_and(mask, roi_mask)
    return mask

def main():
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    stained = cv2.imread("Materiale/Locale/fullsize_stained.tif")

    mask_lf = np.ones((label_free.shape[0], label_free.shape[1]), dtype=np.uint8) * 255
    mask_st = np.ones((stained.shape[0], stained.shape[1]), dtype=np.uint8) * 255
    # Si usano diversi parametri sub_shape (2=metà del lato) e grid per trovare la maschera (3=3 quadri per lato)
    for divisor, grid in [(2, 3), (4, 6), (6, 9), (8, 15)]:
        print(f"Divisor: {divisor}, Grid: {grid}")
        # Si trova la maschera per l'immagine label_free
        sub_shape = (label_free.shape[0]//divisor, label_free.shape[1]//divisor)
        # Si trova la maschera con i parametri specificati
        _mask = find_mask_with_grid(label_free, sub_shape, grid)
        mask_lf = cv2.bitwise_and(mask_lf, _mask)

        # Si trova la maschera per l'immagine stained
        sub_shape = (stained.shape[0]//divisor, stained.shape[1]//divisor)
        # Si trova la maschera con i parametri specificati
        _mask = find_mask_with_grid(stained, sub_shape, grid)
        mask_st = cv2.bitwise_and(mask_st, _mask)

    # Si applica la maschera alle immagini
    masked_lf = cv2.bitwise_and(label_free, label_free, mask=mask_lf)
    masked_st = cv2.bitwise_and(stained, stained, mask=mask_st)

    # Si salvano le immagini
    cv2.imwrite("Materiale/Immagini/mask_label_free.tif", mask_lf)
    cv2.imwrite("Materiale/Immagini/mask_stained.tif", mask_st)

    # Si abbassa la risoluzione delle immagini per visualizzarle
    label_free = cv2.resize(label_free, (label_free.shape[1]//4, label_free.shape[0]//4))
    masked_lf = cv2.resize(masked_lf, (masked_lf.shape[1]//4, masked_lf.shape[0]//4))
    stained = cv2.resize(stained, (stained.shape[1]//4, stained.shape[0]//4))
    masked_st = cv2.resize(masked_st, (masked_st.shape[1]//4, masked_st.shape[0]//4))
    show_images([
        [(label_free, "Immagine originale"), (masked_lf, "Immagine filtrata")],
        [(stained, "Immagine originale"), (masked_st, "Immagine filtrata")]])

if __name__ == "__main__":
    main()