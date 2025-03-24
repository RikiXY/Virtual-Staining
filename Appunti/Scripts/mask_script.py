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

def split_in_4(img):
    h, w = img.shape[:2]
    return img[:h//2, :w//2], img[:h//2, w//2:], img[h//2:, :w//2], img[h//2:, w//2:]

def join_4(img1, img2, img3, img4):
    return np.vstack((np.hstack((img1, img2)), np.hstack((img3, img4))))

def mask(img):
    # Binarizza l'immagine con una soglia
    _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)
    # Trova i componenti connessi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"Number of components: {num_labels}")

    # Ordina in modo decrescente i componenti per area
    n_filtered = 10
    sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    # Crea una maschera vuota per filtrare i componenti
    mask = np.zeros_like(binary)

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
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

def split_and_mask(img, n=1):
    n -= 1
    if n == 0:
        return mask(img)
    images = split_in_4(img)
    masked_images = [split_and_mask(img, n) for img in images]
    return join_4(*masked_images)

def main():
    label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    #label_free = split_in_4(label_free)[1]
    #cv2.imwrite("Materiale/Locale/cut_images/alto_sx_1.png", label_free)
    masked = split_and_mask(label_free, 3)
    show_images([[(label_free, "Immagine originale"), (masked, "Immagine filtrata")]])

if __name__ == "__main__":
    main()