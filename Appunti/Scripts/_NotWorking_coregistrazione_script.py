import cv2
import numpy as np
import matplotlib.pyplot as plt

label_free_path = "../../Materiale/Prove/non_colorato.png"
stained_path = "../../Materiale/Prove/colorato.png"

STD_FIGSIZE = (20, 20)

def show_side_images(image_1, title_1, image_2, title_2, cmap='gray', figsize=(10, 5)):
    """
    Show images side by side with titles
    
    Args:
        image_1 (numpy.ndarray): Prima immagine
        title_1 (str): Titolo della prima immagine
        image_2 (numpy.ndarray): Seconda immagine
        title_2 (str): Titolo della seconda immagine
        cmap (str, optional): Mappa di colori. Default è 'gray'.
        figsize (tuple, optional): Dimensioni della figura. Default è (10, 5).
    """

    # Creazione della figura con due sottoplot
    figure = plt.figure(figsize=figsize)
    subplots = figure.subplots(1, 2)
    figure.subplots_adjust(wspace=0.01)

    # Prima immagine
    subplots[0].set_title(title_1)
    subplots[0].axis('off')
    subplots[0].imshow(image_1, cmap=cmap)

    # Seconda immagine
    subplots[1].set_title(title_2)
    subplots[1].axis('off')
    subplots[1].imshow(image_2, cmap=cmap)

    # Mostra le immagini
    #plt.show()

def show_side_hist(image_1, title_1, image_2, title_2, figsize=(10, 5)):
    """
    Displays the histograms of two grayscale images side by side.

    Args:
        image_1 (numpy.ndarray): The first image (grayscale).
        title_1 (str): Title for the first histogram.
        image_2 (numpy.ndarray): The second image (grayscale).
        title_2 (str): Title for the second histogram.
        figsize (tuple, optional): Figure size. Default is (10, 5).

    Raises:
        TypeError: If inputs are not numpy arrays.
    """
    
    # Creazione della figura con i subplots
    figure = plt.figure(figsize=figsize)
    subplots = figure.subplots(1, 2)

    # Istogramma prima immagine
    subplots[0].set_title(title_1)
    subplots[0].hist(image_1.ravel(), bins=256, range=(0, 256), color='black', alpha=0.75)
    subplots[0].set_xlabel("Pixel Intensity")
    subplots[0].set_ylabel("Frequency")

    # Istogramam seconda immagine
    subplots[1].set_title(title_2)
    subplots[1].hist(image_2.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.75)
    subplots[1].set_xlabel("Pixel Intensity")
    subplots[1].set_ylabel("Frequency")

    # Migliora layout
    plt.tight_layout()
    plt.show()

def __hist_label_free(label_free):
    # Applico le varie normalizzazioni per i test
    label_free_eq = cv2.equalizeHist(label_free)
    label_free_norm = cv2.normalize(label_free, None, 0, 255, cv2.NORM_MINMAX)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    label_free_clahe = clahe.apply(label_free)

    # Creazione della figura con i subplots
    fig1, axes1 = plt.subplots(2, 4, figsize=(15, 10))

    # Immagini Label Free
    axes1[0, 0].imshow(label_free, cmap='gray')
    axes1[0, 0].set_title("Label Free originale")
    axes1[0, 1].imshow(label_free_eq, cmap='gray')
    axes1[0, 1].set_title("Label Free Equalized")
    axes1[0, 2].imshow(label_free_norm, cmap='gray')
    axes1[0, 2].set_title("Label Free Normalized")
    axes1[0, 3].imshow(label_free_clahe, cmap='gray')
    axes1[0, 3].set_title("Label Free CLAHE")

    # Istogrammi Label Free
    axes1[1, 0].hist(label_free.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 0].set_title("Hist Label Free originale")
    axes1[1, 1].hist(label_free_eq.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 1].set_title("Hist Label Free Equalized")
    axes1[1, 2].hist(label_free_norm.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 2].set_title("Hist Label Free Normalized")
    axes1[1, 3].hist(label_free_clahe.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 3].set_title("Hist Label Free CLAHE")

    # Migliora layout
    plt.tight_layout()
    plt.show()

def __hist_stained(stained):
    # Applico le varie normalizzazioni per i test
    stained_eq = cv2.equalizeHist(stained)
    stained_norm = cv2.normalize(stained, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    stained_clahe = clahe.apply(stained)

    # Creazione della figura con i subplots
    fig1, axes1 = plt.subplots(2, 4, figsize=(15, 10))

    # Immagini Stained
    axes1[0, 0].imshow(stained, cmap='gray')
    axes1[0, 0].set_title("Stained originale")
    axes1[0, 1].imshow(stained_eq, cmap='gray')
    axes1[0, 1].set_title("Stained Equalized")
    axes1[0, 2].imshow(stained_norm, cmap='gray')
    axes1[0, 2].set_title("Stained Normalized")
    axes1[0, 3].imshow(stained_clahe, cmap='gray')
    axes1[0, 3].set_title("Stained CLAHE")

    # Istogrammi Stained
    axes1[1, 0].hist(stained.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 0].set_title("Hist Stained originale")
    axes1[1, 1].hist(stained_eq.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 1].set_title("Hist Stained Equalized")
    axes1[1, 2].hist(stained_norm.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 2].set_title("Hist Stained Normalized")
    axes1[1, 3].hist(stained_clahe.ravel(), bins=256, range=(0, 256), color='black')
    axes1[1, 3].set_title("Hist Stained CLAHE")

    # Migliora layout
    plt.tight_layout()
    plt.show()

def show_hist(label_free, stained):
    
    # Mostro gli istogrammi originali delle due imamgini
    show_side_hist(label_free, 'Label free pre-eq', stained, 'Stained pre-eq', figsize=(10, 5))
    
    # Analisi Label Free
    __hist_label_free(label_free)
    __hist_stained(stained)
    
    
if __name__ == "__main__":

    # Controllo validità path
    if label_free_path is None or stained_path is None:
        print("Errore: una delle immagini non è stata caricata correttamente.")
        exit()

    # Lettura immagini con conversione in scala di grigi
    label_free = cv2.imread(label_free_path, cv2.IMREAD_GRAYSCALE)
    stained = cv2.imread(stained_path, cv2.IMREAD_GRAYSCALE)
    
    # Display immagini iniziali (grayscale)
    show_side_images(label_free, 'Label free', stained, 'Stained')
    
    show_hist(label_free=label_free, stained=stained)
    