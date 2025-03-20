import cv2, os
import numpy as np
from random import randint

# Ritaglio dell'immagine
class Cropping:
    @staticmethod
    def crop_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Crop an image.
        
        :param img: Image to crop.
        :param x: Top-left x coordinate of the crop rectangle.
        :param y: Top-left y coordinate of the crop rectangle.
        :param w: Width of the crop rectangle.
        :param h: Height of the crop rectangle.
        :return: Cropped image.
        """
        return img[y:y+h, x:x+w]

# Funzioni di pre-elaborazione
class Preprocessing:
    @staticmethod
    def to_clahe(img: np.ndarray, clip_limit: float = 18.0, tile_size: tuple = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
        
        :param img: Image to enhance.
        :param clip_limit: Threshold for contrast limiting.
        :param tile_size: Size of the contextual regions for histogram equalization.
        :return: Image enhanced with CLAHE.
        """
        # Converti l'immagine in scala di grigi se non lo è già
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Applica CLAHE all'immagine
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)

# Funzioni di misura
class Measurements:
    @staticmethod
    def computer_mi(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
        """
        Compute the mutual information (MI) between two grayscale images.
        
        :param img1: First image in grayscale.
        :param img2: Second image in grayscale.
        :param bins: Number of histogram bins to use (256 for 8-bit images).
        :return: Mutual information (in bits) between img1 and img2.
        """
        # Converti le immagini in array numpy 1D (appiattisci) se non lo sono già
        img1 = img1.ravel()
        img2 = img2.ravel()

        # Calcola l'istogramma congiunto
        # histogram2d restituisce l'istogramma 2D e i bordi dei bin
        joint_hist, x_edges, y_edges = np.histogram2d(img1, img2, bins=bins)
        
        # Normalizza l'istogramma congiunto per ottenere la PDF congiunta p(x,y)
        joint_pdf = joint_hist / np.sum(joint_hist)
        
        # Calcola le PDF marginali p(x) e p(y)
        p_x = np.sum(joint_pdf, axis=1)  # Somma sulle colonne -> marginale su x
        p_y = np.sum(joint_pdf, axis=0)  # Somma sulle righe -> marginale su y

        # Considera solo i valori non nulli per evitare log(0)
        # MI = somma p(x,y)*log( p(x,y)/(p(x)*p(y)) )
        non_zero_idxs = joint_pdf > 0
        mi = np.sum(joint_pdf[non_zero_idxs] * 
                    np.log2(joint_pdf[non_zero_idxs] / 
                            (p_x[np.newaxis].T @ p_y[np.newaxis])[non_zero_idxs]))
        
        return mi

    @staticmethod
    def compute_mse(img1, img2) -> float:
        """
        Compute the mean squared error (MSE) between two images.
        
        :param img1: First image.
        :param img2: Second image.
        :return: Mean squared error between img1 and img2.
        """
        # Calcola l'errore quadratico tra le due immagini
        se = (img1 - img2) ** 2
        
        # Calcola l'errore quadratico medio
        mse = np.mean(se)
        
        return mse

# Filtri
class Filters:
    FILTER_DISTANCE = 200
    LOWES_RATIO = 0.75

    @staticmethod
    def filter_LR(matches: list, threshold: float = LOWES_RATIO) -> list:
        """
        Filter matches using Lowe's ratio test.
        
        :param matches: List of DMatch objects.
        :param threshold: Lowe's ratio threshold.
        :return: List of filtered DMatch objects.
        """
        filtered_matches = []
        for m in matches:
            if m[0].distance < threshold * m[1].distance:
                filtered_matches.append(m[0])
        return filtered_matches

    @staticmethod
    def filter_ED(matches: list, keypoints: tuple, distance: float = FILTER_DISTANCE) -> list:
        """
        Filter matches based on Euclidean distance between keypoints.
        
        :param matches: List of DMatch objects.
        :param keypoints: Tuple of keypoints from two images.
        :param distance: Euclidean distance threshold.
        :return: List of filtered DMatch objects.
        """
        filtered_matches = []
        for m in matches:
            # Calcola la distanza euclidea tra i punti chiave corrispondenti
            euclidean_distance = np.linalg.norm(np.array(keypoints[0][m.queryIdx].pt) - np.array(keypoints[1][m.trainIdx].pt))
            if euclidean_distance < distance:
                filtered_matches.append(m)
        return filtered_matches

# Allineamento
class Alignment:
    MODE_SIFT = 0
    MODE_ORB = 1

    NFEATURES = 10000

    def align(img1, img2, img_to_align=None, mode=MODE_SIFT, lr=False, ed=False, nfeatures=NFEATURES, ed_distance=Filters.FILTER_DISTANCE):
        """
        Align two images using SIFT or ORB features.
        Can use Lowe's ratio test and Euclidean distance filtering.
        """
        # Se img_to_align non è specificata, allinea img2 rispetto a img1
        if img_to_align is None:
            img_to_align = img2
        
        # Calcola i punti chiave e i descrittori delle immagini
        if mode == Alignment.MODE_SIFT:
            sift = cv2.SIFT_create(nfeatures=nfeatures)
            keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
        elif mode == Alignment.MODE_ORB:
            orb = cv2.ORB_create(nfeatures=nfeatures)
            keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
            keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)
        else:
            raise Exception("Mode not supported")
        
        # Match dei descrittori con il metodo di forza bruta
        if lr:
            # Se si usa il test di Lowe va disattivato il crossCheck
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
            matches = Filters.filter_LR(matches)
        else:
            # Senza test di Lowe si usa il crossCheck
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors_1, descriptors_2)

        if ed: # Se si usa il filtro sulla distanza euclidea
            filtered_matches = Filters.filter_ED(matches, (keypoints_1, keypoints_2), distance=ed_distance)
        else:
            filtered_matches = matches

        # Se non ci sono abbastanza corrispondenze, non è possibile allineare le immagini
        if len(filtered_matches) < 3:
            raise Exception("Not enough matches to align images")
        
        # Estrai i punti chiave corrispondenti per calcolare la matrice di trasformazione
        points = ([], [])
        for match in filtered_matches:
            points[0].append(keypoints_1[match.queryIdx].pt)
            points[1].append(keypoints_2[match.trainIdx].pt)
        
        # Calcola la matrice di trasformazione per allineare le immagini
        # estimateAffine2D crea una matrice di trasformazione che può solo
        # traslare, ruotare e ridimensionare l'immagine
        points_1 = np.array(points[0])
        points_2 = np.array(points[1])
        warp_matrix, mask = cv2.estimateAffinePartial2D(points_2, points_1)

        # Applica la matrice di trasformazione all'immagine da allineare
        height, width = img_to_align.shape[:2]
        aligned_image = cv2.warpAffine(img_to_align, warp_matrix, (width, height))

        return aligned_image

def crop_and_align(fullsize_label_free: np.ndarray, fullsize_stained: np.ndarray):
    """
    Crop and align two images, then compute the mutual information and mean squared error.
    """
    # Parametri casuali per ritagliare le immagini
    fullsize_width = min(fullsize_label_free.shape[1], fullsize_stained.shape[1])
    fullsize_height = min(fullsize_label_free.shape[0], fullsize_stained.shape[0])
    size = randint(2000, 4000)
    x = randint(int(fullsize_width * 0.1), int(fullsize_width * 0.9) - size)
    y = randint(int(fullsize_height * 0.1), int(fullsize_height * 0.9) - size)

    # Ritaglia le immagini
    cropped_label_free = Cropping.crop_image(fullsize_label_free, x, y, size, size)
    cropped_stained = Cropping.crop_image(fullsize_stained, x, y, size, size)
    print(f"Immagini ritagliate correttamente alle coordinate ({x}, {y}) con dimensioni {size}x{size}")

    # Applica CLAHE alle immagini
    clahe_label_free = Preprocessing.to_clahe(cropped_label_free)
    clahe_stained = Preprocessing.to_clahe(cropped_stained)
    original_mi = Measurements.computer_mi(clahe_label_free, clahe_stained)
    original_mse = Measurements.compute_mse(clahe_label_free, clahe_stained)
    print("Immagini pre-elaborate con CLAHE")

    # Allinea le immagini
    try:
        aligned_stained = Alignment.align(clahe_label_free, clahe_stained, img_to_align=cropped_stained, mode=Alignment.MODE_SIFT, lr=False, ed=True)
    except Exception as e:
        print("Errore nell'allineamento delle immagini:", e)
        cv2.imwrite(f"Locale/Errore/{x}_{y}_{size}x{size}_lf.tif", cropped_label_free)
        cv2.imwrite(f"Locale/Errore/{x}_{y}_{size}x{size}_st.tif", cropped_stained)
        return
    print("Immagini allineate correttamente")

    # Calcola la misura di similarità tra le immagini
    clahe_aligned_stained = Preprocessing.to_clahe(aligned_stained)
    mi = Measurements.computer_mi(clahe_label_free, clahe_aligned_stained)
    mse = Measurements.compute_mse(clahe_label_free, clahe_aligned_stained)

    mi_diff = mi - original_mi
    mse_diff = mse - original_mse

    print(f"MI originale: {original_mi:.3f} - MSE originale: {original_mse:.3f}")
    print(f"MI: {mi:.3f} - MSE: {mse:.3f}")
    print(f"Differenza MI: {mi_diff:.3f} - Differenza MSE: {mse_diff:.3f}")

def main():
    # Carica le immagini
    fullsize_label_free = cv2.imread("Materiale/Locale/fullsize_label_free.tif")
    fullsize_stained = cv2.imread("Materiale/Locale/fullsize_stained.tif")

    if fullsize_label_free is None:
        print("Errore nel caricamento dell'immagine fullsize_label_free.tif")
        return
    if fullsize_stained is None:
        print("Errore nel caricamento dell'immagine fullsize_stained.tif")
        return
    print("Immagini caricate correttamente")

    # Creazione delle cartelle se non esistenti
    if not os.path.exists("Locale/MI"):
        os.makedirs("Locale/MI")
    if not os.path.exists("Locale/MSE"):
        os.makedirs("Locale/MSE")
    if not os.path.exists("Locale/Errore"):
        os.makedirs("Locale/Errore")

    for i in range(50):
        crop_and_align(fullsize_label_free, fullsize_stained)

if __name__ == "__main__":
    main()