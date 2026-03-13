import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ================ PRIMA PARTE =================
# Caricamento delle immagini
label_free = cv2.imread("../Materiale/Immagini/label_free_0.png")
stained = cv2.imread("../Materiale/Immagini/stained_0.png")

# Calcolo delle maschere
_, binary = cv2.threshold(cv2.cvtColor(label_free, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)
_, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
lf_mask = np.zeros_like(binary).astype(np.uint8)
for i in sorted_indices[:10]:
    x, y, w, h, area = stats[i]
    if w < 100 and h < 100:
        continue
    component_mask = (labels == i).astype(np.uint8) * 255
    countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
    roi = label_free[y:y+h, x:x+w]
    roi_mask = component_mask[y:y+h, x:x+w]
    std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]
    if std_dev < 10:
        lf_mask[component_mask == 255] = 255
lf_mask = cv2.bitwise_not(lf_mask)

_, binary = cv2.threshold(cv2.cvtColor(stained, cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY)
_, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
st_mask = np.zeros_like(binary).astype(np.uint8)
for i in sorted_indices[:10]:
    x, y, w, h, area = stats[i]
    if w < 100 and h < 100:
        continue
    component_mask = (labels == i).astype(np.uint8) * 255
    countours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(component_mask, countours, -1, 255, thickness=cv2.FILLED)
    roi = stained[y:y+h, x:x+w]
    roi_mask = component_mask[y:y+h, x:x+w]
    std_dev = cv2.meanStdDev(roi, mask=roi_mask)[1][0, 0]
    if std_dev < 10:
        st_mask[component_mask == 255] = 255
st_mask = cv2.bitwise_not(st_mask)

# Applicazione CLAHE
clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(8, 8))
lf_clahe = clahe.apply(cv2.cvtColor(label_free, cv2.COLOR_BGR2GRAY))
st_clahe = clahe.apply(cv2.cvtColor(stained, cv2.COLOR_BGR2GRAY))

# Calcolo delle features con SIFT
sift = cv2.SIFT_create(nfeatures=10000)
keypoints_1, descriptors_1 = sift.detectAndCompute(lf_clahe, lf_mask)
keypoints_2, descriptors_2 = sift.detectAndCompute(st_clahe, st_mask)

# Controllo se ci sono abbastanza features
if len(keypoints_1) < 4 or len(keypoints_2) < 4:
    raise ValueError("Non ci sono abbastanza features")

# Matching delle features
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf.match(descriptors_1, descriptors_2)

# Filtraggio dei match con distanza euclidea
filtered_matches_sift = []
for match in matches_sift:
    distance = np.linalg.norm(np.array(keypoints_1[match.queryIdx].pt) - np.array(keypoints_2[match.trainIdx].pt))
    if distance <= 200:
        filtered_matches_sift.append(match)

# Controllo se ci sono abbastanza match
if len(filtered_matches_sift) < 4:
    raise ValueError("Non ci sono abbastanza match")

# Estrazione dei punti
points_1 = np.float32([keypoints_1[match.queryIdx].pt for match in filtered_matches_sift]).reshape(-1, 1, 2)
points_2 = np.float32([keypoints_2[match.trainIdx].pt for match in filtered_matches_sift]).reshape(-1, 1, 2)

# Calcolo della matrice di trasformazione
warp_matrix, lf_mask = cv2.estimateAffinePartial2D(points_2, points_1)

# Allineamento dell'immagine e della maschera
st_aligned = cv2.warpAffine(stained, warp_matrix, (label_free.shape[1], label_free.shape[0]))

# Calcolo delle differenze assolute
original_diff = cv2.absdiff(lf_clahe, st_clahe)
st_aligned_clahe = clahe.apply(cv2.cvtColor(st_aligned, cv2.COLOR_BGR2GRAY))
aligned_diff = cv2.absdiff(lf_clahe, st_aligned_clahe)

# ELEMENTI IMPORTANTI DA VISUALIZZARE:
# 1. Immagine Label Free                        →       label_free
# 2. Immagine Stained Allineata                 →       st_aligned
# 3. Differenza Assoluta Originale              →       original_diff
# 4. Differenza Assoluta Allineata              →       aligned_diff
# 5. Istogramma Differenza Assoluta Originale   →       hist(original_diff.ravel(), bins=256, range=(0, 256), color='black')
# 6. Istogramma Differenza Assoluta Allineata   →       hist(original_diff.ravel(), bins=256, range=(0, 256), color='red', alpha=0.5)
#                                                       hist(aligned_diff.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5)

# ================ FINE PRIMA PARTE =================


# ================ SECONDA PARTE =================
# Caricamento delle immagini
label_free_2 = cv2.cvtColor(cv2.imread("../Materiale/Immagini/label_free_0.png"), cv2.COLOR_BGR2GRAY)
stained_2 = cv2.cvtColor(cv2.imread("../Materiale/Immagini/stained_0.png"), cv2.COLOR_BGR2GRAY)

# Calcolo delle features con SIFT
sift = cv2.SIFT_create(nfeatures=10000)
keypoints_1_sift, descriptors_1_sift = sift.detectAndCompute(label_free, None)
keypoints_2_sift, descriptors_2_sift = sift.detectAndCompute(stained, None)

# Calcolo delle features con ORB
orb = cv2.ORB_create(nfeatures=10000)
keypoints_1_orb, descriptors_1_orb = orb.detectAndCompute(label_free, None)
keypoints_2_orb, descriptors_2_orb = orb.detectAndCompute(stained, None)

# Matching delle features calcolate con SIFT
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(descriptors_1_sift, descriptors_2_sift)

# Filtraggio dei match con distanza euclidea
filtered_matches_sift = []
for match in matches_sift:
    distance = np.linalg.norm(np.array(keypoints_1_sift[match.queryIdx].pt) - np.array(keypoints_2_sift[match.trainIdx].pt))
    if distance <= 200:
        filtered_matches_sift.append(match)
filtered_matches_sift = sorted(filtered_matches_sift, key=lambda x: x.distance)

# Matching delle features calcolate con ORB
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(descriptors_1_orb, descriptors_2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# Filtraggio dei match con distanza euclidea
filtered_matches_orb = []
for match in matches_orb:
    distance = np.linalg.norm(np.array(keypoints_1_orb[match.queryIdx].pt) - np.array(keypoints_2_orb[match.trainIdx].pt))
    if distance <= 200:
        filtered_matches_orb.append(match)
filtered_matches_orb = sorted(filtered_matches_orb, key=lambda x: x.distance)

# ELEMENTI IMPORTANTI DA VISUALIZZARE:
# 1. Confronto delle features con SIFT      →       cv2.drawMatches(label_free, keypoints_1_sift, stained, keypoints_2_sift, filtered_matches_sift[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=5)
# 2. Confronto delle features con ORB       →       cv2.drawMatches(label_free, keypoints_1_orb, stained, keypoints_2_orb, filtered_matches_orb[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=5)
# ================= FINE SECONDA PARTE =================


# ================ VISUALIZZAZIONE FINALE =================
fig = plt.figure(figsize=(22, 10))

outer = gridspec.GridSpec(
    2, 1,
    hspace=0.01          # spazio verticale fra le righe
)

# ------------------- RIGA 1: 6 QUADRETTI -------------------
gs_row0 = outer[0].subgridspec(
    1, 6,                          # 1 riga, 6 colonne
    width_ratios=[1, 1, 1, 1, 1, 1],            
    wspace=0.4
)
ax0 = fig.add_subplot(gs_row0[0, 0])
ax1 = fig.add_subplot(gs_row0[0, 1])
ax2 = fig.add_subplot(gs_row0[0, 2])
ax3 = fig.add_subplot(gs_row0[0, 3])
ax4 = fig.add_subplot(gs_row0[0, 4])
ax5 = fig.add_subplot(gs_row0[0, 5])

# ------------------ RIGA 2: 2 RIQUADRI LARGHI ---------------
gs_row1 = outer[1].subgridspec(
    1, 2,
    width_ratios=[5, 5],  
    wspace=0.01
)
ax6 = fig.add_subplot(gs_row1[0, 0])
ax7 = fig.add_subplot(gs_row1[0, 1])

# ------- prima riga ------
ax0.imshow(cv2.cvtColor(label_free, cv2.COLOR_BGR2GRAY), cmap='gray')
ax0.set_title("Label Free");             ax0.axis('off')

ax1.imshow(cv2.cvtColor(st_aligned, cv2.COLOR_BGR2RGB))
ax1.set_title("Stained Allineata");       ax1.axis('off')

ax2.imshow(original_diff, cmap='gray')
ax2.set_title("Abs Diff Orig.");          ax2.axis('off')

ax3.imshow(aligned_diff, cmap='gray')
ax3.set_title("Abs Diff Allineata");      ax3.axis('off')

ax4.hist(original_diff.ravel(), bins=256, range=(0, 256), color='black')
ax4.set_title("Hist. Diff. Originale")

ax5.hist(original_diff.ravel(), bins=256, range=(0, 256), color='red', alpha=0.5)
ax5.hist(aligned_diff.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5)
ax5.set_title("Hist. Diff. Allineata")
ax5.legend(["Originale", "Allineata"])

for ax in (ax4, ax5):
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        ax.set_aspect('equal', adjustable='box')


# ------ seconda riga ------
ax6.imshow(cv2.drawMatches(cv2.cvtColor(label_free, cv2.COLOR_BGR2GRAY), keypoints_1_sift, cv2.cvtColor(stained, cv2.COLOR_BGR2GRAY), keypoints_2_sift, filtered_matches_sift[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=5))
ax6.set_title("Feature con SIFT");        ax6.axis('off')

ax7.imshow(cv2.drawMatches(cv2.cvtColor(label_free, cv2.COLOR_BGR2GRAY), keypoints_1_orb, cv2.cvtColor(stained, cv2.COLOR_BGR2GRAY), keypoints_2_orb, filtered_matches_orb[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=5))
ax7.set_title("Feature con ORB");         ax7.axis('off')

# -------------------------------------------------
plt.show()