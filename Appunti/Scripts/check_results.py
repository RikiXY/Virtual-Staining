import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Percorsi base
target_dir = 'Materiale/Locale/dataset_split/test'
output_dir = 'Materiale/Locale/output_test'

def compute_similarity_score(ssim_val, delta_e_mean, w_ssim=0.5, w_deltae=0.5):
    # Normalizza deltaE (max valore percepibile = 10)
    sim_ssim = ssim_val  # già in [0,1]
    sim_deltae = 1 - min(delta_e_mean / 10, 1)  # anche questo in [0,1]
    return 100 * (w_ssim * sim_ssim + w_deltae * sim_deltae)

def evaluate_images(img_target_path, img_output_path):
    # Caricamento immagini
    img_tgt = io.imread(img_target_path)
    img_out = io.imread(img_output_path)

    # Controllo dimensioni
    if img_out.shape != img_tgt.shape:
        raise ValueError(f"Immagini diverse: {img_target_path} vs {img_output_path}")

    # Conversione in float [0,1]
    img_out = img_out.astype(np.float32) / 255.0
    img_tgt = img_tgt.astype(np.float32) / 255.0

    # Scegliamo la dimensione della finestra per SSIM
    min_side = min(img_tgt.shape[0], img_tgt.shape[1])
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        raise ValueError(f"Immagine troppo piccola per SSIM: {img_target_path}")

    # SSIM (struttura)
    ssim_val = ssim(img_tgt, img_out, channel_axis=-1, data_range=1.0, win_size=win_size)

    # Lab e ΔE
    lab_out = color.rgb2lab(img_out)
    lab_tgt = color.rgb2lab(img_tgt)
    deltaE_map = color.deltaE_ciede2000(lab_tgt, lab_out)
    deltaE_mean = np.mean(deltaE_map)

    return ssim_val, deltaE_mean

def main():
    risultati = []

    for file in tqdm(os.listdir(target_dir)):
        if file.endswith('_stained.tif'):
            base_name = file.replace('_stained.tif', '')
            path_target = os.path.join(target_dir, file)
            path_output = os.path.join(output_dir, f"{base_name}_label_free_generated.tif")

            if not os.path.exists(path_output):
                print(f"Output mancante per {base_name}, saltato.")
                continue

            try:
                ssim_val, deltaE_val = evaluate_images(path_target, path_output)
                similarity = compute_similarity_score(ssim_val, deltaE_val)
                risultati.append({
                    'filename': base_name,
                    'ssim': ssim_val,
                    'delta_e_mean': deltaE_val,
                    'similarity_percent': similarity
                })
            except Exception as e:
                print(f"Errore con {base_name}: {e}")
    
    # Salva in CSV
    df = pd.DataFrame(risultati)
    media = df['similarity_percent'].mean()
    print(f"\nSimilarità media sul dataset: {media:.2f}%")
    df.to_csv("Materiale/Locale/valutazione_staining.csv", index=False, sep=';', decimal=',')
    # print("Valutazione completata e salvata in 'valutazione_staining.csv'")

if __name__ == '__main__':
    main()
