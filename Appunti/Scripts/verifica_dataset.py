import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns

# Percorso alla cartella delle immagini
base_path = os.path.abspath("Materiale/Locale/aligned")

# Percorso al file CSV di output
csv_path = "Appunti/Scripts/differenze_output_post.csv"

def plot_differenze_da_csv(salva_fig=False, output_path="analisi_differenze.png"):
    """
    Legge un file CSV contenente colonne 'Indice' e 'Diff_media',
    e mostra tre grafici: scatter, istogramma+KDE e boxplot.

    Args:
        csv_path (str): percorso al file CSV
        salva_fig (bool): se True salva il grafico su disco
        output_path (str): percorso dove salvare il grafico (se salva_fig=True)
    """
    # Caricamento CSV
    df = pd.read_csv(csv_path)

    # Setup figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # === 1. Scatter plot ===
    axs[0].scatter(df["Indice"], df["Diff_media"], color="royalblue", s=15)
    axs[0].set_title("Scatter delle differenze medie tra coppie")
    axs[0].set_xlabel("Indice immagine")
    axs[0].set_ylabel("Differenza media (grayscale)")
    axs[0].grid(True)

    # === 2. Istogramma + curva KDE ===
    sns.histplot(df["Diff_media"], bins=30, kde=True, ax=axs[1], color="orange", edgecolor="black")
    axs[1].set_title("Distribuzione delle differenze medie")
    axs[1].set_xlabel("Differenza media")
    axs[1].set_ylabel("Frequenza")

    # Layout e output
    plt.tight_layout()

    if salva_fig:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Grafico salvato in: {output_path}")
    else:
        plt.show()

def esporta_differenze_csv(path_csv, label_free_files, diff_values):
    with open(path_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Indice", "File", "Diff_media"])

        for i, (fname, diff) in enumerate(zip(label_free_files, diff_values)):
            writer.writerow([i, fname, round(diff, 4)])

    print(f"✅ File CSV salvato in: {path_csv}")

def plot_diff_examples(base_path, n=5):
    all_files = sorted(os.listdir(base_path))
    label_free_files = [f for f in all_files if "label_free" in f]

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(f"Prime {n} coppie: Label-Free | Stained | Differenza assoluta", fontsize=16)

    start = 70
    for i in range(n):
        lf_file = label_free_files[start + i]
        stained_file = lf_file.replace("label_free", "stained")

        lf_path = os.path.join(base_path, lf_file)
        st_path = os.path.join(base_path, stained_file)

        img_lf = cv2.imread(lf_path)
        img_lf = cv2.cvtColor(img_lf, cv2.COLOR_BGR2GRAY)
        img_st = cv2.imread(st_path)
        img_st = cv2.cvtColor(img_st, cv2.COLOR_BGR2GRAY)

        if img_lf is None or img_st is None:
            print(f"Errore nel caricamento di {lf_file} o {stained_file}")
            continue

        # Converti da BGR a RGB per matplotlib
        img_lf = cv2.cvtColor(img_lf, cv2.COLOR_BGR2RGB)
        img_st = cv2.cvtColor(img_st, cv2.COLOR_BGR2RGB)
        diff = cv2.absdiff(img_lf, img_st)

        # Plot
        axes[i, 0].imshow(img_lf, cmap="gray")
        axes[i, 1].imshow(img_st, cmap="gray")
        axes[i, 2].imshow(diff, cmap="gray")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_min_max(base_path, label_free_files, min_idx, max_idx):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Confronto: Minima vs Massima differenza", fontsize=16)

    for row, idx in enumerate([min_idx, max_idx]):
        lf_file = label_free_files[idx]
        st_file = lf_file.replace("label_free", "stained")

        lf_path = os.path.join(base_path, lf_file)
        st_path = os.path.join(base_path, st_file)

        img_lf = cv2.imread(lf_path, cv2.IMREAD_GRAYSCALE)
        img_st = cv2.imread(st_path, cv2.IMREAD_GRAYSCALE)
        diff = cv2.absdiff(img_lf, img_st)

        # Plot: col 0 → label_free, col 1 → stained, col 2 → diff
        axes[row, 0].imshow(img_lf, cmap="gray")
        axes[row, 0].set_title("Label-Free")
        axes[row, 1].imshow(img_st, cmap="gray")
        axes[row, 1].set_title("Stained")
        axes[row, 2].imshow(diff, cmap="gray")
        axes[row, 2].set_title("Diff")

        for col in range(3):
            axes[row, col].axis("off")

        label = "Minima" if row == 0 else "Massima"
        print(f"→ {label} differenza: {lf_file} vs {st_file}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    
    # Trova tutti i file nella cartella
    all_files = sorted(os.listdir(base_path))

    # Filtra solo i file label_free (li useremo per trovare le coppie)
    label_free_files = [f for f in all_files if "label_free" in f]

    diff_values = []
    indices = []

    # Controllo per ogni coppia
    for i, lf_file in enumerate(label_free_files):
        index = lf_file.split("_")[0]  # Es. '0001' da '0001_0001_label_free.png'
        stained_file = lf_file.replace("label_free", "stained")

        lf_path = os.path.join(base_path, lf_file)
        st_path = os.path.join(base_path, stained_file)

        if not os.path.exists(st_path):
            print(f"[ERRORE] Immagine stained mancante per {lf_file}")
            continue

        # Carica le immagini
        img_lf = cv2.imread(lf_path)
        img_lf = cv2.cvtColor(img_lf, cv2.COLOR_BGR2GRAY)
        img_st = cv2.imread(st_path)
        img_st = cv2.cvtColor(img_st, cv2.COLOR_BGR2GRAY)

        # Verifica dimensioni
        if img_lf.shape != img_st.shape:
            print(f"[DIMENSIONI NON CORRISPONDENTI] {lf_file} e {stained_file}")
            continue

        # Calcola la differenza assoluta (per ogni pixel e canale)
        diff = cv2.absdiff(img_lf, img_st)
        mean_diff = np.mean(diff)  # Valore medio della differenza per pixel
        
        diff_values.append(mean_diff)
        indices.append(i)
        
        print(f"{lf_file} vs {stained_file} -> diff media: {mean_diff:.2f}")
        
    # Trova indice della differenza minima e massima
    min_idx = indices[np.argmin(diff_values)]
    max_idx = indices[np.argmax(diff_values)]
    print(f"Minimo: {min_idx} - Massimo: {max_idx}")
    plot_min_max(base_path, label_free_files, min_idx, max_idx)
    
    esporta_differenze_csv(csv_path, label_free_files, diff_values)
    

if __name__ == "__main__":
    # plot_diff_examples(os.path.abspath("Materiale/Locale/aligned"))
    main()
    plot_differenze_da_csv()  # solo visualizzazione

    """
    Interpretazione della diff media = 19.68
    Su un'immagine grayscale 8-bit (valori da 0 a 255), una differenza media di ~19 su 1000x1000 pixel è:
        - Piuttosto bassa/moderata (≲ 8% rispetto al range massimo)
        - Coerente con differenze strutturali e non randomiche
        - Buona per una rete neurale che deve imparare la trasformazione di dominio
        
    Sono decisamente buoni, perché:
    Aspetto	Valutazione
    Co-registrazione visiva	    |       Ottima
    Distribuzione delle diff	|       Normale, con pochi outlier
    Quantità di dati	        |       Oltre 1000 coppie è eccellente
    Differenze strutturate  	|       Rete può imparare il mapping
    Rumore/artifact	            |       Non visibile nei dati esaminati
    """