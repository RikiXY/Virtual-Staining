import os
import random
import shutil

def split_dataset(aligned_dir, output_dir,
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  extension=".tif", seed=42):
    """
    Suddivide casualmente le coppie di immagini presenti in `aligned_dir`
    nelle cartelle train/val/test create in `output_dir`, con le proporzioni
    specificate da train_ratio, val_ratio e test_ratio.
    
    Parametri:
    - aligned_dir : cartella contenente tutte le immagini (coppie).
    - output_dir  : cartella dove verranno create train/, val/, test/.
    - train_ratio : percentuale di immagini per il training (default 70%).
    - val_ratio   : percentuale di immagini per la validation (default 15%).
    - test_ratio  : percentuale di immagini per il test (default 15%).
    - extension   : estensione dei file immagine (es. ".png", ".jpg").
    - seed        : seme per la generazione random (riproducibilità).
    """
    # Imposta il seed per ottenere sempre la stessa suddivisione, se necessario
    random.seed(seed)

    # Crea cartelle di destinazione (train/val/test)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # 1) Individua i file che terminano con '_label_free' + extension
    all_files = os.listdir(aligned_dir)
    label_free_files = sorted([f for f in all_files if f.endswith("_label_free" + extension)])
    print(f"Totale file _label_free trovati: {len(label_free_files)}")

    # 2) Ricava il "prefisso" comune (es. "00000_08500") per ogni coppia
    #    e verifica che esista il corrispondente file "_stained"
    pairs_prefixes = []
    for lf_file in label_free_files:
        prefix = lf_file.replace("_label_free" + extension, "")
        stained_file = prefix + "_stained" + extension
        if stained_file in all_files:
            pairs_prefixes.append(prefix)
    print(f"Totale coppie trovate: {len(pairs_prefixes)}")
    # print(f"Elenco primi 50 prefissi: {pairs_prefixes[:50]}")

    # 3) Mescola casualmente i prefissi per suddividere in train/val/test
    random.shuffle(pairs_prefixes)
    num_total = len(pairs_prefixes)

    train_end = int(num_total * train_ratio)
    val_end   = train_end + int(num_total * val_ratio)
    # test_end non serve esplicitamente: tutto quello che rimane va in test

    train_prefixes = pairs_prefixes[:train_end]
    val_prefixes   = pairs_prefixes[train_end:val_end]
    test_prefixes  = pairs_prefixes[val_end:]

    print(f"Totale coppie trovate: {num_total}")
    print(f" - Train: {len(train_prefixes)}")
    print(f" - Val:   {len(val_prefixes)}")
    print(f" - Test:  {len(test_prefixes)}")

    # 4) Funzione per copiare i file dati prefix e cartella di destinazione
    def copy_pair(prefix, subset_folder):
        lf_name = prefix + "_label_free" + extension
        st_name = prefix + "_stained" + extension

        lf_src = os.path.join(aligned_dir, lf_name)
        st_src = os.path.join(aligned_dir, st_name)

        lf_dst = os.path.join(output_dir, subset_folder, lf_name)
        st_dst = os.path.join(output_dir, subset_folder, st_name)

        shutil.copy2(lf_src, lf_dst)
        shutil.copy2(st_src, st_dst)

    # 5) Copia effettiva delle coppie
    for pfx in train_prefixes:
        copy_pair(pfx, "train")
    for pfx in val_prefixes:
        copy_pair(pfx, "val")
    for pfx in test_prefixes:
        copy_pair(pfx, "test")

    print("Suddivisione completata con successo!")

if __name__ == "__main__":
    aligned_dir = "Materiale/Locale/aligned"       # cartella con tutte le 3000 coppie
    output_dir  = "Materiale/Locale/dataset_split" # cartella dove creare train/val/test

    split_dataset(aligned_dir, output_dir,
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  extension=".tif", seed=123)