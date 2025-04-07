import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import sys

def main():

    os.makedirs(save_path, exist_ok=True)

    # Trova tutti i file *_label_free.tif nella cartella di testing
    input_files = sorted(glob(os.path.join(test_folder, '*_label_free.tif')))
    # print(f"Found {len(input_files)} input files.")

    for input_file in input_files:
        base_name = os.path.basename(input_file).replace('_label_free.tif', '')
        # print(f"Processing {base_name}...")
        
        # Percorsi degli altri due file
        target_file = os.path.join(test_folder, f'{base_name}_stained.tif')
        # print(f"Target file: {target_file}")
        output_file = os.path.join(output_folder, f'{base_name}_label_free_generated.tif')
        # print(f"Output file: {output_file}")
        
        # Verifica che tutti e tre i file esistano
        if not all(os.path.exists(f) for f in [input_file, output_file, target_file]):
            # print(f"Skipping {base_name}: one or more files not found.")
            continue

        # Carica le immagini
        img_input = mpimg.imread(input_file)
        img_output = mpimg.imread(output_file)
        img_target = mpimg.imread(target_file)

        # Crea la figura
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_input)
        axes[0].set_title('Input')
        axes[0].axis('off')

        axes[1].imshow(img_output)
        axes[1].set_title('Output')
        axes[1].axis('off')

        axes[2].imshow(img_target)
        axes[2].set_title('Target')
        axes[2].axis('off')

        plt.tight_layout()

        # Salva la figura
        save_file = os.path.join(save_path, f'{base_name}.png')
        plt.savefig(save_file, dpi=300)
        plt.close()

        # print(f"Saved: {save_file}")
    print("All images processed and saved.")

if __name__ == "__main__":
    # Path comune
    test_folder = "Materiale/Locale/dataset_split/test/"

    if(len(sys.argv) >= 2 and sys.argv[1] == "Pix2Pix"):
        # Percorsi delle cartelle
        output_folder = "Materiale/Locale/Pix2Pix/output_test"
        save_path = "Materiale/Locale/Pix2Pix/graphs_test"
        main()
    elif (len(sys.argv) >= 2 and sys.argv[1] == "Pix2Pix+"):
        # Percorsi delle cartelle
        output_folder = "Materiale/Locale/Pix2Pix+/output_test"
        save_path = "Materiale/Locale/Pix2Pix+/graphs_test"
        main()
    else:
        print("Specificare il tipo di modello: Pix2Pix o Pix2Pix+")