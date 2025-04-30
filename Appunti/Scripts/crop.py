import SimpleITK as sitk

# Carica l'immagine intera (grande)
img = sitk.ReadImage("Materiale/Locale/liver_stained.tif")

# Imposta dimensione e posizione del crop
start = [6000, 10000]  # pixel di partenza [x, y]
size = [1024, 1024]   # dimensione [larghezza, altezza]

# Esegui crop
region = sitk.RegionOfInterest(img, size, start)
sitk.WriteImage(region, "Materiale/Locale/liver_stained_crop.tif")
