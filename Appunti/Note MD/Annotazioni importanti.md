## Apertura immagini
Alcuni test hanno riportato come il metodo di apertura delle immagini causa variazioni nel numero di matches trovati.
Aprendo le due immagini di riferimento con `cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)` e con `cv2.imread(image_path) + cv2.cvtColor(image, BGR2GRAY)` si creano incongruenze.

Abbiamo aperto la prima volte le due immagini con `cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)`, normalizzato con **CLAHE**, applicato il **BFMatcher** con **crossCheck=True** (e di conseguenza la funzione `.match()`) e abbiamo ottenuto un numero di matches pari a: **2971**.

La seconda volta l'abbiamo aperto eseguendo `cv2.imread(image_path)` e poi convertendo con `cv2.cvtColor(image, BGR2GRAY)`. In questo modo invece il numero dei matches è stato: **2983**

