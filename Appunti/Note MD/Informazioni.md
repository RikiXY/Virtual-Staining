### Cosa è il featuring match?
Il **Feature Matching** serve a trovare punti corrispondenti tra due immagini e stimare la trasformazione necessaria per **coregistrarle correttamente**. È particolarmente utile quando le immagini possono avere **rotazioni, traslazioni o leggere deformazioni**.

Il **Feature Matching** ci aiuta a:
- **Identificare punti chiave comuni** tra le due immagini.
- **Stimare una trasformazione ([affine o omografica](https://aimagelab.ing.unimore.it/didattica/corsi/eia09/materiale/1%29%20dispense/6_eia_trasformazionispaziali.pdf))**, nel nostro caso forse è meglio omografica, per riallinearle correttamente.
- **Ridurre errori di allineamento** prima di applicare la registrazione finale (ECC).

Come funziona?
1. Trova punti chiave distintivi nelle due immagini
	- Algoritmi come ORB, SIFT o SURF trovano caratteristiche invarianti (es. angoli, texture).
2. Abbina i punti tra le due immagini
	- Con un matcher (es. BFMatcher) confrontiamo i descrittori delle feature.
3. Stima una trasformazione geometrica
	- Usiamo i punti corrispondenti per calcolare una matrice di trasformazione.

### Alternativa
**Perché potremmo non averne bisogno?**
Se le immagini sono già quasi perfettamente allineate e hanno solo piccole differenze di traslazione, possiamo saltare il Feature Matching e passare direttamente alla registrazione con ECC (cv2.findTransformECC).

Se abbiamo scalature, rotazioni o deformazioni, il Feature Matching ti aiuta a stimare meglio l’allineamento.

---

