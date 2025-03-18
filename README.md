# Virtual Staining - Deep Learning per la Colorazione Istologica
## Introduzione
Questo progetto è sviluppato nell'ambito di un **progetto universitario** e si concentra sull'implementazione di un sistema di **Virtual Staining** basato su **Deep Learning**. L'obiettivo è trasformare immagini non colorate di tessuti biologici in immagini virtualmente colorate, simulando tecniche istologiche tradizionali come **Hematoxylin & Eosin (H&E)**. Il Virtual Staining rappresenta un'alternativa innovativa alla colorazione chimica, riducendo tempi e costi e migliorando l'integrità dei campioni per l'analisi istopatologica.

L'obiettivo di questo progetto è sviluppare un modello di **Machine Learning** in **Python** per applicare la colorazione virtuale a immagini di campioni biologici, migliorando l'accuratezza e la velocità delle diagnosi mediche.

Le immagini sono reperibili e scaricabili dal seguente [drive](https://drive.google.com/drive/folders/1IXghZigB_MtO467T4DVMgGQ_lThMao8I?usp=sharing)
All'interno della repository è presente il file _.gitignore_ che esclude (per questioni di dimensioni di caricamento) le immagini contenute nella cartella locale _Materiale/Images/_  

## To Do List
- [x] Crea repository e carica i file attuali
- [ ] Coregistrazione immagini (da valutare il metodo)
	- [X] Selezione di una regione comune
	- [x] Valutazione metodi normalizzazione (equalizeHist, normalize o CLAHE)
	- [x] Valutazione algoritmo di **featuring match** (ORB/SIFT) da applicare e in che modalità (matcher, filtri, ...)
	- [ ] Filtraggio ulteriore dei risultati con RANSAC (metodo di stima)
	- [ ] **Refinement** con ECC
	- [ ] (Eventuale) ulteriore filtro di distanza euclidea (DA VALUTARE)
	- [ ] Verifica della coregistrazione ()
	- [ ] Ritaglio finale
- [ ] Tutto il resto che andrà aggiunto

---
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
