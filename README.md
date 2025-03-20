# Virtual Staining - Deep Learning per la Colorazione Istologica
## Introduzione
Questo progetto è sviluppato nell'ambito di un **progetto universitario** e si concentra sull'implementazione di un sistema di **Virtual Staining** basato su **Deep Learning**. L'obiettivo è trasformare immagini non colorate di tessuti biologici in immagini virtualmente colorate, simulando tecniche istologiche tradizionali come **Hematoxylin & Eosin (H&E)**. Il Virtual Staining rappresenta un'alternativa innovativa alla colorazione chimica, riducendo tempi e costi e migliorando l'integrità dei campioni per l'analisi istopatologica.

L'obiettivo di questo progetto è sviluppare un modello di **Machine Learning** in **Python** per applicare la colorazione virtuale a immagini di campioni biologici, migliorando l'accuratezza e la velocità delle diagnosi mediche.

All'interno della repository è presente il file _.gitignore_ che esclude (per questioni di dimensioni di caricamento) le immagini contenute nella cartella locale _Materiale/Images/_  

---
## File importanti
All'interno della repository sono riportati una serie di file contenuti in varie cartelle:
- La cartella _Materiale/_ contiene i file sui quali basiamo test (es: _Materiale/Prove/_) e documenti
- La cartella _Appunti/_ invece contiene tre sotto cartelle:
	- _"Note MD"/_ contiene alcuni file .md (e occasionalmente i corrispettivi .pdf) in cui sono presenti 
	- _Scripts/_ contiene dei tentativi (mai portati avanti\*) di script python
	- _Notebooks/_ infine forse è la directory **più importante** e contiene i notebook jupyter in cui sono presenti:
		- **_alignment-images_**: trascurabile (prototipo sviluppato in un altro notebook)
		- **_allineamento_immagini_**: confronta l'allineamento (mostrando anche i grafici\*\*)  dell'allineamento eseguito attraverso differenti modalità (es: differenti normalizzazioni) 
		- **_confronti_**: il nome è auto esplicativo, contiene una serie di confronti tra metodi e combinazioni di filtri; è presente il primo plot di allineamento delle immagini
		- **_presentazione_coregistrazione_**: è il file presentato all'appuntamento del 19/03 (insieme al **_confronti_**) e contiene spiegazioni e chiarimenti su alcune parti del codice che hanno svolto un ruolo cruciale nella prima parte dei test
		- **_ritaglio_immagini_**: test di automatizzazione del ritaglio di una sotto immagine per i test

\* lo script è stato momentaneamente abbandonato per questioni logistiche e per permetterci di concentrarci principalmente sui notebook
\*\* nel caso in cui i grafici o gli output non dovessero comparire basta runnare nuovamente le celle necessarie
## To Do List
- [x] Crea repository e carica i file attuali
- [x] Coregistrazione immagini (da valutare il metodo)
	- [X] Selezione di una regione comune
	- [x] Valutazione metodi normalizzazione (equalizeHist, normalize o CLAHE)
	- [x] Valutazione algoritmo di **featuring match** (ORB/SIFT) da applicare e in che modalità (matcher, filtri, ...)
	- [x] Filtraggio ulteriore dei risultati con RANSAC (metodo di stima)
	- [x] **Refinement** con ECC
	- [x] (Eventuale) ulteriore filtro di distanza euclidea (DA VALUTARE)
	- [x] Verifica della coregistrazione (differenza assoulta e istogrammi)
	- [x] Ritaglio finale
	- [ ] Sistemare i bordi perché creano conflitti con la normalizzazione e la coregistrazione
	- [ ] Testare e trovare un metodo efficace per coregistrare immagini 20kx20k
- [ ] Tutto il resto che andrà aggiunto

---


