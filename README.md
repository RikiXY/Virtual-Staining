# Virtual Staining - Deep Learning per la Colorazione Istologica
A cura di Riccardo Giuseppe Amato, Andrea Mura
## Introduzione
Questo progetto è sviluppato nell'ambito di un **progetto universitario** e si concentra sull'implementazione di un sistema di **Virtual Staining** basato su **Deep Learning**. L'obiettivo è trasformare immagini non colorate di tessuti biologici in immagini virtualmente colorate, simulando tecniche istologiche tradizionali come **Hematoxylin & Eosin (H&E)**. Il Virtual Staining rappresenta un'alternativa innovativa alla colorazione chimica, riducendo tempi e costi e migliorando l'integrità dei campioni per l'analisi istopatologica.

L'obiettivo di questo progetto è sviluppare un modello di **Machine Learning** in **Python** per applicare la colorazione virtuale a immagini di campioni biologici, migliorando l'accuratezza e la velocità delle diagnosi mediche.

All'interno della repository è presente il file _.gitignore_ che esclude (per questioni di dimensioni di caricamento) le immagini contenute nella cartella locale _Materiale/Images/_  

Dato che GitHub non consente la corretta visualizzazione di un file markdown è consigliabile (per un'esperienza migliore) scarica il suo associato file .pdf (es: Timeline_Progetto)

---
## File importanti
All'interno della repository sono riportati una serie di file contenuti in varie cartelle:
- La cartella _Materiale/_ contiene i file sui quali basiamo test (es: _Materiale/Prove/_) e documenti
- La cartella _Appunti/_ invece contiene tre sotto cartelle:
	- _"Note MD"/_ contiene alcuni file .md (e occasionalmente i corrispettivi .pdf) in cui sono presenti pensieri e annotazioni 
	- _Scripts/_ contiene dei tentativi di script python
	- _Notebooks/_ infine forse è la directory **più importante** e contiene i notebook jupyter in cui sono presenti:
		- **_allineamento_immagini_**: confronta l'allineamento (mostrando anche i grafici\*) dell'allineamento eseguito attraverso differenti modalità (es: differenti normalizzazioni) 
		- **_confronti_**: il nome è auto esplicativo, contiene una serie di confronti tra metodi e combinazioni di filtri; è presente il primo plot di allineamento delle immagini
		- **_presentazione_coregistrazione_**: è il file presentato all'appuntamento del 19/03 (insieme al **_confronti_**) e contiene spiegazioni e chiarimenti su alcune parti del codice che hanno svolto un ruolo cruciale nella prima parte dei test
		- **_ritaglio_immagini_**: test di automatizzazione del ritaglio di una sotto immagine per i test
		- e molti altri

\* nel caso in cui i grafici o gli output non dovessero comparire basta runnare nuovamente le celle necessarie

---  
## Ordine esecuzione
Avviare dalla cartella root \(_Virtual-Staining/_\) nel seguente ordine:  
- Per addestramento:
	1. `python Appunti\Scripts\fullsize_alignment_lower_resolution.py`: esegue l'allineamento;  
	2. `python Appunti\Scripts\divide_fullsize_script.py`: divide in _n_ coppie allineate (in _aligned/_);  
	3. `create_dataset.py`: suddivide in 3 sottocartelle \(_test/_, _train/_ e _val/_\) le _n_ coppie;  
	4. `Pix2Pix.py`: addestra la rete neurale sulla cartelle _train/_ ed esegue valutazioni grazie a _val/_, i risultati della valutazioni vengono salvati in _output\_val/_;  
- Per testare:
	1. `python Pix2Pix.py test`: testa la rete sulle immagini della cartella _test/_ utilizzando il checkpoint di addestramento (da impostare manualmente nello script) e restituisce i risultati in _output\_test/_;  
	2. `python save_graphs.py`: mi salva delle immagini basate sulla tripla input/output/target nella cartella _graphs\_test/_;  

---
## To Do List
- [x] Crea repository e carica i file attuali
- [x] Coregistrazione immagini (*SIFT*)
	- [x] Selezione di una regione comune
	- [x] Valutazione metodi normalizzazione (equalizeHist, normalize o *CLAHE*)
	- [x] Valutazione algoritmo di **featuring match** (ORB/SIFT) da applicare e in che modalità (matcher, filtri, ...)
	- [x] Filtraggio ulteriore dei risultati con RANSAC (metodo di stima)
	- [ ] ~~Refinement con ECC~~ 
	- [x] Applicazione filtro di distanza euclidea
	- [x] Verifica della coregistrazione (differenza assoluta e istogrammi)
	- [x] Ritaglio finale
	- [x] Creazione maschera per risolvere problema sfondo per migliorare allineamento
	- [x] Automatizzare divisioni in sub images per il dataset. Se sono allineate ok, sennò vanno segnalate.
	- [x] Valutazione tramite miglioramento percentuale dell'istogramma (risultato ottenuto: incremento minimo)
	- [x] Sistemare i bordi perché creano conflitti con la normalizzazione e la coregistrazione
	- [x] Testare e trovare un metodo efficace per coregistrare immagini 20kx20k
- [X] Verifica del dataset
	- [X] creazione script per verificare che le dimensioni delle immagini siano coerenti
	- [X] verifica che siano abbastanza allineate (differenza assoluta)
- [ ] Sviluppo rete neurale (supervisionato)
	- [X] Suddivisione dataset 70-15-15 (Training, Validation e Test)
	- [ ] Valutazione di rete da usare (probabilmente GAN dato che viene già utilizzata. Pix2Pix; implementata in PyTorch)
	- [ ] Controlla i diffusion model
- [ ] Tutto il resto che andrà aggiunto
