Il progetto di **_Virtual Staining_** nasce dall’esigenza di semplificare e velocizzare il processo di analisi istopatologica attraverso tecniche di **colorazione virtuale**. In ambito medico, le immagini ottenute tramite microscopia _label-free_ (generalmente in scala di grigi), pur essendo meno invasive e più rapide da acquisire, risultano di difficile interpretazione senza una colorazione specifica come quella H&E (ematossilina-eosina). L’obiettivo di questo lavoro è stato quindi quello di sviluppare un sistema in grado di trasformare automaticamente un’immagine _label-free_ in una sua controparte virtualmente colorata (_stained_), riducendo i tempi di attesa e di valutazione da parte di specialisti.

Per raggiungere questo scopo, abbiamo progettato una pipeline che parte dall’**allineamento** tra immagini _label-free_ e colorate reali, passa per la **suddivisione in sottosezioni** e culmina nell’addestramento di un modello di deep learning basato sull’architettura **Pix2Pix**. Quest’ultima, composta da un **generatore** e da un **discriminatore**, è in grado di apprendere la trasformazione da un dominio all’altro attraverso un approccio supervisionato. Il risultato finale è un sistema in grado di generare immagini realisticamente colorate a partire da input privi di colorazione, con applicazioni potenzialmente molto rilevanti in ambito diagnostico.

---
### Fasi del lavoro svolto
Il progetto prevede una pipeline articolata in più fasi distinte, ciascuna delle quali è risultata fondamentale per il corretto apprendimento e la generazione delle immagini colorate. Di seguito si riportano le principali tappe operative:
1. **Allineamento delle immagini:** La prima fase ha riguardato l’allineamento spaziale tra le immagini _label-free_ e le corrispondenti immagini colorate H&E. Poiché le due immagini possono presentare disallineamenti dovuti al processo di acquisizione, è stato necessario calcolare una trasformazione affine basata su feature locali ottenute attraverso l'ausilio di algoritmi come **ORB** e **SIFT**. A tal fine, si è utilizzato l’algoritmo **SIFT** per l’estrazione dei _keypoints_, associato a un sistema di normalizzazione del contrasto (**CLAHE**), sviluppo di maschere e filtraggio euclideo, al fine di migliorare la qualità delle corrispondenze. La matrice di trasformazione risultante è stata poi applicata all’intera immagine colorata e alla relativa maschera.
2. **Suddivisione in _patch_:**  Una volta ottenuta l’immagine allineata, si è proceduto a suddividerla in sottoregioni regolari (_patch_) di dimensione 512×512 pixel, con passo di 300 pixel. Questa suddivisione ha permesso di generare un numero consistente di coppie immagine _label free_-_stained_ da utilizzare durante l’addestramento. Per garantire la qualità del dataset, sono stati esclusi automaticamente i quadranti contenenti prevalentemente sfondo.
3. **Creazione del dataset strutturato:** Le patch ottenute sono state organizzate in un dataset strutturato secondo la convenzione **train/val/test**, rispettando una suddivisione **70/15/15**. Durante questa fase è stata verificata la coerenza tra ciascuna immagine _label-free_ e la sua corrispondente _stained_, assicurando che ogni coppia fosse completa e correttamente nominata. L’uso di un seme randomico fisso ha garantito la riproducibilità della partizione.
4. **Addestramento del modello Pix2Pix:** Per la fase di colorazione virtuale è stata adottata una rete generativa condizionata basata sull’architettura **Pix2Pix**, composta da un generatore tipo **UNet** e da un discriminatore **PatchGAN**. Il modello è stato addestrato in modalità supervisionata, ottimizzando una combinazione di metriche di loss come **Binary Cross Entropy** (**BCE**) e **Mean Absolute Error** (**L1**). Durante l’addestramento è stata eseguita una validazione periodica per monitorare l’andamento delle prestazioni su dati non visti.

---
### Risultati ottenuti
L’addestramento e la successiva fase di testing hanno permesso di ottenere immagini virtualmente colorate che, in molti casi, riproducono fedelmente le strutture e le cromie delle rispettive immagini target. La qualità visiva degli output è stata valutata principalmente attraverso un confronto diretto tra triple di immagini (_input, output, target_), rese disponibili grazie a uno script di visualizzazione automatica.  

Di seguito è riportata una delle triple sopra menzionate:![[Pasted image 20250516125608.png]]

---
### Competenze acquisite
Nel corso del progetto sono state acquisite competenze tecniche rilevanti sia nell’ambito dell’elaborazione di immagini che del deep learning. In particolare, è stato approfondito l’uso di tecniche di **preprocessing** come l’equalizzazione locale del contrasto (**CLAHE**), l’allineamento tra immagini tramite feature locali (**SIFT**) e la mascheratura selettiva per l’esclusione automatica di regioni non informative.

Un’altra area centrale ha riguardato l’addestramento di reti neurali generative, dove è stato utilizzato il framework **PyTorch** per implementare e addestrare il modello **Pix2Pix** e, successivamente, una variante avanzata con **UNet** e discriminatore migliorato. Questo ha incluso l’impiego di tecniche come la normalizzazione delle immagini, la gestione dei batch su GPU, l’ottimizzazione di più funzioni di perdita e il salvataggio dei checkpoint per un monitoraggio efficiente dell’apprendimento.

Infine, il progetto ha richiesto una buona **organizzazione della pipeline sperimentale**, includendo il logging delle sessioni, la validazione periodica e la generazione automatica dei risultati grafici. Questo ha permesso di affinare anche capacità trasversali di debugging, documentazione e valutazione critica del comportamento dei modelli.

---
### Considerazioni finali
L’esperienza progettuale si è rivelata estremamente formativa sia dal punto di vista tecnico che metodologico. Ha permesso di applicare concetti teorici a un problema reale e attuale, affrontando sfide pratiche legate alla qualità del dato, all’efficienza computazionale e alla progettazione di modelli generativi condizionati.

In prospettiva, questo lavoro potrebbe rappresentare un punto di partenza utile per chiunque voglia approfondire l’applicazione di tecniche di deep learning alla colorazione virtuale di immagini biomediche. La struttura modulare della pipeline e la chiarezza del codice la rendono facilmente adattabile ad altri contesti o a nuovi dataset. L’intero progetto è stato reso disponibile pubblicamente nella repository GitHub: [https://github.com/RikiXY/Virtual-Staining](https://github.com/RikiXY/Virtual-Staining), con l’obiettivo di condividerne i risultati e favorire possibili estensioni o contributi futuri da parte della comunità.




<p align="right">
Università degli Studi di Cagliari<br>
Corso di Laurea in Ingegneria Elettronica, Informatica e delle Telecomunicazioni<br>
A.A. 2024/2025<br>
<br>
<strong>Riccardo Giuseppe Amato</strong> e <strong>Andrea Mura</strong>
</p>
---
### Bibliografia
- Bijie Bai, Xilin Yang, Yuzhu Li, Yijie Zhang, Nir Pillar, Aydogan Ozcan - _Deep learning-enabled virtual histological staining of biological samples_ [link](https://www.nature.com/articles/s41377-023-01104-7)
- Umair Khan, Sonja Koivukoski, Mira Valkonen, Leena Latonen, Pekka Ruusuvuori - _The effect of neural network architecture on virtual H&E staining: Systematic assessment of histological feasibility_ [link](https://www.sciencedirect.com/science/article/pii/S266638992300065X)
- OpenCV library – [https://opencv.org/](https://opencv.org/)  
- PyTorch library – [https://pytorch.org/](https://pytorch.org/)  
