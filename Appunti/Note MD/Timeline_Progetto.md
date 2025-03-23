# Timeline Progetto Virtual-Staining 2025
Purtroppo ho pensato una settimana troppo tardi di creare questo file per tenere traccia in maniera rigorosa e più dettagliata del lavoro svolto, quindi nel lasso temporale tra il 14/03 e il 19/03 saranno presenti tutte le cose fatte (con problemi riscontrati e soluzioni attuate).  
Per riscontri dettagliati vedere i notebook.  

Legenda colori:  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#e15c64; border-radius: 2px;"></span> <b style="margin-left: 8px;">Problemi riscontrati</b> – criticità rilevate nel processo  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#71c78c; border-radius: 2px;"></span> <b style="margin-left: 8px;">Soluzioni attuate</b> – strategie o interventi correttivi  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#f7b267; border-radius: 2px;"></span> <b style="margin-left: 8px;">Futura implementazione</b> – idee o ottimizzazioni da valutare  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#f4d35e; border-radius: 2px;"></span> <b style="margin-left: 8px;">Parole chiave</b> – concetti centrali del progetto  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#9bb1ff; border-radius: 2px;"></span> <b style="margin-left: 8px;">Metodi applicati</b> – tecniche e algoritmi impiegati  

---
## 11 Marzo 
- **Inizio progetto**  
- **Assegnazione lettura** [paper](https://www.nature.com/articles/s41377-023-01104-7)  
## Dal 14 Marzo al 19 Marzo 
- **Creazione repository**  

<span style="color:#e15c64"><b>Problema riscontrato</b></span> : durante la discussione valutativa iniziale sul progetto è emersa la mancanza di un dataset adibito al training e testing della rete neurale da addestrare per eseguire il virtual staining.  
<span style="color:#71c78c"><b>Soluzione proposta</b></span>: dato che ci sono state fornite alcune immagini di grandi dimensioni, seguendo anche i consigli dei docenti, abbiamo deciso di lavorare con sotto porzioni di essa.  

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: le immagini fornite non erano coregistrate e ciò avrebbe indotto degli errori durante la fase di training dell'algoritmo.  
<span style="color:#71c78c"><b>Soluzione attuata</b></span>: sviluppo programma per la coregistrazione delle immagini.  

- **Selezione sotto porzione comunque tra l'immagine label-free e stained**  

- **Valutazione metodi di normalizzazione**  
*Spiegazione*: abbiamo optato, come primo step (in realtà secondo dato che il primo è la <span style="color:#9bb1ff"><b>conversione in scala di grigi</b></span>) di effettuare il processo di normalizzazione delle foto. Ciò è dovuto sia a una questione legata alla robustezza del programma: normalizzare tutte le immagini permette un confronto effettivo tra esse, risolvendo alcuni problemi dettati dalla scannerizzazione (diverse tipologie di scanner) e illuminazione. Inoltre, dato che la coregistrazione si basa sullo sviluppo di una <span style="color:#f4d35e"><b>matrice omografica</b></span> (trasformazione geometrica che mappa punti da un'immagine a un'altra quando le immagini rappresentano lo stesso soggetto ma da prospettive differenti), e di conseguenza dell'identificazione e confronto di <span style="color:#f4d35e"><b>feature</b></span> (_feature matching_), abbiamo ritenuto che normalizzare le immagini potesse migliorare la ricerca (dato un incremento di contrasto) degli elementi morfologici più distintivi dell'immagine.  
Dai test svolti abbiamo deciso di usare la normalizzazione <span style="color:#9bb1ff"><b>CLAHE</b></span>.  

- **Valutazione algoritmo di Feature match**  
*Spiegazione*: abbiamo optato come soluzione per la coregistrazione delle immagini quello di applicare algoritmi di feature match per via della semplicità di implementazione e per via del buon compromesso tempo-risultati. <span style="color:#f7b267"><b>Una valida opzione da considerare in futuro potrebbe essere quella di applicare algoritmi di tipo Intensity-based.</b></span> Abbiamo eseguito test sia su <span style="color:#9bb1ff"><b>ORB</b></span> che su <span style="color:#9bb1ff"><b>SIFT</b></span>; entrambi potrebbero essere usati (dipende dalle risorse della macchina e dalla grandezza dell'immagine).  

- **Applicazione filtri**  
*Spiegazione*: per iniziare una fase di scrematura abbiamo utilizzato due principali filtri: <span style="color:#9bb1ff"><b>Lowe's Ratio</b></span> e <span style="color:#9bb1ff"><b>Distanza Euclidea</b></span>.  

- **Filtraggio con RANSAC**  
*Spiegazione*: <span style="color:#9bb1ff"><b>RANSAC</b></span> è un algoritmo che ci permette di rendere la nostra matrice omografica più robusta attraverso l'eliminazione dei valori anomali (<span style="color:#f4d35e"><b>outlier</b></span>) dal nostro insieme di match tra feature.  

- **Rifinimento con ECC**  
*Spiegazione*: È un algoritmo di registrazione di immagini basato sull’intensità, che cerca di massimizzare la correlazione tra due immagini per trovare la migliore trasformazione geometrica. A differenza di ORB o SIFT cerca di allineare due immagini trovando la trasformazione che massimizza la somiglianza globale tra i loro pixel, senza usare punti chiave o feature.  
<span style="color:#f7b267"><b>I risultati ottenuti finora non lo prevedono; l'abbiamo testato e tenuto in considerazione ma non ancora applicato</b></span>.  

- **Verifica della coregistrazione**  
*Spiegazione*: una volta eseguito il processo sopra indicato abbiamo valutato i risultati ottenuti attraverso dei criteri standard come <span style="color:#f4d35e"><b>Mutual Information</b></span> e <span style="color:#f4d35e"><b>Mean Square Error</b></span>. Inoltre abbiamo ritenuto valido il confronto degli istogrammi dell'immagine label-free in scala di grigi con la differenza assoluta ottenuta tra label-free e stained allineata. Questo processo confronta il numero di pixel neri presenti nell'immagine originale con quella allineata (un pixel perfettamente allineato assume il valore 0). E' evidentemente un metodo immediato per osservare il miglioramento.  

- **Ritaglio finale**  

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: applicando il programma sviluppato a una serie di immagini abbiamo osservato come molte di esse venissero allineate in maniera sufficientemente valida, ma alcune no. Questa discontinuità nei risultati è dovuta alla presenza dello sfondo e da come esso venga mal gestito dalla normalizzazione, causando una (presunta) creazione di false feature e di conseguenza non permettendo agli algoritmi di funzionare correttamente.  
<span style="color:#71c78c"><b>Possibile soluzione</b></span>: due possibili soluzioni possono essere o il riconoscimento del bordo della cellula e un conseguente isolamento dello sfondo, sul quale non applicheremmo la normalizzazione, oppure l'applicazione di algoritmi differenti come ECC oppure Intensity-based.  
<span style="color:#f7b267"><b>Questo problema e questa soluzione ancora non sono stati affrontati. Saranno affrontati probabilmente nella settimana del 24 Marzo</b></span>.  

## 24 Marzo
