# Timeline Progetto Virtual-Staining 2025
Purtroppo ho pensato una settimana troppo tardi di creare questo file per tenere traccia in maniera rigorosa e più dettagliata del lavoro svolto, quindi nel lasso temporale tra il 14/03 e il 19/03 saranno presenti tutte le cose fatte (con problemi riscontrati e soluzioni attuate).  
Per riscontri dettagliati vedere i notebook.  

Legenda colori:  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#e15c64; border-radius: 2px;"></span> <b style="margin-left: 8px;">Problemi riscontrati</b> – criticità rilevate nel processo  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#71c78c; border-radius: 2px;"></span> <b style="margin-left: 8px;">Soluzioni attuate</b> – strategie o interventi correttivi  
<span style="display:inline-block; width: 14px; height: 14px; background-color:#f7b267; border-radius: 2px;"></span> <b style="margin-left: 8px;">Futura implementazione/risultati ottenuti</b> – idee o ottimizzazioni da valutare  
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
<span style="color:#e15c64"><b>Problema riscontrato</b></span>: eseguendo diversi test su sotto immagini abbiamo notato come si creassero degli artefatti dovuti alla normalizzazione in zone di sfondo o genericamente bianche e grandi (rispetto alla dimensione delle immagini). Questo induceva l'algoritmo <span style="color:#9bb1ff"><b>SIFT</b></span> (o ORB) a cercare delle feature in queste zone, rischiando di conseguenza di trovare feature inesistenti.
<span style="color:#71c78c"><b>Soluzione proposta</b></span>: abbiamo optato, seguendo il consiglio dei docenti, per lo sviluppo di una maschera che permettesse di evitare di normalizzare eventuali zone critiche e di ricercare in esse feature. In sintesi abbiamo applicato la maschera sia alla normalizzazione che a SIFT. 

- **Abbiamo valutato diversi metodi per creare la maschera**
*Spiegazione*: abbiamo valutato principalmente tre metodi: il <span style="color:#f4d35e"><b>Flood Fill from Edges</b></span>, <span style="color:#f4d35e"><b>Connected Components</b></span> e infine un possibile <span style="color:#f4d35e"><b>White pixel counter</b></span>. 

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: il white pixel counter, per via della sua eccessiva semplicità non otteneva i risultati desiderati, e di conseguenza l'abbiamo scartato come opzione; il flood fill from edges poteva essere una valida opzione ma, in corso d'opera, abbiamo deciso di considerare all'interno della maschera anche le zone bianche interne all'immagine.
<span style="color:#71c78c"><b>Soluzione attuata</b></span>: utilizzo di connected components per l'ottenimento della maschera finale.

- **Binarizzazione dell'immagine per rilevamento dei componenti**
*Spiegazione*: binarizzando l'immagine imponendo una soglia di 230 (valore ottenuto sperimentalmente e osservando i valori di intensità all'interno dell'immagine) abbiamo ottenuto una matrice binaria che descrive una maschera primitiva da filtrare successivamente (per via dei troppi dettagli trascurabili)

- **Inizio dell'analisi dei componenti**
*Spiegazione*: una volta ottenuta la matrice binaria abbiamo filtrato le zone più piccole e ininfluenti considerando come criterio il fatto che lo sfondo debba essere abbastanza grande e omogeneo; in questo modo filtriamo tutte le zone più piccole presenti o non omogenee.

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: come capiamo se le zone sono omogenee?
<span style="color:#71c78c"><b>Soluzione attuata</b></span>: attraverso un soglia di deviazione standard possiamo filtrare gruppi candidati alla partecipazione della maschera. I gruppi che presentano rumore o poco omogenei saranno scartati; questo ci permette inoltre di mantenere zone interne della cellula contenenti informazioni.

## 25 Marzo
- **Creazione script per ottenere la maschera dell'immagine originale (20k x 20k)**
*Spiegazione*: applicare l'algoritmo per la creazione della maschera sull'immagine originale non era possibile per via delle risorse hardware insufficienti, di conseguenza abbiamo diviso l'immagine in _n_ sotto immagini e applicato ad esse l'algoritmo per la maschera. Dividendo l'immagine originale in quarti (ad esempio) abbiamo ricavato nella prima riga tre quadrati (\[0, 0.5\*xTot\], \[0.25\*xTot, 0.75\*xTot\], \[0.5\*xTot, xTot\]), in questo modo abbiamo ottenuto, sovrapponendo attraverso un AND logico le rispettive maschere, una robustezza maggiore e una sovrapposizione corretta.

## 26 Marzo
- **Applicazione maschera in fase di allineamento**
*Spiegazione*: dopo aver rifinito l'algoritmo per la creazione della maschera e testato ulteriori parametri (_threshold_ e _stddev\_threshold_) l'abbiamo applicato alle sotto immagini dell'immagine originale e osservato come in realtà <span style="color:#f7b267"><b>il miglioramento ottenuto tramite l'ausilio della maschera è veramente minimo</b></span>.

- **Creazione griglia senza sovrapposizione**
*Spiegazione*: abbiamo creato un breve script per ottenere un numero _n_ di coppie di sotto immagini label-free/stained. <span style="color:#f7b267"><b>La griglia applicata al momento è abbastanza elementare e non presenta elementi di ridondanza (al contrario della griglia per la maschera fatta il 25 Marzo) nel posizionamento dei quadrati. In futuro sarebbe meglio introdurre ridondanza per aumentare la robustezza</b></span>.

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: volevamo costruire il nostro dataset costituito dalle sotto immagini dell'immagine originale, ma applicando lo script sopra citato abbiamo notato come le coppie di immagini non fossero (ovviamente) coregistrate, ciò è dovuto al fatto che il processo di coregistrazione sarebbe stato applicato dopo.
<span style="color:#71c78c"><b>Soluzione proposta</b></span>: possiamo applicare una coregistrazione meno raffinata giusto per avere un miglior punto di partenza.

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: ECC sull'immagine 20k x 20k non ha effetto.
<span style="color:#71c78c"><b>Soluzione proposta</b></span>: applicare ECC su quarti (o sedicesimi) dell'immagine.

<span style="color:#e15c64"><b>Problema riscontrato</b></span>: ECC non ha effetto nemmeno su sotto porzioni dell'immagine
<span style="color:#71c78c"><b>Soluzione attuata</b></span>: Non facciamo l'allineamento iniziale e ci limitiamo a eseguire la coregistrazione su ogni sotto immagine.

