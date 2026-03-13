import re
import matplotlib.pyplot as plt
import numpy as np

# Percorso al file di log
LOG_PATH = "Appunti/Scripts/log.txt"

# Espressione regolare per identificare le righe di validazione
val_regex = re.compile(
    r"\[Epoca (\d+)\] Validation: loss_G=([\d.]+) loss_D=([\d.]+)"
)

# Liste per salvare i dati
epoche = []
loss_G = []
loss_D = []

# Estrazione dal file
with open(LOG_PATH, "r", encoding="utf-8") as file:
    for line in file:
        match = val_regex.search(line)
        if match:
            epoche.append(int(match.group(1)))
            loss_G.append(float(match.group(2)))
            loss_D.append(float(match.group(3)))

# Calcola i coefficienti della retta (grado 1)
trend_G = np.polyfit(epoche, loss_G, deg=1)
trend_D = np.polyfit(epoche, loss_D, deg=1)

# Crea le y delle linee di regressione
fit_G = np.polyval(trend_G, epoche)
fit_D = np.polyval(trend_D, epoche)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(epoche, loss_G, label="Loss Generatore (G)")
plt.plot(epoche, loss_D, label="Loss Discriminatore (D)")

# Aggiunta linee di tendenza
plt.plot(epoche, fit_G, linestyle=':', label='Regressione Lineare Generatore', color='blue')
plt.plot(epoche, fit_D, linestyle=':', label='Regressione Lineare Discriminatore', color='red')

plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("Andamento delle perdite durante l'addestramento")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Appunti/Scripts/perdite_andamento.png", dpi=300, bbox_inches='tight')
plt.show()