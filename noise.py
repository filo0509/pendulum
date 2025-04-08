import pandas as pd
import numpy as np

# Carica il dataset originale
df = pd.read_csv("data/30gradi_38.6.csv", sep=";")

# Aggiungi rumore alla colonna 'period' (simula l'errore umano)
# ad esempio, ±0.01s (deviazione standard 0.005s per valori realistici)
np.random.seed(42)  # per riproducibilità
noise = np.random.normal(loc=0.0, scale=0.05, size=len(df))  # media 0, std dev 0.005
df["period"] = df["period"] + noise

# Salva un nuovo CSV con i dati rumorosi
df.to_csv("data/30gradi_noisy_38.6.csv", sep=";", index=False)

print("Dati con rumore salvati.")