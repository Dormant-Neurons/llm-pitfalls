import pandas as pd
import torch
import os
import numpy as np
import ast  # sicher zum Parsen von Listen aus Strings
import seaborn as sns

# Parameter
input_path = "perplexity_dict.csv"
output_dir = "tikz_csv"
num_bins = 401

# Sicherstellen, dass der Output-Ordner existiert
os.makedirs(output_dir, exist_ok=True)

# Einlesen
try:
    df = pd.read_csv(input_path)
except Exception as e:
    raise RuntimeError(f"Fehler beim Einlesen der Datei: {e}")

# Alle Perplexity-Listen parsen
perplexity_dict = {}
for col in df.columns:
    try:
        # Liste aus String parsen
        values = ast.literal_eval(df.at[0, col])
        if not isinstance(values, list):
            raise ValueError
        perplexity_dict[col] = torch.tensor(values)
    except Exception:
        raise ValueError(f"Konnte die Werte in Spalte '{col}' nicht als Liste parsen.")

# Alle Perplexities sammeln für globale Bins
all_perplexities = torch.cat(list(perplexity_dict.values()))

# Bins logarithmisch anlegen
bins = torch.logspace(
    torch.log10(torch.tensor(all_perplexities.min())),
    torch.log10(torch.tensor(all_perplexities.max())),
    steps=num_bins
)

# Histogramme berechnen und CSV-Dateien schreiben
for gen_name, perplexities in perplexity_dict.items():
    counts, edges = torch.histogram(perplexities, bins=bins, density=True)
    mids = (edges[:-1] + edges[1:]) / 2

    out_df = pd.DataFrame({
        "Perplexity": mids.numpy(),
        "Probability": counts.numpy()
    })

    gen_label = gen_name.replace("Generation ", "").strip()
    out_path = os.path.join(output_dir, f"Generation{gen_label}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Gespeichert: {out_path}")

colors = sns.color_palette("vlag", n_colors=10).as_hex()
print(colors)
