"""helper library to plot histograms of perplexities for different models"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl          #  ←  NEU (für Kompression)
import torch
import seaborn as sns

os.makedirs("plots", exist_ok=True)
mpl.rcParams.update(mpl.rcParamsDefault)

perplexity_dict = torch.load(
    "generated_datasets/perplexity_dict_bs128_Qwen2.5-Coder-0.5B.pt"
)
all_perplexities = torch.load(
    "generated_datasets/all_perplexities_bs128_Qwen2.5-Coder-0.5B.pt"
)

bins = torch.logspace(
    torch.log10(torch.tensor(min(all_perplexities))),
    torch.log10(torch.tensor(max(all_perplexities))),
    steps=401,
)

custom_colors = [
    "#2369BD",  # darker blue
    "#006BA4",  # dark blue
    "#5F9ED1",  # light blue
    "#A2C8EC",  # very light blue
    "#ABABAB",  # gray
    "#898989",  # dark gray
    "#898989",  # darker gray
    "#FFBC79",  # light orange
    "#FF800E",  # orange
    "#C85200",  # dark orange
    "#A9373B",  # dark red
]

cb_palette = sns.color_palette(custom_colors, n_colors=10, as_cmap=True)
sns.set_palette(cb_palette)
sns.set_style("whitegrid")

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 22,
    "font.weight": "bold",             # <--- Make default font bold
    "axes.labelsize": 22,
    "axes.labelweight": "bold",        # <--- Bold axis labels
    "axes.titlesize": 20,
    "axes.titleweight": "bold",        # <--- Bold title
    "legend.fontsize": 17,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.major.width": 2,          # Optional: thicker ticks
    "ytick.major.width": 2,
    "pdf.compression": 9
})

plt.figure(figsize=(10, 6))
for name, perplexities in perplexity_dict.items():
    sns.histplot(
        perplexities,
        bins=bins,
        stat="density",
        label=name,
        element="step",
        alpha=0.4,
    )

plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-5, 1)

plt.xlabel("Perplexity", fontweight='bold')
plt.ylabel("Probability", fontweight='bold')
plt.title("")
#plt.title("Perplexity of generated datapoints for blocksize of 128")
plt.legend(loc="upper right")

for spine in plt.gca().spines.values():
    spine.set_color('black')

plt.tight_layout()
plt.savefig("plots/perplexity_histogram_bs128.pdf")
plt.savefig("plots/perplexity_histogram_bs128.png")
plt.show()
