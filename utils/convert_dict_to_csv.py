"""helper library to convert a dictionary to a CSV file"""
import csv
import torch

perplexity_dict = torch.load(
    "generated_datasets/perplexity_dict_bs128_Qwen2.5-Coder-0.5B.pt"
)
all_perplexities_list = torch.load(
    "generated_datasets/all_perplexities_bs128_Qwen2.5-Coder-0.5B.pt"
)
all_perplexities = {"all_perplexities": all_perplexities_list}

with open("perplexity_dict.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=perplexity_dict.keys())
    w.writeheader()
    w.writerow(perplexity_dict)

with open("all_perplexities.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_perplexities.keys())
    w.writeheader()
    w.writerow(all_perplexities)
