"""short helper script to count the occurence of each pitfall for every category in the csv file"""
import pandas as pd

df = pd.read_csv("llm-pitfalls-data.csv")

pitfalls = [""
    "P1 - Data Poisoning",
    "P2 - Label Inaccuracy",
    "P3 - Data Leakage",
    "P4 - Model Collapse",
    "P5 - Spurious Correlations",
    "P6 - Context Truncation",
    "P7 - Prompt Sensitivity",
    "P8 - Surrogate Fallacy",
    "P9 - Model Ambiguity"]

categories = [
    "Does not apply",
    "Not present",
    "Unclear from text",
    "Likely present",
    "Likely present (but discussed)",
    "Partly present",
    "Partly present (but discussed)",
    "Present",
    "Present (but discussed)"
]

# count the occurence of each pitfall for every category
for pitfall in pitfalls:
    TOTAL_COUNT = 0
    counts = df[pitfall].value_counts()
    print(f"{pitfall}:")

    for category in categories:
        count = counts.get(category, 0)
        print(f"  {category}: {count}")
        TOTAL_COUNT += count

    print(f"Total: {TOTAL_COUNT}\n")
    print()
