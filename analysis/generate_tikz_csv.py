"""
This file reads stats.json and creates for each pitfall one csv file in output. 
Those are needed for the pipeline plot with tikz (pie charts).
It calculates the angles for each pie chart directly
"""

import json 
import csv
from pathlib import Path

ALL_LABELS = [
    "Not present",
    "Does not apply",
    "Unclear from text",
    "Likely present (but discussed)",
    "Likely present",
    "Partly present (but discussed)",
    "Partly present",
    "Present (but discussed)",
    "Present"
]

DONUT_LABELS = [
    "Not present",
    "Does not apply",
    "Unclear from text",
    "Likely present",
    "Partly present",
    "Present"
]

with open("output/stats.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pitfall_stats = data["pitfall_stats"]

output_dir = Path("tikz_csv")
output_dir.mkdir(exist_ok=True)

merged_angles = True

for pitfall_key, stats in pitfall_stats.items():
    pitfall_number = pitfall_key.split(".")[0]
    angles = {label: 0 for label in ALL_LABELS} # Set all labels initial 0
    # convert values to angles!
    total = sum(stats.values())
    for key in stats:
        stats[key] = (stats[key] / total)*360
    angles.update(stats)
    if abs(sum(angles.values())-360) > 1e-5:
        raise ValueError(f"Sum of angles is unequal to 360!")

    if merged_angles:
        # Hier discussed und not-discussed pitfalls zusammenfassen!
        merged_angles = {
            "Not present": angles.get("Not present", 0),
            "Does not apply": angles.get("Does not apply", 0),
            "Unclear from text": angles.get("Unclear from text", 0),
            "Likely present": angles.get("Likely present", 0) + angles.get("Likely present (but discussed)", 0),
            "Partly present": angles.get("Partly present", 0) + angles.get("Partly present (but discussed)", 0),
            "Present": angles.get("Present", 0) + angles.get("Present (but discussed)", 0)
        }

        # Write csv file
        csv_path = output_dir / f"{pitfall_number}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Angles"])
            for label in DONUT_LABELS:
                writer.writerow([merged_angles[label]])
    else:
                # Write csv file
        csv_path = output_dir / f"{pitfall_number}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Angles"])
            for label in ALL_LABELS:
                writer.writerow([angles[label]])

print("CSV file are successfully created")

