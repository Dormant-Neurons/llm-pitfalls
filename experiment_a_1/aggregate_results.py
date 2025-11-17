# Final step to get metrics
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from llm_settings import evaluated_models


save_location = Path("./Results/")
assert save_location.exists()

results = {}
for f in save_location.rglob("*.pck"):
    print(f"Found {f}")
    with open(f, "rb") as input_file:
       results[f] = pickle.load(input_file)

dataframes = [val for val in results.values()]

# check that columns from all loaded files match
columns_list = [df.columns.tolist() for df in dataframes]
assert all(columns == columns_list[0] for columns in columns_list), "check column names in loaded files"

test_df = pd.concat(dataframes, ignore_index=True)

for cur_model in evaluated_models:
    col_name = 'Response-LLM-Label-' + str(cur_model)
    lab_name = f"LLM:{cur_model}"
    accuracy_i = accuracy_score(test_df['ground_truth'], test_df[col_name])
    prec_i = precision_score(test_df['ground_truth'], test_df[col_name])
    recall_i = recall_score(test_df['ground_truth'], test_df[col_name])
    conf = confusion_matrix(test_df['ground_truth'], test_df[col_name])
    fp_i = conf[0][1]
    tn_i = conf[0][0]
    fp_i = fp_i / (fp_i + tn_i)
    print(
        f"{lab_name} & {round(accuracy_i * 100, 2)}\% & {round(prec_i * 100, 2)}\% & {round(recall_i * 100, 2)}\% \\\\")  # & {round(fp_i * 100, 2)}\%