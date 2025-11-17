from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import os
import pandas as pd
import argparse
from prompts import prompt_verify_output, prompt_completion_wave_hate_original_with_terms_targets
from targets_and_terms import TermsTargetsExtractor
from llm_settings import eval_model, evaluated_models


def run_experiment(target_category: str, quick: bool = False):

    if quick:
        print("In Quick Mode")

    # Load the .env file
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OpenAI Key")

    # Input
    target_categories = ['vaccine', 'asian', 'ageism', 'mask', 'us_capitol', 'rus_ukr']
    assert target_category in target_categories, f"Target category {target_category} not supported"

    # 1. Dataset Setup
    df = pd.read_csv('data_ground_truth.csv')
    categories = list(df['category'].unique())

    # Dataset has 3 topics and 6 categories (divided covid into 4 sub-categories)
    possible_quarters = df[(df['category'] == target_category)]["quarter"].unique()

    print(f"Categories: {categories}, Chosen Category: {target_category}")
    print("Quarters possible:", possible_quarters)
    target_quarter = possible_quarters[0]
    test_quarter = possible_quarters[1]
    print("Choosing target quarter:", target_quarter, ", Test quarter: ", test_quarter)

    train_df = df[(df['category'] == target_category) & (df['quarter'] == target_quarter)]
    test_df = df[(df['category'] == target_category) & (df['quarter'] == test_quarter)].reset_index()
    print("Shapes Train, Test:", train_df.shape, test_df.shape)


    # 2. Evaluation Setup
    print(f"Evaluated Models: {evaluated_models}, Model for Label Extraction: {eval_model}")

    for cur_model in evaluated_models:
        test_df.loc[:, 'Response-'+str(cur_model)] = None
        test_df.loc[:, 'Response-LLM-Label-'+str(cur_model)] = None

    if quick:
        print("Running evaluation on a small train and test set only for debugging")
        train_df = train_df.iloc[:10,:]
        test_df = test_df.iloc[:10, :]

    termstargetextractor: TermsTargetsExtractor = TermsTargetsExtractor()
    termstargetextractor.extract_all_from_dataframe(df=train_df)

    print("Len of terms and targets:", len(termstargetextractor.targets), len(termstargetextractor.derogatory_terms))

    # 3. Run Evaluation
    for i in tqdm(range(test_df.shape[0])):
        cur_text = test_df["text"].iloc[i]
        # cur_label = test_df["ground_truth"].iloc[i]

        for cur_model in evaluated_models:
            response = prompt_completion_wave_hate_original_with_terms_targets(model=cur_model, text=cur_text,
                                                                               target_terms=termstargetextractor.targets,
                                                                               derogatory_terms=termstargetextractor.derogatory_terms)
            test_df.at[i, 'Response-'+cur_model] = response

            llm_label = prompt_verify_output(eval_model, response)
            test_df.at[i, "Response-LLM-Label-"+cur_model] = llm_label


    # 4. Save Results
    # Convert column type
    for cur_model in evaluated_models:
        test_df['Response-LLM-Label-'+cur_model] = test_df['Response-LLM-Label-'+cur_model].astype(int)

    # Get location and save results there
    save_location = Path("./Results/")
    save_location.mkdir(exist_ok=True)

    model_str = "_".join(evaluated_models)
    test_df.to_pickle(save_location / f"results_{target_category}_{model_str}.pck")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the hate detection experiment for one target category')

    parser.add_argument("-t", '--target_category', type=str, required=True,
                        help='Must be one of [vaccine|asian|ageism|mask|us_capitol|rus_ukr]')
    parser.add_argument("-q", '--quick', type=bool, default=False,
                        help='Set true if you want to run the experiment with only a small subset for quick testing')
    args = parser.parse_args()

    run_experiment(target_category=args.target_category, quick=args.quick)