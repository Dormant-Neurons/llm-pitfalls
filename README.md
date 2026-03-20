# Chasing Shadows: Pitfalls in LLM Security Research

[![Zenodo Artifact](https://img.shields.io/badge/Zenodo-Artifact-16825d?style=flat)](https://doi.org/10.5281/zenodo.17847798)
[![NDSS Paper](https://img.shields.io/badge/NDSS'26-Paper-007ec6?style=flat)](https://www.ndss-symposium.org/wp-content/uploads/2026-f1749-paper.pdf)

This is the artifact repository for the NDSS'26 paper "Chasing Shadows: Pitfalls in LLM Security Research".

## Citation
If you used our data or want to cite our work, please use the following BibTeX entry:
```bibtex
@inproceedings{evertz-26-chasing,
    title     = {Chasing Shadows: Pitfalls in LLM Security Research},
    author    = {Evertz, Jonathan and Risse, Niklas and Neuer, Nicolai and M{\"u}ller, Andreas and Normann, Philipp and Sapia, Gaetano and Gupta, Srishti and Pape, David and Shaw, Soumya and Srivastav, Devansh and Wressnegger, Christian and Quiring, Erwin and Eisenhofer, Thorsten and Arp, Daniel and Sch{\"o}nherr, Lea},
    booktitle = {Network and Distributed System Security Symposium (NDSS)},
    year      = {2026}
}
```

## Pitfall Review Data
The review data (with omitted paper titles) can be found [here](./analysis/llm-pitfalls-data.csv).

## Living Appendix
The project website with the living appendix can be found [here](https://llmpitfalls.org).

## Reproducing the Experiments

### Set up the environment

Every experiment is contained in its own directory (e.g., experiment_a_1 for Experiment A.1). Each experiment has its own Dockerfile that contains all dependencies required to run the experiment. Simply follow the instructions below for each experiment to build the Docker image and run the experiment.
Most experiments require API keys. Insert/Update the necessary keys in the `.env.template` file in the respective experiment directory and rename it to `.env`.

> [!NOTE]
> You do not need docker necessarily. You can also just create a virtual environment, install all dependencies via pip install -r requirements, and then just run the python scripts as you need.

### Experiment A: Model Ambiguity & Surrogate Fallacy

#### A.1 Hate Detection

1. Request dataset access from https://huggingface.co/datasets/keyan96/New-Wave-Hate
   Then, after getting access, go to the tab: "Files and Versions" (next to Dataset card and Data Studio).
   Download data_ground_truth.csv.
   Save the data_ground_truth.csv file in the experiment_a_1 directory.

2. To use the repository, you need to set up the env file with the OpenAI key.

```bash
cd experiment_a_1
cp .env.template .env
nano -w .env

```

Replace the OpenAI API key in .env with your API key.

3. Build Docker and Run the Experiment

```bash
cd experiment_a_1
docker build -t experiment_a_1 .
docker run -e QUICK_CHECK=true experiment_a_1
```

You have two options:

-   `QUICK_CHECK=True`: Here, you run the experiment on 10 samples (Good to check the functionality quickly).

-   `QUICK_CHECK=False` (default) Here, you run the experiment on the full set of all samples (of the considered quarters).

4. Excepted output

-   You will get a few debug outputs.
-   The previous script will run 3 experiments for three categories.
-   The final output after running all experiments and aggregating the results should look like this (here, with QUICK_CHECK=True):

```
Found Results/results_vaccine_gpt-4-0613_gpt-4-0125-preview_gpt-4.1-2025-04-14.pck
Found Results/results_us_capitol_gpt-4-0613_gpt-4-0125-preview_gpt-4.1-2025-04-14.pck
Found Results/results_rus_ukr_gpt-4-0613_gpt-4-0125-preview_gpt-4.1-2025-04-14.pck
LLM:gpt-4-0613 & 96.67\% & 90.91\% & 100.0\% \\
LLM:gpt-4-0125-preview & 73.33\% & 100.0\% & 20.0\% \\
LLM:gpt-4.1-2025-04-14 & 86.67\% & 100.0\% & 60.0\% \\
```

The columns are: Accuracy, Precision, and Recall.
Although we just used a small number of samples, we can already see a trend as reported in the paper. To see the numbers from the paper, you need to re-run the experiment with QUICK_CHECK=False. Note that the reported and observed numbers may slightly differ, as LLM outputs are not deterministic.

#### A.2 LLM Robustness

For the evaluation of the LLM robustness experiments we used the [LLM confidentiality framework](https://github.com/LostOxygen/llm-confidentiality).

1. Add your OpenAI API key to the `.env.template` file.

2. Build the Docker and run the experiment with:

```bash
cd experiment_a_2
cp .env.template .env
nano -w .env # open and edit the file to add the API keys
docker build -t experiment_a_2 .
docker run --gpus all experiment_a_2 --llm_type <model_specifier> --strategy secret-key --attacks all --iterations 1000 --temperature 0.1 --device <cpu, cuda, or mps>
```

Replace the `<model_specifier>` with the model to evaluate. For our experiments, we used the following models:

| Model Name              | Specifier                                                                      |
| ----------------------- | ------------------------------------------------------------------------------ |
| ChatGPT 3.5 Turbo       | `gpt-3.5-turbo-1106` and `gpt-3.5-turbo-0125`                                  |
| ChatGPT 4               | `gpt-4-0314` and `gpt-4-0613 `                                                 |
| ChatGPT 4 Turbo         | `gpt-4-1106-preview`, `gpt-4-0125-preview`, and `gpt-4-turbo-2024-04-09`       |
| ChatGPT 4o              | `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, and `gpt-4o-2024-11-20`              |
|                         |
| CodeLlama 7b (Ollama)   | `codellama-7b-fp16` (no quant.) or `codellama-7b-quant-<2,3,4,5,6,8>bit`       |
| CodeLlama 7b (TheBloke) | `codellama-7b` (no quant.) or `codellama-7b-quant-<2,3,4,5,6,8>bit-bloke`      |
| Llama2 7b (Ollama)      | `llama2-7b-fp16` (no quant.) or `llama2-7b-quant-<2,3,4,5,6,8>bit`             |
| Llama2 7b (TheBloke)    | `llama2-7b-fp16-bloke` (no quant.) or `llama2-7b-quant-<2,3,4,5,6,8>bit-bloke` |
| Llama3.1 8b (Ollama)    | `llama3.1-8b-fp16` (no quant.) or `llama3.1-8b-quant-<2,3,4,5,6,8>bit`         |

To keep the runtime reasonable, we recommend changing the `--iterations` parameter to a lower value (e.g., 100) when necessary.

The results will then be printet to the console (stdout) after the experiment has finished. The results and logs will also be saved to `logs/` directory. Files ending in `_logs.txt`contain the detailed logs, while files ending in `_results.txt` contain the plain result metrics.

> [!NOTE]
> Deprecation warnings and potentially missing Huggingface tokens can be ignored as they do not affect the models used in our experiments.

### Experiment B: Data Leakage

#### B.1 Lab-Setting

1. The experiment requires a huggingface token to download the datasets from huggingface (free-of-charge). Create an account [here](https://huggingface.co/join) and generate an access token following [this](https://huggingface.co/docs/hub/security-tokens) guide. The experiments will not incur any costs.

```bash
cd experiment_b_1
cp .env.template .env
nano -w .env

```

2. Afterwards, build the Docker and run the experiment with:

```bash
docker build -t experiment_b_1 .
docker run --rm -v "$(pwd)/../plots:/app/plots" experiment_b_1
```

This will run the data leakage lab setting experiment for all three datasets (Devign, PrimeVul, DiverseVul) and all six leakage ratios (0, 20%, 40%, 60%, 80%, 100%) that were presented in the paper. The resulting plot will be saved to the `plots` directory.

> [!NOTE]
> Since the experiment uses fine-tuning, it is normal that the experiment takes a while to run. On our system with an Nvidia A100 GPU the experiment took around 5-6 days.

3. Scaled-down Version: We have implemented a `--fast` option to execute a scaled down version of the experiment (smaller model, only one dataset, less epochs, only 600 samples) which runs on a commodity desktop in 1-10 hours. Run with:

```bash
docker build -t experiment_b_1 .
docker run --rm -v "$(pwd)/../plots:/app/plots" experiment_b_1 --fast
```

4. It is also possible to run the script for one dataset and one leakage ratio only. In this case, no plot will be generated.

```python
python run_experiment.py --dataset <Devign, DiverseVul, or PrimeVul> --ratio=<0, 0.2, 0.4, 0.6, 0.8, or 1.0>
```

#### B.2 Commercial LLMs

This experiment evaluates commit-message and code completion performance across **commercial** and **local OpenAI-compatible** LLM endpoints.

We evaluate the following models:

| Backend       | Model Identifiers                                            |
| ------------- | ------------------------------------------------------------ |
| **OpenAI**    | `gpt-3.5-turbo-0125`, `gpt-4o-2024-08-06`                    |
| **Anthropic** | `claude-3-5-haiku-20241022`                                  |
| **DeepSeek**  | `DeepSeek-V3-0324`                                           |
| **Local**     | `meta-llama-3-8b-instruct`, `qwen3-14b`, `qwen2.5-coder-14b` |

##### Setup

1. The experiment requires API tokens from huggingface (free of charge), openai, deepseek and anthropic. Create a Huggingface account [here](https://huggingface.co/join) and generate an access token following [this](https://huggingface.co/docs/hub/security-tokens) guide. The experiments will not incur any costs on HuggingFace.

```bash
cd experiment_b_2
cp .env.template .env
nano -w .env

```

2. For **local models**, we used **LM Studio** to host models with an OpenAI-compatible REST API.  
   Download LM Studio [here](https://lmstudio.ai/) and set up the local API for the three local models according to [this](https://lmstudio.ai/docs/developer/core/server) Tutorial.

3. Afterwards, build the Docker and run the experiment with:

```bash
docker build -t experiment_b_2 .
docker run experiment_b_2
```

To run the experiment for a specific model:

```python
python run_experiment.py --backend <local, openai, anthropic, or deepseek> --model <one of the model identifiers from the table above>
```

### Experiment C: Context Truncation

1. The experiment requires a huggingface token to download the datasets from huggingface (free-of-charge). Create an account [here](https://huggingface.co/join) and generate an access token following [this](https://huggingface.co/docs/hub/security-tokens) guide. The experiments will not incur any costs.

```bash
cd experiment_c
cp .env.template .env
nano -w .env

```

2. Afterwards, build the Docker and run the experiment with:

```bash
docker build -t experiment_c .
docker run --rm -v "$(pwd)/../plots:/app/plots" experiment_c
```

The resulting latex table (exact latex that was used in the paper) will be saved to the `plots` directory.

### Experiment D: Model Collapse

Build the Docker and run the experiment with:

```bash
cd experiment_d
docker build -t experiment_d .
docker run --gpus all experiment_d --device <cpu, cuda, or mps> --num_generations <number_of_generations>
```

This will run the experiment on the specified device with the following default parameters:

-   Model: unsloth/Qwen2.5-Coder-0.5B
-   Dataset: bigcode/self-oss-instruct-sc2-exec-filter-50k
-   Block Size: 128

The resulting perplexity plots will be saved to the root directory of the docker image. You may need to copy them out of the docker container with
```bash
docker cp <container_id>:/app/plots/perplexity_histogram_bs128_Qwen2.5-Coder-0.5B-Instruct.png <destination_path>
```

> [!NOTE]
> Since the experiment uses fine-tuning, it is normal that the experiment in its original scope (10 generations) takes a while to run. On our system with an Nvidia L40S GPU the experiment took around 4-5 days.

To run the experiment on a reduced set of generations, use the `--num_generations` argument. For example, to run the experiment with only 1 generation:

```bash
docker run --gpus all experiment_d --device <cpu, cuda, or mps> --num_generations 1
```

This will significantly reduce the runtime of the experiment.
