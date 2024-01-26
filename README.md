# implementing safety via debate

## Getting Started

We have multiple composable stages for generating debates, this is to reduce re-computation. Everything interfaces through `csv` files, with most work being in creating your first `input.csv`.

Create a SECRETS file with these entries:
```
DB_USER=psqluser
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=debate
API_KEY=<KEY>
ANTHROPIC_API_KEY=<KEY>
PERSONAL_ORG=<KEY>
NYU_ORG=org-rRALD2hkdlmLWNVCKk9PG5Xq
ARG_ORG=org-4L2GWAH28buzKOIhEAb3L5aq
FARAI_ORG=org-AFgHGbU3MeFr5M5QFwrBET31
```

### Input

Each csv requires the following fields:

| Field          | Type   | Description                                       |
| -------------- | ------ | ------------------------------------------------- |
| id             | `str`  | uuid from dataset                                 |
| question       | `str`  | question (does not include answers)               |
| prompt         | `str`  | this is the n-shot prompt for a given baseline    |
| correct answer | `str`  | the answer to the question                        |
| negative answer| `str`  | misleading or incorrect answer                    |
| complete       | `bool` | used by pipelines to check stage is complete      |
| transcript     | `str`  | entire debate transcript, generated by debate     |
| answer         | `str`  | predicted answer generatedby baselines or judge   |
| cot prompt     | `str`  | Chain-of-Thought prompt for a given baseline      |


### Running Methods

```bash
# Simultaneous (Quality) on index 0 (first question)
python -m core.main \
    exp_dir='test' \
    +experiment='debate' \
    +index=0 \
    +swap=False \
    ++num_steps=2

# Debate with Claude models and BoN 4
python -m core.debate\
    exp_dir='test'\
    +experiment='debate'\
    ++correct_debater.language_model.model='claude-2.0'\
    ++incorrect_debater.language_model.model='claude-2.0'\
    ++correct_debater.BoN=4\
    ++incorrect_debater.BoN=4\
    ++anthropic_num_threads=10

# Honest Consultant + Intermediary Judge
# Honest Consultant + Intermediary Judge
python -m core.debate  \
    exp_dir='test'\
    +experiment='consultancy'\
    method_type='correct'
```

We can also call methods within other python scripts:

```python
from core.debate import main as debate
from hydra import compose, initialize

with initialize(version_base=None,
    config_path="../core/config/quality",
    job_name="exp1"):

    cfg = compose(
        config_name="config",
        overrides=["method=debate","exp_dir=test"])
    debate(cfg)
```

### Pipelines
We compose pipelines for making models work. Below we list some standard examples.

```bash
# 5-Shot
python load/mmlu.py --filename data/mmlu/5-shot.csv
python -m core.baseline filename='data/mmlu/5-shot.csv' method=0-shot model=claude-2.0
python score.py  --expdir data/mmlu --method 5-shot --model gpt-4 --dataset mmlu --save --results_file new_results.csv

# Chain-Of-Thought (GPT-4)
python load/mmlu.py --filename data/mmlu/COT.csv
python -m core.baseline --filename data/mmlu/COT.csv --method COT --model gpt-4
python score.py  --expdir data/mmlu --method COT --model gpt-4 --dataset mmlu --save --results_file new_results.csv

# Debate (Claude)
python load/mmlu.py --filename data/mmlu/debate.csv
python -m core.debate.py \
--filename data/mmlu/debate.csv \
--rollout_type sim \
--answer_type double \
--model claude-v1.3 \
--num_threads 1 \
--num_steps 2 \
--temperature 0.7

python -m core.debate --filenamedata/mmlu/debate.csv \
--model gpt-4 \
--rollout_config core/config/rollout.yaml \
--correct_debater_config core/config/debater.yaml \
--incorrect_debater_config core/config/debater.yaml \
--num_threads 3 \
--num_steps 4 \
--temperature 0.2 \
--limit 2

python core.judge.py --filename data/mmlu/debate.csv --model claude-v1.3 --num_threads 1
--filename_out data/mmlu/debate_judgement.csv \
--judge_config core/config/judge.yaml \
--rollout_config core/config/rollout.yaml \
--model gpt-4 \
--num_threads 3

python score.py  --expdir data/mmlu --method debate --model claude-v1.3 --dataset mmlu --save --results_file new_results.csv
```

## Scoring

```
python score.py --expdir [data_directory] --method [method] --model [model_name] --dataset [dataset_name] --verbose
```

Example:

```
python score.py --expdir data/mmlu --method debate --model gpt-4 --dataset mmlu --verbose
```

This command will output the mean and standard deviation of accuracy from the specified model and method on the provided dataset. Note that it expects at least 3 seeds to be run to score.

The scoring script supports a variety of optional arguments:
- `--results_file [file_name]`: Append the results to the specified file.
- `--save`: Save the results to a csv file.
- `--alt`: Swap the position of the correct answer. This changes the placement of the correct and incorrect answers in the dataset. Note: for debate methods it means a full swap of position.
- `--reverse`: Swap the names of the agents. In a debate, the agents Alice and Bob will swap positions.
- `--judge_plus`: Score the new scheme which fixes judge bias.
- `--combine`: Combine the accuracy of swapped answers. The accuracy of the original and swapped position datasets are averaged.
- `--verbose`: Print results and warnings to the terminal

For example, to score a debate with swapped answer positions, run:

```
python score.py --expdir data/mmlu --method debate --model gpt-4 --dataset mmlu --verbose --alt
```

To score with a combined accuracy of original and swapped answers, run:

```
python score.py --expdir data/mmlu --method debate --model gpt-4 --dataset mmlu --verbose --combine
```



## Datasets

Descriptions of existing datasets.

### ARC HARD
A new dataset of 7,787 genuine grade-school level, multiple-choice science questions, assembled to encourage research in advanced question-answering. The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.

When 0-shot the model achieves 94% leaving 72 questions which are answered incorrectly. From these we generate multiple debates between the correct answer and incorrect answers. This generates a dataset of over 217 binary questions.

### LUDWIG
Implicature dataset by Ruis et al.

### HellaHardSwag
0-shot the model achieves 0.8695478988249352 (4 Way). This leaves 1310 datapoints where the model is wrong, from this we create 3930 debates (binary classification).

### TruthfulQA
Multiple-choice option that tests a model's ability to identify true statements, we use MC1 (Single-true). Given a question and 4-5 answer choices, select the only correct answer.

### Ethics
We check all subsets, finding GPT-4 scores close to human performance in CommonSense, Deontology and Virtue Ethics. We choose to use Justice where the baseline model gets 83%.

### LogiQA
Sourced from expert-written questions for testing human Logical reasoning. It consists of 8,678 QA instances, covering multiple types of deductive reasoning. We generate debates but test validation over the first 1000 questions.

## Setting Up Remote Box
```bash
sudo apt update && sudo apt upgrade
sudo apt install git
sudo apt-get install tmux
git clone git@github.com:akbir/debate.git
sudo apt install -y python3-pip
pip3 install -r requirements.txt
```
Copy secrets
```bash
gcloud compute scp debate/SECRETS instance-2:~/debate --project=ageless-span-371720 --zone=us-central1-a
```
