import logging
import re
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from core.file_handler import Experiment

UNKNOWN_THRESHOLD = 20
LOGGER = logging.getLogger(__name__)


def find_answer(text, letter):
    pattern = re.compile(
        f"^(?:Final answer: )?Answer:\s?(<{letter}>|{letter})(?:\n|$)", re.MULTILINE
    )
    matches = pattern.findall(text)
    found_match = len(matches) > 0
    if f"correct answer is most likely to be ({letter}" in text:
        found_match = True
    final_line = text.strip().splitlines()[-1]
    # Sometimes the final line says: "Final answer when all evidence is considered: Answer: <A|B>"
    if not found_match and "final answer" in final_line.lower():
        try:
            answer = final_line.split(": ")[-1]
            answer = (
                answer.replace("<", "")
                .replace(">", "")
                .replace("(", "")
                .replace(")", "")
            )
            if answer.strip() == letter:
                found_match = True
        except IndexError:
            pass
    return found_match


def find_inconclusive(text):
    pattern = re.compile(
        r"^Answer: (<A\|B>|Insufficient|Unable|Unclear|Can\'t determine|\?|Inconclusive|I do not have sufficient evidence).*?",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    found_match = len(matches) > 0
    print("Bhakti unknown", matches)
    return found_match


def get_complete_rows(df, complete_col, ensure_complete=True):
    if complete_col not in df.columns:
        raise ValueError(f"no complete column {complete_col}")

    len_before = len(df)
    df = df[df[complete_col]]
    len_after = len(df)

    if df.empty:
        raise ValueError("no complete rows")

    if len_before != len_after:
        warning_string = f"{len_before - len_after} incomplete"
        logging.warning(f"WARNING: {warning_string}")

        if ensure_complete:
            raise ValueError(warning_string)
    return df


def is_human_judgement(judgement: str):
    return "Confidence correct: " in judgement.strip().splitlines()[-1]


def is_human_correct(judgement: str):
    last_line = judgement.strip().splitlines()[-1]
    answer = last_line.split(": ")[1]
    return float(answer) > 0.5

def func_correct_ab(judgement, swap=False, debug=False):
    # This stores the extracted verdict from judge.
    # Correct column is true when it matches answer in correct answer column (majority)
    # False when it matches answer in incorrect column (minority)
    # Tie and unknown if it calls a tie or if answer cant be parsed respectively.
    judgement = judgement.strip()
    if len(judgement) > 3:
        if is_human_judgement(judgement):
            return is_human_correct(judgement)
        if find_answer(judgement, "A"): # Majority
            return True if not swap else False
        elif find_answer(judgement, "B"): # Minority
            return False if not swap else True
        elif find_answer(judgement, "Tie"):
            print("Bhakti this is tie")
            return "Tie"
        elif find_inconclusive(judgement):
            return "Unknown"
        else:
            if debug:
                print(judgement)
                print("======================\n\n")
                user_input = input("Do you want to see more? (y/n)")
                if user_input == "n":
                    raise StopIteration("User stopped")
            return "Unknown"
    else:
        # for logprob judge
        if "A)" in judgement or judgement == "A":
            return True if not swap else False
        elif "B)" in judgement or judgement == "B":
            return False if not swap else True
        else:
            return "Unknown"

def fix_df_nvotes(df):
    # TODO remove temporary fix for broken data in future
    if "complete_judge" not in df.columns and "complete_judge1" in df.columns:
        df["complete_judge"] = df["complete_judge1"]
        df["answer_judge"] = df["answer_judge1"]
    if "complete_judge" not in df.columns and "complete_judge_llm" in df.columns:
        df["complete_judge"] = df["complete_judge_llm"]
    return df


def get_accuracy(df, swap, debug=False, n_votes=1, verbose=False):
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df, encoding="utf-8")
    full = len(df)
    correct_columns = []
    df = fix_df_nvotes(df)
    for n_vote in range(n_votes):
        n_vote = n_vote if n_vote > 0 else ""
        complete_column = f"complete_judge{n_vote}" 
        judge_column = f"answer_judge{n_vote}" # complete answer from judge
        correct_column = f"correct{n_vote}"
        correct_columns.append(correct_column)
        df = get_complete_rows(df, complete_column)
        df[correct_column] = df[judge_column].apply(
            func_correct_ab, swap=swap, debug=debug 
        )
        if verbose:
            accuracy = (df[correct_column] == True).sum() / full
            print(f"Accuracy {n_vote}: {accuracy}")
    df_tmp = df.copy()
    # These correct and incorrect metrics don't make much sense for us as we dont have a right answer.
    # df_tmp["correct_true_count"] = df_tmp[correct_columns].apply(
    #     lambda row: (row == True).sum(), axis=1
    # )
    # df_tmp["correct_false_count"] = df_tmp[correct_columns].apply(
    #     lambda row: (row == False).sum(), axis=1
    # )
    df_tmp["correct_tie_count"] = df_tmp[correct_columns].apply(
        lambda row: (row == "Tie").sum(), axis=1
    )
    df_tmp["correct_stance_count"] = df_tmp[correct_columns].apply(
    lambda row: ((row == True) | (row == False)).sum(), axis=1
    ) # Either way majority/minority a stance is a stance as long as it is not unknown or tie.
    df_tmp["correct_unknown_count"] = df_tmp[correct_columns].apply(
        lambda row: (row == "Unknown").sum(), axis=1
    )
    # df_tmp["correct_voted"] = (
    #     df_tmp["correct_true_count"] > df_tmp["correct_false_count"]  # Majority > Minority
    # )
    count_unknown = df_tmp["correct_unknown_count"].sum()
    count_tie = df_tmp["correct_tie_count"].sum()
    count_stance = df_tmp["correct_stance_count"].sum()
    # majority_count = (df_tmp["correct_true_count"] == True).sum()
    # minority_count = (df_tmp["correct_false_count"] == False).sum()
    # accuracy = majority_count / full
    # accuracy_N = df_tmp.groupby("question")["correct_voted"].all().mean()
    #return accuracy, accuracy_N, count_unknown, count_tie, full, df, majority_count, minority_count
    return count_tie, count_stance, count_unknown, full, df

def score_file(
    filename: Path,
    swap: bool = False,
    method: str = None,
    model: str = None,
    dataset: str = None,
    results_file: Path = None,
    resave_df: bool = False,
    verbose: bool = False,
    debug: bool = False,
    n_votes: int = 1,
):
    count_tie, count_stance, count_unknown, full, df  = get_accuracy(
        filename, swap, debug=debug, n_votes=n_votes, verbose=verbose
    )
     # TODO test this whole block when n_votes>1.
    unknown_proportion = 100 * count_unknown / full / n_votes
    tie_proportion = 100 * count_tie / full / n_votes
    stance_proportion = 100 * count_stance/ full / n_votes
    if unknown_proportion > UNKNOWN_THRESHOLD:
        raise ValueError(
            f"WARNING: {unknown_proportion} unknown proportion ({count_unknown} out of {full}))"
        )

    results = pd.DataFrame(
        {
            "method": [method],
            "tie_proportion": [tie_proportion],
            "stance_proportion": [stance_proportion],
            "unknown_proportion": [unknown_proportion],
            "tie_count": [count_tie],
            "stance_count": [count_stance],
            "unknown_count": [count_unknown],
            "dataset": [dataset],
            "model": [model],
            "swap": [swap],
            "n_votes": [n_votes],
            "unknown_proportion": [unknown_proportion],
            "num_matches": [full],
        }
    )
    if verbose:
        print(results.round(3).to_markdown(index=False))
    if results_file is not None:
        if results_file.exists():
            print(f"Appending to {results_file}")
            results.to_csv(results_file, mode="a", header=False, index=False)
        else:
            print(f"Writing to {results_file}")
            results.to_csv(results_file, mode="w", header=True, index=False)
    if resave_df:
        df.to_csv(filename, index=False)
        print(f"SAVED DF to {filename}")
    return results


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(
    cfg: DictConfig,
):
    verbose = cfg.logging == "DEBUG"
    exp_dir = Path(cfg.exp_dir)
    experiment = Experiment(
        exp_dir,
        cfg.method,
        cfg.method_type,
        cfg.use_intermediary,
    )

    # using exp_suffix to easily run differing number of rounds with judge
    exp_suffix = f"_{cfg.round_limit}rounds" if cfg.round_limit is not None else ""

    filename_judgement = experiment.get_judge_filename(
        cfg.judge_name, seed=cfg.seed, swap=False, exp_suffix=exp_suffix
    )
    filename_swap_judgement = experiment.get_judge_filename(
        cfg.judge_name, seed=cfg.seed, swap=True, exp_suffix=exp_suffix
    )
    results_file = exp_dir / cfg.results_file_name

    dfs = []
    for filename, swap in zip(
        [filename_judgement, filename_swap_judgement], [False, True]
    ):
        df = score_file(
            filename,
            swap=swap,
            method=cfg.method,
            model=cfg.judge_name,
            dataset=cfg.dataset,
            n_votes=cfg.n_votes,
            debug = cfg.debug,
            verbose = cfg.verbose,
            resave_df = cfg.resave_df
        )
        dfs.append(df)

    df = pd.concat(dfs)
    df["method"] = str(experiment)
    df["method_type"] = cfg.method
    df["debate_type"] = cfg.method_type if cfg.method == "debate" else None
    df["consultant_type"] = cfg.method_type if cfg.method == "consultancy" else None
    df["use_intermediary"] = cfg.use_intermediary
    df["seed"] = cfg.seed
    df["exp_suffix"] = exp_suffix

    if verbose:
        print(df.to_markdown())
    df.to_csv(results_file, mode="w", index=False) # Should this be in append mode?
    print(f"Saved ALL results to {results_file}")
    # both ways
    print(f"overall tie count: {df['tie_count'].sum()}")
    print(f"overall stance_count: {df['stance_count'].sum()}")
    print(f"overall unknown_count: {df['unknown_count'].sum()}")
    print(f"overall matches: {df['num_matches'].sum()}")
    # TODO add an overall metric for decisiveness, does the verdict change when order is swapped

    return df['tie_count'],df['stance_count'],df['unknown_count']


if __name__ == "__main__":
    main()
