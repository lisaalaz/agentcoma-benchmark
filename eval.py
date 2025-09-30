import ast
import asyncio
import glob
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import uuid

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

openai_api_key = os.environ["OPENAI_API_KEY"]


class RateLimiter:
    """Simple rate limiter to control API call frequency."""

    def __init__(self, max_calls_per_minute=60):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []

    async def acquire(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]

        if len(self.calls) >= self.max_calls_per_minute:
            # Wait until we can make the next call
            wait_time = 60 - (now - self.calls[0]) + 0.1  # Small buffer
            await asyncio.sleep(wait_time)
            return await self.acquire()

        self.calls.append(now)


def parse_answer(answer, pattern: str = "so the final answer is:"):
    """Extracts number from answer."""
    if pattern in str(answer).lower():
        answer = answer.split(pattern)[-1]
        answer = answer.strip().strip("\n").strip("\\n")
        answer = answer.replace(",", "")
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0
    else:
        answer = str(answer).split("\n")[-1]
        answer = answer.strip().strip("\n").strip("\\n")
        answer = answer.replace(",", "")
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0
    return answer


def accuracy(true, pred):
    """Computes exact match accuracy."""
    is_accurate = []
    assert len(pred) == len(true)
    for p, t in tqdm(zip(pred, true), total=len(true)):
        try:
            if float(parse_answer(p)) == float(t):
                is_accurate.append(1)
            else:
                is_accurate.append(0)
        except ValueError:
            print(p)
            print(t)
            is_accurate.append(0)
    return is_accurate


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type((Exception,)))
async def evaluate_answer(question, gold_answer, response, client, rate_limiter, eval_model_id="gpt-4o-2024-08-06"):
    """Async version of evaluate_answer with rate limiting and retry logic."""
    await rate_limiter.acquire()

    eval_prompt = f"""Evaluate the following response to the question, and determine if the answer is correct given the reference. Respond only with "yes" if the answer is equivalent, or "no" if it is not.
    Question: {question}
    Response: {response}
    Reference: {gold_answer}"""

    # Convert to async call using thread pool
    loop = asyncio.get_event_loop()
    response_obj = await loop.run_in_executor(None, lambda: client.responses.create(model=eval_model_id, input=eval_prompt))
    eval_response = response_obj.output_text

    if "yes" in eval_response.lower():
        result = True
    elif "no" in eval_response.lower():
        result = False
    return result


async def process_single_evaluation(i, question, true, pred, client, rate_limiter):
    """Process a single evaluation with all its gold answers."""
    is_correct = 0
    try:
        text = pred[i].split("So the final answer is:")[1].strip()
        for ga in ast.literal_eval(true[i]):
            if await evaluate_answer(question[i], ga, text, client, rate_limiter):
                is_correct = 1
                break
        if not is_correct:
            for ga in ast.literal_eval(true[i]):
                if await evaluate_answer(question[i], ga, pred[i], client, rate_limiter):
                    is_correct = 1
                    break
    except:
        for ga in ast.literal_eval(true[i]):
            if await evaluate_answer(question[i], ga, pred[i], client, rate_limiter):
                is_correct = 1
                break
    return i, is_correct


async def check_text(question, true, pred, max_concurrent=10, max_calls_per_minute=180):
    """Parallelized LLM-as-a-judge eval with rate limiting."""
    assert len(question) == len(true) == len(pred)
    client = OpenAI(api_key=openai_api_key)
    rate_limiter = RateLimiter(max_calls_per_minute)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(i):
        async with semaphore:
            return await process_single_evaluation(i, question, true, pred, client, rate_limiter)

    # Create tasks for all evaluations
    tasks = [process_with_semaphore(i) for i in range(len(pred))]

    # Process tasks with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Evaluating") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])
    return [result[1] for result in results]


def check_text_wrapper(question, true, pred, max_concurrent=10, max_calls_per_minute=200):
    """Wrapper to run async check_text from sync code."""
    return asyncio.run(check_text(question, true, pred, max_concurrent, max_calls_per_minute))


def discover_inference_files(output_folder, model_name, split, sampling_config=None):
    """Auto-discover inference result files based on naming pattern.

    Args:
        output_folder: Path to folder containing inference results
        model_name: Model name (e.g., 'Llama-3.2-1B-Instruct')
        split: Data split ('dev' or 'test')
        sampling_config: Specific sampling config, or None for all configs

    Returns:
        Dictionary mapping question types to file paths, or list of dicts if multiple configs
    """
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    if sampling_config:
        # Single config
        pattern = f"{split}_{model_short}_*_{sampling_config}.csv"
        files = glob.glob(os.path.join(output_folder, pattern))

        data_paths = {}
        for file in files:
            filename = os.path.basename(file)
            if "_commonsense_" in filename:
                data_paths["commonsense_inference_data_path"] = file
            elif "_math_" in filename:
                data_paths["math_inference_data_path"] = file
            elif "_composition_" in filename:
                data_paths["composition_inference_data_path"] = file
        return data_paths
    else:
        # All configs
        pattern = f"{split}_{model_short}_*.csv"
        files = glob.glob(os.path.join(output_folder, pattern))

        configs = set()
        for file in files:
            parts = os.path.basename(file).replace(".csv", "").split("_")
            if len(parts) >= 4:
                configs.add(parts[-1])  # sampling config is last part

        return [discover_inference_files(output_folder, model_name, split, config) for config in sorted(configs)]


def evaluate_all_question_types_and_print_report(data_paths, data_split, output_folder=None, model_config_name=None):
    """Evaluates all commonsense-only, math-only and compositional answers of a
      model, saves 0-1 scores for each answer for each csv file (overwrites the
      file), and computes the compositionality gap between the proportion of
      responses where both math and commonsense are solved correctly in isolation
      and the compositional accuracy. Prints out a performance report with all
      metrics.

    Args:
      data_paths: a dictionary of string key-value pairs. Must have the
        following keys:
          'commonsense_inference_data_path': the path to the csv file containing
            the commonsense-only answers.
          'math_inference_data_path': the path to the csv file containing the
            math-only answers.
          'composition_inference_data_path': the path to the csv file containing
            the compositional answers.
      data_split: 'dev' or 'test'.
      output_folder: (optional) path to folder where to save accuracy report.
    """
    print("Starting evaluations...")
    ground_truths = pd.read_csv(f"data/agentcoma_{data_split}.csv")
    accs = {"commonsense": [], "math": [], "composition": []}

    for path in data_paths.keys():
        answers = pd.read_csv(data_paths[path])
        assert len(ground_truths) == len(answers), f"""The lengths of the outputs and the ground truths in {path} do not correspond.
You have set to evaluate on the {data_split} split. Are you sure you are providing an output file for the correct split?"""
        if "commonsense" in path:
            print("Evaluating commonsense-only answers...")
            acc = check_text_wrapper(
                ground_truths["question_commonsense"],
                ground_truths["answers_commonsense"],
                answers["generation"],
            )
            accs["commonsense"] = acc
        if "math" in path:
            print("Evaluating math-only answers...")
            acc = accuracy(ground_truths["answer_math"], answers["generation"])
            accs["math"] = acc
        elif "composition" in path:
            print("Evaluating compositional answers...")
            acc = accuracy(ground_truths["answer_composition"], answers["generation"])
            accs["composition"] = acc

        answers["accuracy"] = acc
        answers.to_csv(data_paths[path])

    commonsense_acc = np.mean(accs["commonsense"])
    math_acc = np.mean(accs["math"])
    both_steps_acc = len([x for x in zip(accs["commonsense"], accs["math"]) if x == (1, 1)]) / len(accs["math"])
    comp_acc = np.mean(accs["composition"])
    comp_gap = comp_acc - both_steps_acc

    accuracy_report = f"""
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  PERFORMANCE REPORT FOR {data_split} DATA SPLIT
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Commonsense-only accuracy: {commonsense_acc}
  ---
  Math-only accuracy: {math_acc}
  ---
  Percentage of questions where both reasoning steps are solved correctly in isolation: {both_steps_acc}
  ---
  Compositional accuracy: {comp_acc}
  ---
  Compositionality gap: {comp_gap}
  """

    print(accuracy_report)

    if output_folder is not None:
        print("Saving evaluation report...")
        uid = uuid.uuid4()
        with open(os.path.join(output_folder, f"report_{uid.hex}.txt"), "w") as text_file:
            text_file.write(accuracy_report)
        print("Done.")

    # Return metrics for CSV compilation
    return {
        "commonsense_acc": commonsense_acc,
        "math_acc": math_acc,
        "both_steps_acc": both_steps_acc,
        "comp_acc": comp_acc,
        "comp_gap": comp_gap,
    }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python eval.py <output_folder> <model_name> <split> [sampling_config]")
        print("Example: python eval.py outputs/ meta-llama/Llama-3.2-1B-Instruct test")
        print("Example: python eval.py outputs/ meta-llama/Llama-3.2-1B-Instruct test top_p_90")
        sys.exit(1)

    output_folder, model_name, split = sys.argv[1:4]
    sampling_config = sys.argv[4] if len(sys.argv) > 4 else None

    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    results = []

    if sampling_config:
        # Evaluate single config
        data_paths = discover_inference_files(output_folder, model_name, split, sampling_config)
        if not data_paths:
            print(f"No files found for {model_name} {split} {sampling_config}")
            sys.exit(1)
        print(f"Evaluating {sampling_config} configuration...")
        metrics = evaluate_all_question_types_and_print_report(data_paths, split, output_folder)
        results.append({"model_config": f"{model_short}_{sampling_config}", **metrics})
    else:
        # Evaluate all configs
        all_configs = discover_inference_files(output_folder, model_name, split)
        if not all_configs:
            print(f"No files found for {model_name} {split}")
            sys.exit(1)

        for i, data_paths in enumerate(all_configs):
            config_name = list(data_paths.values())[0].split("_")[-1].replace(".csv", "")
            print(f"\n{'=' * 50}")
            print(f"Evaluating configuration: {config_name}")
            print(f"{'=' * 50}")
            metrics = evaluate_all_question_types_and_print_report(data_paths, split, output_folder)
            results.append({"model_config": f"{model_short}_{config_name}", **metrics})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_folder, f"evaluation_results_{model_short}_{split}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")
