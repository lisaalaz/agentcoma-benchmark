import os
import pandas as pd
import sys

from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts import templates

## Model configurations
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
    # Add more model IDs here
]

## Sampling configurations for parameter sweeping
SAMPLING_CONFIGS = {
    "greedy": {"temperature": 0.0, "top_k": 1},
    # "top_p_50": {"temperature": 0.0, "top_p": 0.5, "top_k": -1},
    "top_p_90": {"temperature": 0.0, "top_p": 0.9, "top_k": -1},
    # "min_p_01": {"temperature": 0.0, "min_p": 0.01, "top_k": -1},
    "min_p_05": {"temperature": 0.0, "min_p": 0.05, "top_k": -1},
}

## optionally uncomment and set cache path below for downloading ckpts
# os.environ['TRANSFORMERS_CACHE'] = ""
# os.environ['HF_HOME'] = ""
# os.environ['HF_DATASETS_CACHE'] = ""


def run_inference(
    model: str, split: str, tp_size: int, output_folder: str, max_tokens: int = 2048, max_model_len: int = 4096, sampling_config: str = "greedy"
):
    """Runs inference on all question types in AgentCoMa (commonsense-only,
    math-only, composition). Saves respective CSV outputs in the OUTPUT_FOLDER.

    Args:
        model: The Hugging Face model indentifier.
        split: Dataset split. Options: 'dev' or 'test'.
        tp_size: Size of tensor parallel.
        output_folder: Path to the output folder where to save the responses.
    """
    assert split in ["test", "dev"], 'Allowed split values are ["test", "dev"]'
    question_types = ["commonsense", "math", "composition"]

    print(f"Loading {split} split...")
    data = pd.read_csv(f"data/agentcoma_{split}.csv")
    print("Data loaded!")

    for question_type in question_types:
        out = []
        prompts = [templates[f"few_shot_cot_{question_type}"].format(question=q) for q in data[f"question_{question_type}"]]

        sampling_params = SamplingParams(max_tokens=max_tokens, **SAMPLING_CONFIGS[sampling_config])
        llm = LLM(
            model=model,
            tensor_parallel_size=int(tp_size),
            max_model_len=max_model_len,
            download_dir="",  # set the download dir
            trust_remote_code=True,
            gpu_memory_utilization=0.4,  # Use 50% of available GPU memory
        )

        outputs = llm.generate(prompts, sampling_params)
        print(f"Running inference for question type {question_type}...")
        for n, output in tqdm(enumerate(outputs)):
            generated_text = output.outputs[0].text
            generated_text_clean = generated_text.split("Question:")[0]
            out.append({"idx": n, "generation": generated_text_clean})
        out_df = pd.DataFrame(out)
        out_df.to_csv(os.path.join(output_folder, f"{split}_{model.split('/')[1]}_{question_type}_{sampling_config}.csv"))


def run_sampling_sweep(model: str, split: str, tp_size: int, output_folder: str, max_tokens: int = 2048, max_model_len: int = 4096):
    """Runs inference with all sampling configurations for comparison.

    Usage: python inference.py sweep model_name split tp_size output_folder [max_tokens] [max_model_len]
    Example: python inference.py sweep meta-llama/Llama-3.2-1B-Instruct dev 1 outputs/
    """
    for config_name in SAMPLING_CONFIGS.keys():
        print(f"Running inference with {config_name} sampling...")
        run_inference(model, split, tp_size, output_folder, max_tokens, max_model_len, config_name)


def run_multi_model_sweep(split: str, tp_size: int, output_folder: str, max_tokens: int = 2048, max_model_len: int = 4096):
    """Runs inference with all models and all sampling configurations.

    Usage: python inference.py multi_sweep split tp_size output_folder [max_tokens] [max_model_len]
    Example: python inference.py multi_sweep dev 1 outputs/
    """
    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Running inference for model: {model}")
        print(f"{'=' * 60}")
        run_sampling_sweep(model, split, tp_size, output_folder, max_tokens, max_model_len)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        run_sampling_sweep(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "multi_sweep":
        run_multi_model_sweep(*sys.argv[2:])
    else:
        run_inference(*sys.argv[1:])
