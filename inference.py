import os
import pandas as pd
import sys

from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts import templates

## optionally uncomment and set cache path below for downloading ckpts
# os.environ['TRANSFORMERS_CACHE'] = ""
# os.environ['HF_HOME'] = ""
# os.environ['HF_DATASETS_CACHE'] = ""


def run_inference(model: str, split: str, tp_size: int, output_folder: str, max_tokens: int = 2048, max_model_len: int = 4096):
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

        sampling_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=max_tokens)
        llm = LLM(
            model=model,
            tensor_parallel_size=int(tp_size),
            max_model_len=max_model_len,
            download_dir="",  # set the download dir
            trust_remote_code=True,
        )

        outputs = llm.generate(prompts, sampling_params)
        print(f"Running inference for question type {question_type}...")
        for n, output in tqdm(enumerate(outputs)):
            generated_text = output.outputs[0].text
            generated_text_clean = generated_text.split("Question:")[0]
            out.append({"idx": n, "generation": generated_text_clean})
        out_df = pd.DataFrame(out)
        out_df.to_csv(os.path.join(output_folder, f"{split}_{model.split('/')[1]}_{question_type}.csv"))


if __name__ == "__main__":
    run_inference(*sys.argv[1:])
