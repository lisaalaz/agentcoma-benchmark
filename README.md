
# <img src="https://agentcoma.github.io/images/agent.png" alt="Agent icon" width="65"/> AgentCoMa    

[**Paper**](https://arxiv.org/abs/2508.19988) | [**Data**](https://huggingface.co/datasets/LisaAlaz/AgentCoMa) | [**Leaderboard**](https://agentcoma.github.io/) 

Welcome to the official repository for [AgentCoMa: A Compositional Benchmark Mixing Commonsense and
Mathematical Reasoning in Real-World Scenarios](https://arxiv.org/abs/2508.19988).

See instructions below to submit to the [**Leaderboard**](https://agentcoma.github.io/)✨

AgentCoMa is an **Agent**ic **Co**mmonsense and **Ma**th benchmark where each compositional task requires both commonsense and mathematical reasoning to be solved. The tasks are set in real-world scenarios: *house working*, *web shopping*, *science experiments*, *smart assistant* and *travel agent*. The benchmark is designed to test the mixed-type compositional reasoning abilities of LLMs. Contemporary LLMs perform well on commonsense and math reasoning in isolation, but are far less effective at solving AgentCoMa tasks that require their composition. See some dev set example questions below.

<img src="https://agentcoma.github.io/images/question_examples.svg" alt="Question examples" width="1000"/>

For each compositional task, we also provide its underlying reasoning steps as individual questions. Performance on AgentCoMa is measured as the *compositionality gap* — i.e., the difference between the accuracy on the compositional tasks and the proportion of samples where all individual reasoning steps are answered correctly in isolation.

## How to evaluate on AgentCoMa and be added to the Leaderboard


### Step 1: Clone this repository and install required packages
```bash
git clone https://github.com/lisaalaz/agentcoma-benchmark.git
cd agentcoma-benchmark
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install --no-cache-dir -r requirements.txt
```

### Step 2: Download the data

Download the data from [Hugging Face](https://huggingface.co/datasets/LisaAlaz/AgentCoMa) (requires HF login + sharing email and username), and place `agentcoma_dev.csv` and `agentcoma_test.csv` in the `data/` folder.


### Step 3: Run inference on AgentCoMa

We provide a script to run inference on AgentCoMa (on both the individual reasoning steps and the compositional task) using existing Hugging Face models with the VLLM library. You may adapt this to evaluate your own framework.

To run inference with a given model, run `inference.py` followed by the arguments below (in this order):
- the HF model ID (note: we set `trust_remote_code=True` in VLLM, yet some models may still not run)
- the data split to run inference on (`dev` or `test`)
- the tensor parallel size (equal to the number of GPUs being used)
- the path to the folder where to save the model outputs

For example, to run inference on the test set with Llama 3.1 70B Instruct, with 4 GPUs, saving results in an existing 'outputs' folder located at the root, run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 inference.py meta-llama/Llama-3.1-70B-Instruct test 4 ~/outputs
```

The script automatically saves three distinct output .csv files -- for commonsense-only subquestions, math-only subquestions, and compositional questions.
These three files can be used to automatically evaluate the overall performance of the model in the next step.

### Step 4: Compute the evaluation report 

*Note: this step requires an OpenAI API key to use GPT-4o as the LLM-as-a-judge for commonsense answer evaluation.*

- Set your OPENAI_API_KEY environment variable.
- In `eval.py`, edit the `data_paths` dictionary at line 189 by adding the three .csv output files generated at the previous step (for commonsense, math, and compositional responses).
- run `eval.py` followed by the data split you want to evaluate (`dev` or `test`), and (optionally) the path to an existing output folder. For example:
```
python3 eval.py test ~/outputs
``` 
- This will append and save an additional column with the individual accuracies to each input file, and will print out an evaluation report. If an output folder path is provided, the report will also be saved as a .txt file. The evaluation report is structured as follows:
  
```
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  PERFORMANCE REPORT FOR test DATA SPLIT
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Commonsense-only accuracy: 0.956
  ---
  Math-only accuracy: 0.978
  ---
  Percentage of questions where both reasoning steps are individually correct: 0.939
  ---
  Compositional accuracy: 0.667
  ---
  Compositionality gap: -0.272
```

### Step 5: Self-report your results for inclusion in the Leaderboard

*Note: to self-report for leaderboard inclusion, your results should already be in a paper (ArXiv preprints are accepted).*

Use [this Google form](https://docs.google.com/forms/d/1Ymye0kIaKNPXoHuPSiTt6oIhgQ1phCtwMSEVeP2mA_k) to self-report results. Please only provide test set results (not dev) for the Leaderboard. In the form, you will need to provide the following scores from the evaluation report:
- Percentage of questions where both reasoning steps are individually correct
- Compositional accuracy
- Compositionality gap

## Citation

If you use this dataset, please cite our work:

````bibtex
@misc{alazraki2025agentcomacompositionalbenchmarkmixing,
      title={AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios}, 
      author={Lisa Alazraki and Lihu Chen and Ana Brassard and Joe Stacey and Hossein A. Rahmani and Marek Rei},
      year={2025},
      eprint={2508.19988},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.19988}, 
}
````
