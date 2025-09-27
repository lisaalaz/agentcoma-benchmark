import ast
import numpy as np
import os
import pandas as pd
import re
import sys
import uuid

from openai import OpenAI
from tqdm import tqdm

openai_api_key = os.environ["OPENAI_API_KEY"]




def parse_answer(answer, pattern:str="so the final answer is:"):
    """Extracts number from answer."""
    if pattern in str(answer).lower():
        answer = answer.split(pattern)[-1]
        answer = answer.strip().strip("\n").strip('\\n')
        answer = answer.replace(",", "")
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0
    else:
        answer = str(answer).split("\n")[-1]
        answer = answer.strip().strip("\n").strip('\\n')
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

def evaluate_answer(question, gold_answer, response, client, eval_model_id="gpt-4o-2024-08-06"):
    """Evaluates a text response against the gold answer."""
    eval_prompt = (f'''Evaluate the following response to the question, and determine if the answer is correct given the reference. Respond only with "yes" if the answer is equivalent, or "no" if it is not.
    Question: {question}
    Response: {response}
    Reference: {gold_answer}''')
    response = client.responses.create(
        model=eval_model_id,
        input=eval_prompt
    )
    eval_response = response.output_text
    if "yes" in eval_response.lower():
        result = True
    elif "no" in eval_response.lower():
        result = False
    return result

def check_text(question, true, pred):
  """LLM-as-a-judge eval."""
  assert len(question) == len(true) == len(pred)
  client = OpenAI(api_key=openai_api_key)
  res = []
  for i in tqdm(range(len(pred))):
    is_correct = 0
    try:
      text = pred[i].split("So the final answer is:")[1].strip()
      for ga in ast.literal_eval(true[i]):
        if evaluate_answer(question[i], ga, text, client):
          is_correct = 1
          break
      if not is_correct:
        for ga in ast.literal_eval(true[i]):
          if evaluate_answer(question[i], ga, pred[i], client):
            is_correct = 1
            break
    except:
      for ga in ast.literal_eval(true[i]):
        if evaluate_answer(question[i], ga, pred[i], client):
          is_correct = 0
          break
    res.append(is_correct)
  return res

def evaluate_all_question_types_and_print_report(data_paths, data_split,
                                                 output_folder=None):
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
          acc = check_text(ground_truths["question_commonsense"], 
                          ground_truths["answers_commonsense"], 
                          answers['generation'])
          accs["commonsense"] = acc
        if "math" in path:
          print("Evaluating math-only answers...")
          acc = accuracy(ground_truths["answer_math"],
                        answers["generation"])
          accs["math"] = acc
        elif "composition" in path:
          print("Evaluating compositional answers...")
          acc = accuracy(ground_truths["answer_composition"],
                        answers["generation"])
          accs["composition"] = acc

        answers["accuracy"] = acc
        answers.to_csv(data_paths[path])

    commonsense_acc = np.mean(accs['commonsense'])
    math_acc = np.mean(accs['math'])
    both_steps_acc = len(
       [x for x in zip(accs['commonsense'], accs['math']) if x==(1,1)]
      ) / len(accs['math'])  
    comp_acc = np.mean(accs['composition'])  
    comp_gap = comp_acc - both_steps_acc

    accuracy_report = f"""
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  PERFORMANCE REPORT FOR {data_split} DATA SPLIT
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Commonsense-only accuracy: { commonsense_acc }
  ---
  Math-only accuracy: { math_acc }
  ---
  Percentage of questions where both reasoning steps are solved correctly in isolation: {
    both_steps_acc
  }
  ---
  Compositional accuracy: { comp_acc }
  ---
  Compositionality gap: { comp_gap }
  """

    print(accuracy_report)

    if output_folder is not None:
       print("Saving evaluation report...")
       uid = uuid.uuid4()
       with open(os.path.join(output_folder,
                              f"report_{uid.hex}.txt"), "w") as text_file:
          text_file.write(accuracy_report)
       print("Done.")
        
if __name__ == "__main__":
    data_paths = {
        # paste the paths to the inference csv files here. 
        'commonsense_inference_data_path': "",
        'math_inference_data_path': "",
        'composition_inference_data_path': ""
    }
    evaluate_all_question_types_and_print_report(data_paths, *sys.argv[1:])