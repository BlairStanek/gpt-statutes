# This complements call_gpt_with_sara.py, which handles the Entailment/Contradiction
# SARA cases.  This file calls GPT-* with the SARA tax cases where the answer is a
# dollar figure.

import json, sys, argparse
import re

sys.path.append('../')
import utils

MODEL = "gpt-4-0314" # maximizes reproducability by using frozen version

# This is the system prompt, used by OpenAI at minute 19:30 of the GPT-4
# launch livestream, available at https://www.youtube.com/watch?v=outcGtbnMuQ
SYSTEM_TEXT = "You are TaxGPT, a large language model trained by OpenAI.\n\n" + \
            "Carefully read & apply the tax code, being certain to spell out your " + \
            "calculations & reasoning so anyone can verify them. Spell out everything " + \
            "in painstaking detail & don't skip any steps!"

def dollar_string_to_float(txt) -> float:
    txt = txt.strip()
    txt = txt.lstrip("$")
    txt = txt.rstrip(".")
    txt = txt.replace(",", "")
    txt = txt.strip()
    return float(txt)

# Gather the set of just the numerical cases (i.e., where answers are dollar figures)
dollar_cases = [] # stored as a list of tuples of 2-tuples of (question, answer)
json_records = json.load(open('statutory-reasoning-gpt-prompts.json', 'r'))
for json_item in json_records:
    if "$" in json_item["answer"]:
        dollar_cases.append((json_item["test case"], json_item["answer"]))
    for training_case in ["case1", "case2", "case3", "case4"]:
        if json_item[training_case][-1].isnumeric():
            idx_dollar = json_item[training_case].rfind("$")
            assert idx_dollar > (len(json_item[training_case]) - 10)
            dollar_cases.append((json_item[training_case][:idx_dollar].rstrip(),
                                 json_item[training_case][idx_dollar:].strip()))

print("len(dollar_cases)=",len(dollar_cases))
# for rec in dollar_cases:
#     print(rec[0],"\n\t", rec[1])

# Load SARA statutes to put at start of prompt
with open('all_sara_statutes.txt', 'r') as f:
    all_sara_statutes = f.read()

# The results will be stored as 2-tuples of (float groundtruth, float predicted by GPT)
groundtruth_vs_predicted = []

for case in dollar_cases:
    prompt = all_sara_statutes + "\n\n" + case[0]

    print("RUNNING:", case[0])
    print("Groundtruth:", case[1])
    groundtruth = dollar_string_to_float(case[1])

    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": prompt}
    ]

    response = utils.call_gpt_raw(messages, MODEL)
    print("Response 1:", response)
    utils.add_comment("Correct answer=" + case[1])

    messages2 = messages.copy()
    messages2.append({"role": "assistant", "content": response})
    messages2.append({"role": "user", "content": "Therefore, the answer (dollar figure) is:"})
    response2 = utils.call_gpt_raw(messages2, MODEL, max_tokens=300) # may run out of space

    response2_dollar_figure = re.search("\$(\d|,)*\d(\.\d\d)?\.?\s*$",  response2)

    if response2_dollar_figure is not None:
        dollar_amount = dollar_string_to_float(response2_dollar_figure[0])
    else:
        print("Got no good dollar figure:", response2)
        continue

    print("RESULT gt {:11.2f} pred {:11.2f}".format(groundtruth, dollar_amount))
    groundtruth_vs_predicted.append((groundtruth, dollar_amount))

print("len(groundtruth_vs_predicted)=", len(groundtruth_vs_predicted))

