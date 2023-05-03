# Basic experimentation to see what GPT3 knows of SARA.
import json, random, sys
sys.path.append('../')
import utils

json_records = json.load(open('statutory-reasoning-gpt-prompts.json', 'r'))
all_cases_text = []
for test_case in json_records:
    for id in ["test case", "case1", "case2", "case3", "case4"]:
        if test_case[id] not in all_cases_text:
            all_cases_text.append(test_case[id])

random.shuffle(all_cases_text)

for i in range(20):
    prompt = all_cases_text[i] + "\nWhere is the text above from?"
    print(prompt)
    # response = utils.call_gpt3_withlogging(prompt,
    #                                        "text-davinci-003",
    #                                        max_tokens=1000)
    # print(response)
    print("--------")

