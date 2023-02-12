import json, sys, argparse
sys.path.append('../')
import utils
from utils import is_entail, is_contra, is_entail_or_contra, reformat_case, print_confusion_matrix

DEBUG_ONCE = False

parser = argparse.ArgumentParser(description='Call GPT3 with 4-shot dynamic prompts for SARA')
parser.add_argument('--letsthink', required=True, choices=["Yes", "no"],
                    help='Whether to add "Lets think step by step." per Kojima et al 2022')
parser.add_argument('--withstatute', required=True, choices=["Yes", "no"],
                    help='Whether to include the relevant statute at the top of the prompt')
parser.add_argument('--ptype', required=True, choices=["0shot", "4shot", "chainofthought"],
                    help='These are the basic types of prompting we handle')

args = parser.parse_args()

print("RUNNING WITH: args.letsthink=", args.letsthink,
      "args.withstatute=", args.withstatute,
      "args.ptype=", args.ptype)

max_tokens = 1200  # works safely for most of our calls to GPT3

CoT_text = None # if we are doing chain of thought reasoning, we should read in the hand-crafted chains now
if args.ptype == "chainofthought":
    if args.withstatute == "Yes":
        with open("sara-chain-of-thought-prompt.txt", "r") as fCOT:
            CoT_text = fCOT.read()
        max_tokens = 336 # most that can be accomodated with this prompt as is
    else:
        with open("sara-chain-of-thought-prompt-NOSTATUTES.txt", "r") as fCOT:
            CoT_text = fCOT.read()

num_run = 0

# used to get the confusion matrix for dollar-figure-based entailment problems
num_dollar_entail_cor = 0
num_dollar_contra_cor = 0
num_dollar_entail_answercontra = 0
num_dollar_contra_answerentail = 0

# used to get the confusion matrix for NON-dollar-figure-based entailment problems
num_nodollar_entail_cor = 0
num_nodollar_contra_cor = 0
num_nodollar_entail_answercontra = 0
num_nodollar_contra_answerentail = 0

skipped_due_to_no_answer = 0
had_to_force_answer = 0

utils.add_comment("START " + __file__)

START_PROMPT = "We are going to be doing Entailment/Contradiction reasoning applying the statute below:\n\n"

json_records = json.load(open('statutory-reasoning-gpt-prompts.json', 'r'))
for json_item in json_records:
    if is_entail_or_contra(json_item["answer"]): # only handling non-number issues right now
        has_dollar = ("$" in json_item['test case']) # separates out the numerical and non-numerical ones

        if has_dollar:
            continue # DEBUG FOR NOW

        prompt = ""
        if args.ptype == "chainofthought":
            prompt += CoT_text
        else:
            if args.withstatute == "Yes":
                prompt += START_PROMPT
                prompt += json_item['statute'].replace("\n\n", "\n")  # removes double newlines
                prompt = prompt.strip() + "\n\n"

            if args.ptype == "4shot":
                prompt += reformat_case(json_item['case1'],     "Premise: ", "Hypothesis: ", "Answer: ",
                                        add_cite_before_section=(args.withstatute == "no")) + "\n\n"
                prompt += reformat_case(json_item['case2'],     "Premise: ", "Hypothesis: ", "Answer: ",
                                        add_cite_before_section=(args.withstatute == "no")) + "\n\n"
                prompt += reformat_case(json_item['case3'],     "Premise: ", "Hypothesis: ", "Answer: ",
                                        add_cite_before_section=(args.withstatute == "no")) + "\n\n"
                prompt += reformat_case(json_item['case4'],     "Premise: ", "Hypothesis: ", "Answer: ",
                                        add_cite_before_section=(args.withstatute == "no")) + "\n"
        prompt = prompt.strip() + "\n\n"

        # Now the one we want answered:
        prompt += reformat_case(json_item['test case'], "Premise: ", "Hypothesis: ", "Answer: ", True,
                                add_cite_before_section=(args.withstatute == "no"))

        if args.letsthink == "Yes":
            prompt += "Let's think step by step." # following Kojima et al. 2022

        prompt = prompt.strip() # GPT-3 apparently does not like whitespace at the start or end of the prompt

        # DEBUG below; used to probe a particular case
        # if "Bob is Alice's son since April 15th, 2014" not in prompt or \
        #     "relationship to Bob under section 152(d)(2)(C)" not in prompt:
        #     continue

        utils.add_comment("Doing case id=" + json_item["case id"])
        raw_response = utils.call_gpt3_withlogging(prompt, "text-davinci-003", max_tokens=max_tokens)
        utils.add_comment("NOTE that correct response is " + json_item["answer"])

        second_response = None
        if is_entail_or_contra(raw_response.split()[-1]): # Trying last one, at end of chain of reasoning
            response = raw_response.split()[-1]
        else:
            print("Doing second prompt to clarify", json_item["case id"])
            FORCING_PROMPT = "Therefore, the answer (Entailment or Contradiction) is" # see Kojima et al 2022 A.5
            second_prompt = prompt + raw_response + FORCING_PROMPT
            second_response = utils.call_gpt3_withlogging(second_prompt,
                                                          "text-davinci-003",
                                                          max_tokens=(max_tokens-len(raw_response.split())))
            if not is_entail_or_contra(second_response.split()[0].strip(",")):
                print("GOT UNCLEAR ANSWER with ", json_item["case id"], ":", second_response)
                skipped_due_to_no_answer += 1
                continue # skip
            had_to_force_answer += 1
            response = second_response.split()[0].strip(",")

        if DEBUG_ONCE:
            print("----------")
            print(prompt)
            print("----------")
            print(raw_response)
            print("----------")
            print(response)
            print("----------")
            print("CORRECT IS:", json_item["answer"])
            exit(1)

        num_run += 1
        print("{:15s}".format(json_item["case id"]), "GPT3 Response:", response, "Groundtruth:", json_item["answer"])
        correct = False
        assert is_entail_or_contra(response)
        if has_dollar:
            if is_entail(json_item["answer"]) and is_entail(response):
                num_dollar_entail_cor += 1
                correct = True
            elif is_entail(json_item["answer"]) and is_contra(response):
                num_dollar_entail_answercontra += 1
            elif is_contra(json_item["answer"]) and is_entail(response):
                num_dollar_contra_answerentail += 1
            else:
                num_dollar_contra_cor += 1
                correct = True
                assert is_contra(json_item["answer"]) and is_contra(response)
        else:
            if is_entail(json_item["answer"]) and is_entail(response):
                num_nodollar_entail_cor += 1
                correct = True
            elif is_entail(json_item["answer"]) and is_contra(response):
                num_nodollar_entail_answercontra += 1
            elif is_contra(json_item["answer"]) and is_entail(response):
                num_nodollar_contra_answerentail += 1
            else:
                num_nodollar_contra_cor += 1
                correct = True
                assert is_contra(json_item["answer"]) and is_contra(response)

        if not correct:
            print("Case passed to Prompt:-------")
            print(prompt[len(CoT_text.strip()):])
            print("Response:-------")
            print(raw_response)
            if second_response is not None:
                print("Second Response:-------")
                print(second_response)
            print("--------------")

        assert num_run == num_dollar_entail_cor + num_dollar_contra_cor +        \
                        num_dollar_entail_answercontra + num_dollar_contra_answerentail + \
                        num_nodollar_entail_cor + num_nodollar_contra_cor + \
                        num_nodollar_entail_answercontra + num_nodollar_contra_answerentail
print("----------------------------------")
print("DOLLAR:")
print_confusion_matrix(num_dollar_entail_cor, num_dollar_contra_cor,
                       num_dollar_entail_answercontra, num_dollar_contra_answerentail)
print("----------------------------------")
print("NO DOLLAR:")
print_confusion_matrix(num_nodollar_entail_cor, num_nodollar_contra_cor,
                       num_nodollar_entail_answercontra, num_nodollar_contra_answerentail)
print("----------------------------------")
print("SUM:")
print_confusion_matrix(num_dollar_entail_cor + num_nodollar_entail_cor,
                       num_dollar_contra_cor + num_nodollar_contra_cor,
                       num_dollar_entail_answercontra + num_nodollar_entail_answercontra,
                       num_dollar_contra_answerentail + num_nodollar_contra_answerentail)
print("----------------------------------")
print("skipped_due_to_no_answer=", skipped_due_to_no_answer)
print("had_to_force_answer=", had_to_force_answer)
print("----------------------------------")
print("RUNNING WITH: args.letsthink=", args.letsthink,
      "args.withstatute=", args.withstatute,
      "args.ptype=", args.ptype)
print("Suggested filename for the above output: ", args.ptype +
      "_letsthink_" + args.letsthink +
      "_withstatute_" + args.withstatute + ".txt")
