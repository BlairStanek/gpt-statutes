# This tests on the simple question of what term is defined at a particular cite


import generate_synstat
from generate_synstat import NOT_PARALLEL, NOT_FOUND
import sys, argparse, re
sys.path.append('../')
import utils
import random
from datetime import datetime
from collections import Counter

parser = argparse.ArgumentParser(description='Generate synthetic statutes and questions to pass to GPT')
parser.add_argument('--width', required=True, type=int,
                    help='Width of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--depth', required=True, type=int,
                    help='Depth of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--numruns', default=1, type=int,
                    help='These are the basic types of prompting we handle')
parser.add_argument('--skip_first', type=int, default=0,
                    help='skips this number of queries before actually making calls; used to extend ')
DEFAULT_MODEL = "text-davinci-003"
parser.add_argument('--model', default=DEFAULT_MODEL,
                    help='which openai model to use')
args = parser.parse_args()
print("args=", args)
print(datetime.now())
statute_random = random.Random(42)  # used for shuffling to get the statute

histogram_errors = Counter()
histogram_absolute_loc = Counter()

raw_nonce_list = generate_synstat.read_nonces()

count_calls = 0
count_wrong = 0

for idx_run in range(args.numruns):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ idx_run=", idx_run)
    statute_random.seed(idx_run) # re-seed in a reproducible way
    nonce_list = raw_nonce_list.copy()
    statute_random.shuffle(nonce_list)
    LEVEL_FOR_TERMS = 1 # we need an extra level of statute that is not on separate lines, but folded in
    abst = generate_synstat.generate_abstract(nonce_list, args.depth + LEVEL_FOR_TERMS, args.width)
    curr_statute = generate_synstat.abstract_to_statute(abst, keep_compact=True, collapse_leaves=True)
    print(curr_statute)

    all_parts = generate_synstat.extract_all_used_parts(abst)
    parents_only = [p for p in all_parts if p.has_children()]

    for idx_parent, parent in enumerate(parents_only):
        count_calls += 1

        if args.skip_first > 0 and count_calls <= args.skip_first:
            continue # this implements the skip_first, which is used to extend a set of experiments

        question = "What is the term defined at " + parent.stat_defined + "?"
        query = curr_statute + "\n\n" + question

        messages = [{"role": "user", "content": query}]

        print("PASSED:  ", question)
        statute_response = utils.call_gpt_withlogging(messages, args.model, max_tokens=1000)
        print("RESPONSE:", statute_response.strip())

        lowest_idx = len(statute_response) + 100
        answer_part = None
        for part in all_parts:
            idx = statute_response.lower().find(part.term.lower())
            if idx > 0 and idx < lowest_idx:
                lowest_idx = idx
                answer_part = part

        if answer_part != parent:
            count_wrong += 1
            error_analysis = None
            CHILD_OF_CORRECT = "child of correct"
            CHILD_OF_DIFFERENT = "child of different"
            if answer_part is None:
                error_analysis = [NOT_FOUND]
            elif answer_part in parent.children:
                error_analysis = [CHILD_OF_CORRECT]
            elif not answer_part.has_children():
                error_analysis = [CHILD_OF_DIFFERENT]
            else:
                error_analysis = generate_synstat.analyze_error(parent.stat_defined, answer_part.stat_defined)

            answer_defined = "SOMETHING NOT IN STATUTE"
            if not answer_part is None:
                answer_defined = answer_part.stat_defined
            print("*** WRONG!", "correct=", parent.term, "at", parent.stat_defined, " but got", answer_defined)

            if error_analysis == NOT_PARALLEL:
                error_text = error_analysis
            else:
                error_text = ",".join(sorted(error_analysis))
            histogram_errors.update([error_text])
            print("error_text=", error_text)
            histogram_absolute_loc.update([idx_parent]) # store absolute location

    print("histogram_absolute_loc_list: -------------------")
    histogram_absolute_loc_list = list(histogram_absolute_loc.items())
    histogram_absolute_loc_list.sort(key=lambda x: x[0])
    for x in histogram_absolute_loc_list:
        print(x[0], " ", x[1])
    print("NOTE: len(parents_only)=", len(parents_only))

    print("histogram_errors: -------------------")
    histogram_errors_list = list(histogram_errors.items())
    histogram_errors_list.sort(key=lambda x: x[1], reverse=True)  # sort by COUNT, not errors
    for x in histogram_errors_list:
        print("{:5d}".format(x[1]), " ", x[0])

    print("Wrong =", count_wrong, "of", count_calls)

print(datetime.now())
print("Suggested filename: definedat_d" + str(args.depth) + "w" + str(args.width) +
      "n" + str(args.numruns) + "_" + args.model + ".txt")
