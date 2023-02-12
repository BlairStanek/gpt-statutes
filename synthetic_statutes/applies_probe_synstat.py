# Tests the ability of GPT3 to answer "does section ____ apply to ___"  over statutory text
# versus over non-statutory text with the same semantic meaning.

import generate_synstat
from generate_synstat import statute_part
import sys, argparse
sys.path.append('../')
import utils
import random
from datetime import datetime

start = datetime.now()
print("Start=", start)

random.seed(42)

parser = argparse.ArgumentParser(description='Generate synthetic statutes and questions to pass to GPT3')
parser.add_argument('--width', required=True, type=int,
                    help='Width of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--depth', required=True, type=int,
                    help='Depth of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--termtype', required=True, choices=["nonces", "ids"],
                    help='These are the basic types of prompting we handle')
parser.add_argument('--numruns', required=True, type=int,
                    help='These are the basic types of prompting we handle')
parser.add_argument('--fulloutput', action="store_true",
                    help='whether to output everything rather than a matrix, errors, and stats')
parser.add_argument('--statuteonly', action="store_true",
                    help='whether to do only statutes, leaving off the semantically identical sentences')
parser.add_argument('--noGPT', action="store_true",
                    help='for debugging; if passed, just generate statutes but do not actually call GPT')
parser.add_argument('--twoshot', action="store_true",
                    help='The default is 0-shot, but this generates one Yes and one No example (in random order)')
parser.add_argument('--skip_percent', type=float, default=0.0,
                    help='randomly skips this percent of all valid queries on a statute')


args = parser.parse_args()

# This function derives the ground truth against which we measure accuracy
# Returns True if it definitely applies.
# Returns False if it definitely does NOT apply
# Returns *None* if may or may not apply -- so shouldn't test
# To understand reasoning, consider statute "(i) foo means (I) any bar or (II) any boo"
def does_A_apply_to_anyB(A, B):
    # If Alice is a foo, then does (i) above apply to Alice?
    # It's Unclear, since Alice was a foo even without (i).  So return None.
    if A == B:
        return None

    # if A is a parent, grandparent, etc. of B,  it definitely applies
    # For example, if A is (i) and B is boo.
    x = B
    while not x is None:
        if x == A:
            return True
        x = x.parent
    # if sentence can be used to construct the part, then arguably applies
    y = A
    while not y is None:
        if y == B:
            # If Alice is a foo, then does (I) above apply to Alice?
            # Unclear, so we return None in that circumstance.
            return None
        y = y.parent
    return False

def select_furthest_items(L:list, a:statute_part, b=None) -> list:
    assert len(L) > 0
    max_dist = 0
    for x in L:
        dist = x.get_dist(a)
        if b is not None: # if b is provided, we want items that maximize the distance from b as well
            dist += x.get_dist(b)
        max_dist = max(max_dist, dist)
    rv = []
    for x in L:
        dist = x.get_dist(a)
        if b is not None: # if b is provided, we want items that maximize the distance from b as well
            dist += x.get_dist(b)
        if dist == max_dist:
            rv.append(x)
    return rv

def write_2shot_prompt(args, applies_target, Alice_type, all_parts) -> str:
    # Both Yes and No questions will be based on the same reference section
    # We randomly choose the reference section.
    example_level = args.depth - 1
    candidates = []
    for x in all_parts:
        if x.get_level() == example_level:
            if False == does_A_apply_to_anyB(applies_target, x) and \
                    False == does_A_apply_to_anyB(x, applies_target):
                candidates.append(x)

    # We ideally want an example section that doesn't apply to Alice
    ideal_candidates = [x for x in candidates if False == does_A_apply_to_anyB(x, Alice_type)]
    if len(ideal_candidates) > 0:
        candidates = ideal_candidates
    else:
        assert args.width == 2, "When else would this happen?"

    # We want the example to be as far as possible from both Alice's type and the target section
    candidates = select_furthest_items(candidates, Alice_type, applies_target)
    if args.fulloutput:
        print("For alice=", Alice_type.term, "and ", applies_target.stat_defined,
              "candidates:", [x.term + ":" + x.stat_defined for x in candidates])

    # choose the statute to be used in the few shots
    example_statute = random.choice(candidates)

    # choose the true target
    candidates_true = select_furthest_items(example_statute.get_all_descendants(), Alice_type)
    if args.fulloutput:
        print("true target candidates are:", [x.term + ":" + x.stat_used for x in candidates_true])
    true_item = random.choice(candidates_true)
    assert does_A_apply_to_anyB(example_statute, true_item) == True
    assert does_A_apply_to_anyB(applies_target, true_item) == False

    # choose the false target from valid ones, so that it is farthest from Alice and applies target
    candidates_false = []
    for x in all_parts:
        if False == does_A_apply_to_anyB(example_statute, x):
            candidates_false.append(x)
    assert len(candidates_false) > 0
    candidates_false = select_furthest_items(candidates_false, Alice_type, applies_target)
    ideal_candidates_false = [x for x in candidates_false if False == does_A_apply_to_anyB(applies_target, x)]
    if len(ideal_candidates_false) > 0:
        candidates_false = ideal_candidates_false
    print("false target candidates are:", [x.term + ":" + x.stat_used for x in candidates_false])
    false_item = random.choice(candidates_false)
    assert does_A_apply_to_anyB(example_statute, false_item) == False
    assert does_A_apply_to_anyB(applies_target, false_item) == False or (args.width == 2)

    # Generate the actual text.  With 50% probability, the false one comes first, second only second
    if random.random() < 0.5:
        rv = write_statute_facts_question("Charlie", false_item, example_statute)
        rv += write_explanation_false("Charlie", example_statute)
        rv += write_statute_facts_question("Bob", true_item, example_statute)
        rv += write_explanation_true("Bob", true_item, example_statute)
    else:
        rv  = write_statute_facts_question("Charlie", true_item, example_statute)
        rv += write_explanation_true("Charlie", true_item, example_statute)
        rv += write_statute_facts_question("Bob", false_item, example_statute)
        rv += write_explanation_false("Bob", example_statute)
    return rv

def write_out_sentence(statute):
    rv = " S" + statute.stat_defined[1:] + " says that " + statute.term.lower() + " means "
    for idx, c in enumerate(statute.children):
        rv += "any " + c.term.lower()
        if idx < len(statute.children) - 2:
            rv += ", "
        if idx == len(statute.children) - 2:
            rv += " or "
    rv += "."
    return rv

def write_explanation_true(person_name, person_type, example_statute) -> str:
    rv = write_out_sentence(example_statute)
    rv += " " + person_name + " is " + generate_synstat.get_article(person_type.term) + " " + \
            person_type.term.lower() + ", so " + example_statute.stat_defined + " does apply to him.\n\n"
    return rv

def write_explanation_false(person_name, example_statute) -> str:
    rv = write_out_sentence(example_statute)
    rv += " " + person_name + " is none of these, so "+ example_statute.stat_defined + \
          " does NOT apply to him.\n\n"
    return rv

def write_facts(person_name, person_type) -> str:
    return person_name + " is " + generate_synstat.get_article(person_type.term) + \
    " " + person_type.term.lower() + "."

def write_statute_facts_question(person_name, person_type, target_section) -> str:
    return write_facts(person_name, person_type) + " Does " + \
           target_section.stat_defined + " apply to " + person_name + "?"

if args.termtype == "nonces":
    raw_nonce_list =  generate_synstat.read_nonces()
else:
    raw_nonce_list =  generate_synstat.generate_systematic()

total_statute_results = {"True Positive": 0, "True Negative": 0,
                         "False Positive": 0, "False Negative": 0, "unclear":0}
total_prose_results = {"True Positive": 0, "True Negative": 0,
                       "False Positive": 0, "False Negative": 0, "unclear":0}
total_num = 0

for run_num in range(args.numruns):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("run_num=", run_num)

    random.seed(run_num) # re-seeding right before new shuffle fixes unexpected-reshuffling issues
    nonce_list = raw_nonce_list.copy()
    random.shuffle(nonce_list)

    abst = generate_synstat.generate_abstract(nonce_list, args.depth, args.width)

    statute = generate_synstat.abstract_to_statute(abst)
    print(statute)
    print("")

    prose, num_sentences = generate_synstat.abstract_to_prose(abst, "Sentence {:d}: ")

    if not args.statuteonly:
        print(prose)
        print("")

    all_parts = generate_synstat.extract_all_used_parts(abst)

    dict_sentence_defs = generate_synstat.get_dict_of_sentence_definitions(abst)

    statute_results = {"True Positive": 0, "True Negative": 0,
                       "False Positive":0, "False Negative": 0, "unclear":0}
    string_statute_errors = ""

    prose_results = {"True Positive": 0, "True Negative": 0,
                     "False Positive":0, "False Negative": 0, "unclear":0}
    string_prose_errors = ""

    num_this_run = 0

    for Alice_type in all_parts: # iterate over all the possible types for Alice
        if not args.fulloutput:
            print("{:<17s}".format(Alice_type.term), end="")

        # iterate over all the definitions, which are sentences in prose and subsections, etc. in statutory form
        # We always skip sentence 1, since it always applies, due to the definition.
        for sent_num in range(2, num_sentences):
            applies_target = dict_sentence_defs[sent_num]
            if not args.fulloutput:
                print(sent_num,end="")
            groundtruth = does_A_apply_to_anyB(applies_target, Alice_type)

            if groundtruth == None:
                if not args.fulloutput:
                    print("_        ", end="") # no GPT-3 call to make
            else:
                if random.random() < args.skip_percent/100.0:
                    print("SKIPPING ", end="")
                    continue

                if groundtruth:
                    if not args.fulloutput:
                        print("Y ", end="")
                else:
                    if not args.fulloutput:
                        print("n ", end="")

                examples = "" # If doing 2-shot we have work to do to create the examples
                if args.twoshot:
                    examples = write_2shot_prompt(args, applies_target, Alice_type, all_parts)

                # Build the question to pass into GPT-3
                statute_question = examples + \
                                   write_statute_facts_question("Alice", Alice_type, applies_target)
                if not args.twoshot: # We don't add this if we already have examples
                    statute_question += " Let's think step by step."
                prose_question = write_facts("Alice", Alice_type) + \
                                 " Does sentence " + str(sent_num) + " apply to Alice?"
                prose_question += " Let's think step by step."
                statute_prompt = statute + "\n" + statute_question
                prose_prompt = prose + "\n" + prose_question

                SECOND_PROMPT = "\nTherefore, the answer (Yes or No) is"  # cf. Kojima et al. 2022 appendix A.5

                if args.fulloutput:
                    print("--------")
                    print(statute_question)
                    print("--------")

                # Make the GPT-3 calls for the statutory reasoning version
                if not args.noGPT:
                    utils.add_comment("Synthetic applies probe in " + __file__ + " Alice_type=" + Alice_type.term + " sent_num=" + str(sent_num))
                    statute_response = utils.call_gpt3_withlogging(statute_prompt, "text-davinci-003", max_tokens=2000)
                    utils.add_comment("Synthetic applies probe in " + __file__ + " SECOND PROMPT")
                    second_statute_prompt = statute_prompt + statute_response + SECOND_PROMPT
                    second_statute_response = utils.call_gpt3_withlogging(second_statute_prompt, "text-davinci-003", max_tokens=2000)
                else:
                    statute_response = second_statute_response = ["No.","No","Yes.", "maybe?"][num_this_run % 4]

                statute_result = "unclear"
                if utils.is_yes(second_statute_response):
                    if groundtruth:
                        statute_result = "True Positive"
                        if not args.fulloutput:
                            print("sY ", end="")
                    else:
                        statute_result = "False Positive"
                        if not args.fulloutput:
                            print("sY*", end="")
                elif utils.is_no(second_statute_response):
                    if groundtruth:
                        statute_result = "False Negative"
                        if not args.fulloutput:
                            print("sn*", end="")
                    else:
                        statute_result = "True Negative"
                        if not args.fulloutput:
                            print("sn ", end="")
                else:
                    if not args.fulloutput:
                        print("s? ", end="")
                statute_results[statute_result] += 1

                if not statute_result.startswith("True"):
                    string_statute_errors += statute_result + "\n" + \
                                             statute_question + "\n[prompt above/first response below]\n" + \
                                             statute_response + "\n" + SECOND_PROMPT + "\n " + \
                                             second_statute_response + \
                                             "\n****************************\n"

                if args.fulloutput and not args.noGPT:
                    print(statute_response)
                    print("-----")
                    print(second_statute_response)
                    print("-----")
                    print("Correct is", groundtruth)
                    print(statute_result)
                    print("-----")

                if not args.statuteonly:
                    if args.fulloutput:
                        print(prose_question)
                        print("-----")

                    if not args.noGPT:
                        # Make the GPT-3 calls for the PROSE reasoning version
                        utils.add_comment("Synthetic applies probe PROSE VERSION in " + __file__ + " Alice_type=" + Alice_type.term + " sent_num=" + str(sent_num))
                        prose_response = utils.call_gpt3_withlogging(prose_prompt, "text-davinci-003", max_tokens=2000)
                        utils.add_comment("Synthetic applies probe PROSE VERSION in " + __file__ + " SECOND PROMPT")
                        second_prose_prompt = prose_prompt + prose_response + SECOND_PROMPT
                        second_prose_response = utils.call_gpt3_withlogging(second_prose_prompt, "text-davinci-003", max_tokens=2000)
                    else:
                        prose_response = second_prose_response = statute_response

                    # now check PROSE response
                    prose_result = "unclear"
                    if utils.is_yes(second_prose_response):
                        if groundtruth:
                            prose_result = "True Positive"
                            if not args.fulloutput:
                                print("pY ", end="")
                        else:
                            prose_result = "False Positive"
                            if not args.fulloutput:
                                print("pY*", end="")
                    elif utils.is_no(second_prose_response):
                        if groundtruth:
                            prose_result = "False Negative"
                            if not args.fulloutput:
                                print("pn*", end="")
                        else:
                            prose_result = "True Negative"
                            if not args.fulloutput:
                                print("pn ", end="")
                    else:
                        if not args.fulloutput:
                            print("p? ", end="")
                    prose_results[prose_result] += 1

                    if not prose_result.startswith("True"):
                        string_prose_errors += prose_question + "\n[prompt above/first response below]\n" + \
                                                 prose_response + "\n" + SECOND_PROMPT + "\n " + \
                                                 second_prose_response + \
                                                 "\n****************************\n"

                    if args.fulloutput:
                        print(prose_response)
                        print("-----")
                        print(second_prose_response)
                        print("-----")
                        print("Correct is", groundtruth)
                        print(prose_result)
                        print("-----")
                    else:
                        print(" ", end="", flush=True)

                num_this_run += 1
        print("")

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print("This run STATUTE ERRORS:+++++++++++++++++++++++++++++++++\n", string_statute_errors)
    if not args.statuteonly:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print("This run PROSE ERRORS:\n", string_prose_errors)

    print("num_this_run=", num_this_run)
    print("This run statute_results:" , statute_results)
    if not args.statuteonly:
        print("This run prose_results: ", prose_results)

    total_num += num_this_run
    check_sum = 0
    for t in total_statute_results.keys():
        total_statute_results[t] += statute_results[t]
        total_prose_results[t] += prose_results[t]
        check_sum += total_statute_results[t]
    assert check_sum == total_num

    print("total_num=", total_num)
    print("so-far total_statute_results=", total_statute_results)
    statute_correct = (total_statute_results['True Positive'] + total_statute_results['True Negative'])
    print("so-far statute accuracy: {:.2f}".format(statute_correct/float(total_num)),
          "(" + str(statute_correct) + "/" + str(total_num) + ")")
    if not args.statuteonly:
        print("so-far total_prose_results=", total_prose_results)
        prose_correct = (total_prose_results['True Positive'] + total_prose_results['True Negative'])
        print("so-far prose accuracy: {:.2f}".format(prose_correct / float(total_num)),
              "(" + str(prose_correct) + "/" + str(total_num) + ")")


suggested_filename = args.termtype+"_w"+ str(args.width)+ \
                     "_d"+str(args.depth)+"_"+ \
                     str(args.numruns)+"runs"
if args.statuteonly and not args.twoshot:
    suggested_filename += "_statuteonly"
elif args.twoshot:
    suggested_filename += "_2shot"

suggested_filename += ".txt"

end = datetime.now()
print("End=", end)
print("Time taken=", end-start)


print("Suggested filename:", suggested_filename)
print('\a\a\a\a\a\a\a\a\a\a\a\a\a\a\a\a') # play sounds