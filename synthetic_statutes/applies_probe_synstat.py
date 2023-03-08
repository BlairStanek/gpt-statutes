# Tests the ability of GPT3 to answer "does section ____ apply to ___"  over statutory text.
# Has several variations, including comparing to non-statutory versions (i.e. numbered sentences),
# N-shot, etc.

import generate_synstat
from generate_synstat import statute_part
import sys, argparse
sys.path.append('../')
import utils
import random
from datetime import datetime

start = datetime.now()
print("Start=", start)

# To prevent weird reproducability issues, we need separate random number generators for different purposes
# when one option may be on while the other may be turned off
statute_random = random.Random(42) # used for shuffling to get the statute
Nshot_random = random.Random(42) # used for generating the prompts (including N-shot)
skip_random = random.Random(42) # when running a percentage of a run, this is used

parser = argparse.ArgumentParser(description='Generate synthetic statutes and questions to pass to GPT3')
parser.add_argument('--width', required=True, type=int,
                    help='Width of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--depth', required=True, type=int,
                    help='Depth of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--termtype', required=True, choices=["nonces", "ids"],
                    help='These are the basic types of prompting we handle')
parser.add_argument('--numruns', required=True, type=int,
                    help='These are the basic types of prompting we handle')
parser.add_argument('--do_sentences', action="store_true",
                    help='do the semantically identical sentences in addition to the statutes')
parser.add_argument('--noGPT', action="store_true",
                    help='for debugging; if passed, just generate statutes but do not actually call GPT')
parser.add_argument('--Nshot', required=False, type=int, default=0,
                    help='Allows for N shot with N examples (must be EVEN, so positive and negative balanced)')
parser.add_argument('--Nshot_type', choices=["1statute", "many_statute", "many_statute_same_pos"],
                    help='whether to do N-shot with N questions or N statutes (including same position)')
parser.add_argument('--skip_percent', type=float, default=0.0,
                    help='randomly skips this percent of all valid queries on a statute')
parser.add_argument('--max_num', type=int, default=0,
                    help='stop after this number of queries; often combined with a skip_percent>0')
parser.add_argument('--subdivs', required=True, choices=["leavesonly", "noleaves", "both"],
                    help='which type of subdivisions to consider asking about')
parser.add_argument('--model', default="text-davinci-003",
                    help='which openai model to use')


args = parser.parse_args()

if args.Nshot % 2 == 1:
    print("Nshot was set to", args.Nshot, "but it must be even")
    exit(0)

if args.Nshot > 0 and args.Nshot_type is None:
    print("If you select Nshot you need to select one of the Nshot_type")
    exit(0)

if args.do_sentences and args.subdivs != "noleaves":
    print("Logically cannot do_sentences for leaves, so subdivs must be noleaves")
    exit(0)

if args.do_sentences and args.Nshot > 0:
    print("Doing sentences with N-shot for N>0 not yet implemented")
    exit(0)

# This function derives the ground truth against which we measure accuracy.  It's important.
# Returns True if it definitely applies.
# Returns False if it definitely does NOT apply
# Returns *None* if may or may not apply -- so shouldn't test
# To understand reasoning, consider statute "(i) foo means (I) any bar or (II) any boo"
def does_A_apply_to_anyB(A, B):
    if A == B:
        if not A.has_children():
            # if Alice is a boo, then (i)(II) definitely applies to Alice
            return True
        else:
            # If Alice is a foo, then does (i) above apply to Alice?
            # It's Unclear, since Alice was a foo even without (i).  So return None since ambiguous.
            return None

    # if A is a parent, grandparent, etc. of B,  it definitely applies
    # For example, if A is (i) and B is boo.
    x = B
    while not x is None:
        if x == A:
            assert A != B or not A.has_children()
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

# We need to use random names with equal change of each gender to avoid bias.
# The below were drawn from the top 15 girl names for 2018 and top 15 boy names for 2020 births,
# from data at https://www.ssa.gov/oact/babynames/.

NAMES = [   "Liam", "Olivia",
            "Noah", "Emma",
            "Oliver", "Ava",
            "Elijah", "Charlotte",
            "William", "Sophia",
            "James", "Amelia",
            "Benjamin", "Isabella",
            "Lucas", "Mia",
            "Henry", "Evelyn",
            "Alexander", "Harper",
            "Mason", "Camila",
            "Michael", "Abigail",
            "Ethan", "Gianna",
            "Daniel", "Luna",
            "Jacob", "Ella"]


def write_multistatute_Nshot_prompt(args, applies_target, Nshot_statutes, names_set) -> str:
    rv = ""
    used_sec_nums = {1001}

    for abst in Nshot_statutes:
        # randomly generate a statute
        sec_num = None
        while sec_num is None or sec_num in used_sec_nums:
            sec_num = Nshot_random.randint(1010, 9999)
        used_sec_nums.add(sec_num)

        # append the statute to the prompt
        rv += generate_synstat.abstract_to_statute(abst, sec_num=sec_num) + "\n"

        all_parts = generate_synstat.extract_all_used_parts(abst)

        name = names_set.pop() # the names set was already randomly shuffled

        # both questions will have the same applies-to-target.
        if args.Nshot_type == "many_statute_same_pos":
            assert False, "not yet implemented"
        else:
            # choose an applies-to target of a type that might have been chosen
            if args.subdivs == "leavesonly":
                possible_target_subsection = [p for p in all_parts if not p.has_children()]
            elif args.subdivs == "noleaves":  # we are doing it by numbered sentence, so just store the sentence nums
                possible_target_subsection = [p for p in all_parts if p.has_children() and not p.parent is None]
            elif args.subdivs == "both":
                possible_target_subsection = [p for p in all_parts if not p.parent is None]
            else:
                assert False, "not implemented"
            target_subsection = Nshot_random.choice(possible_target_subsection)

            # Generate a true example with 50% chance
            if Nshot_random.random() < 0.5:
                true_type = \
                    Nshot_random.choice([p for p in all_parts if True == does_A_apply_to_anyB(target_subsection, p)])
                rv += write_statute_facts_question(name, true_type, target_subsection)
                rv += write_explanation_true(name, true_type, target_subsection)
            else:
                false_type = \
                    Nshot_random.choice([p for p in all_parts if False == does_A_apply_to_anyB(target_subsection, p)])
                rv += write_statute_facts_question(name, false_type, target_subsection)
                rv += write_explanation_false(name, target_subsection)
    return rv

def write_1statute_Nshot_prompt(args, applies_target, person_type, all_parts) -> str:
    assert args.Nshot <= len(NAMES)
    assert (args.Nshot % 2) == 0
    already_used_subsections = set()
    rv = ""

    for i in range(args.Nshot, 0, -2):
        name1 = NAMES[i-1]
        name2 = NAMES[i-2]

        # Both Yes and No questions will be based on the same reference section
        # We randomly choose the reference section.
        example_level = args.depth - 1
        candidate_subsections = []
        for x in all_parts:
            if x.get_level() == example_level:
                if False == does_A_apply_to_anyB(applies_target, x) and \
                        False == does_A_apply_to_anyB(x, applies_target) and \
                        not x in already_used_subsections:
                    candidate_subsections.append(x)
        assert len(candidate_subsections) > 0, "Too few subsections for N shot with N=" + str(args.Nshot)

        # We ideally want an example section that doesn't apply to the person type
        ideal_candidate_subsections = [x for x in candidate_subsections if False == does_A_apply_to_anyB(x, person_type)]
        if len(ideal_candidate_subsections) > 0:
            candidate_subsections = ideal_candidate_subsections

        # We want the example to be as far as possible from both the person type and the target section
        candidate_subsections = select_furthest_items(candidate_subsections, person_type, applies_target)
        print("For person_type=", person_type.term, "and ", applies_target.stat_defined,
              "candidates:", [x.term + ":" + x.stat_defined for x in candidate_subsections])

        # choose the statute to be used in the few shots
        target_subsection = Nshot_random.choice(candidate_subsections)
        already_used_subsections.add(target_subsection)
        print("  chosen=", target_subsection.term + ":" + target_subsection.stat_defined)

        # choose the true target
        candidates_true = select_furthest_items(target_subsection.get_all_descendants(), person_type)
        print("true target candidates are:", [x.term + ":" + x.stat_used for x in candidates_true])
        true_item = Nshot_random.choice(candidates_true)
        print("  true_item chosen=", true_item.term + ":" + true_item.stat_used)
        assert does_A_apply_to_anyB(target_subsection, true_item) == True
        assert does_A_apply_to_anyB(applies_target, true_item) == False
        assert applies_target != true_item

        # choose the false target from valid ones, so that it is farthest from the person type and applies target
        candidates_false = []
        for x in all_parts:
            if False == does_A_apply_to_anyB(target_subsection, x):
                candidates_false.append(x)
        assert len(candidates_false) > 0
        candidates_false = select_furthest_items(candidates_false, person_type, applies_target)
        ideal_candidate_subsections_false = [x for x in candidates_false if False == does_A_apply_to_anyB(applies_target, x)]
        if len(ideal_candidate_subsections_false) > 0:
            candidates_false = ideal_candidate_subsections_false
        print("false target candidates are:", [x.term + ":" + x.stat_used for x in candidates_false])
        false_item = Nshot_random.choice(candidates_false)
        print("  false_item=", false_item.term + ":" + false_item.stat_used)
        assert does_A_apply_to_anyB(target_subsection, false_item) == False
        assert does_A_apply_to_anyB(applies_target, false_item) == False or (args.width == 2)
        assert applies_target != false_item

        # Generate the actual text.  With 50% probability, the false one comes first, second only second
        if Nshot_random.random() < 0.5:
            rv += write_statute_facts_question(name1, false_item, target_subsection)
            rv += write_explanation_false(name1, target_subsection)
            rv += write_statute_facts_question(name2, true_item, target_subsection)
            rv += write_explanation_true(name2, true_item, target_subsection)
        else:
            rv += write_statute_facts_question(name1, true_item, target_subsection)
            rv += write_explanation_true(name1, true_item, target_subsection)
            rv += write_statute_facts_question(name2, false_item, target_subsection)
            rv += write_explanation_false(name2, target_subsection)

    return rv

def write_out_sentence(statute):
    if statute.has_children():
        rv = " S" + statute.stat_defined[1:] + " says that " + statute.term.lower() + " means "
        for idx, c in enumerate(statute.children):
            rv += "any " + c.term.lower()
            if idx < len(statute.children) - 2:
                rv += ", "
            if idx == len(statute.children) - 2:
                rv += " or "
        rv += "."
    else:
        rv = " S" + statute.stat_used[1:] + " applies to any " + statute.term.lower() + "."
    return rv

def write_explanation_true(person_name, person_type, target_subsec) -> str:
    assert does_A_apply_to_anyB(target_subsec, person_type)

    # 1) leaf equality
    if person_type == target_subsec:
        assert not target_subsec.has_children(), "expected this only for a leaf node"
        rv = write_out_sentence(target_subsec)
        rv += " " + person_name + " is " + generate_synstat.get_article(person_type.term) + " " + \
              person_type.term.lower() + ", so " + target_subsec.stat_used + \
              " does apply to " + person_name + ".\n\n"
        return rv
    else:
        # might have to run up the chain if application isn't within the single sentence
        while person_type.parent != target_subsec:
            rv = write_out_sentence(person_type.parent)
            rv += " " + person_name + " is " + generate_synstat.get_article(person_type.term) + " " + \
                person_type.term.lower() + ", so " + person_name + " is " + \
                  generate_synstat.get_article(person_type.parent.term) + " " + \
                  person_type.parent.term
            person_type = person_type.parent # working up the chain

        rv = write_out_sentence(target_subsec)
        rv += " " + person_name + " is " + generate_synstat.get_article(person_type.term) + " " + \
                person_type.term.lower() + ", so " + target_subsec.stat_defined + \
                " does apply to " + person_name + ".\n\n"
        return rv

def write_explanation_false(person_name, example_statute) -> str:
    rv = write_out_sentence(example_statute)
    if example_statute.has_children():
        rv += " " + person_name + " is none of these, so " + example_statute.stat_defined
    else:
        rv += " " + person_name + " is not " + \
             generate_synstat.get_article(example_statute.term) + " " + \
             example_statute.term.lower() + ", so "+ example_statute.stat_used
    rv += " does NOT apply to " + person_name + ".\n\n"
    return rv

def write_facts(person_name, person_type) -> str:
    return person_name + " is " + generate_synstat.get_article(person_type.term) + \
    " " + person_type.term.lower() + "."

def write_statute_facts_question(person_name, person_type, target_section) -> str:
    if not target_section.stat_defined is None:
        return write_facts(person_name, person_type) + " Does " + \
               target_section.stat_defined + " apply to " + person_name + "?"
    else:
        return write_facts(person_name, person_type) + " Does " + \
               target_section.stat_used + " apply to " + person_name + "?"


if args.termtype == "nonces":
    raw_nonce_list = generate_synstat.read_nonces()
else:
    raw_nonce_list = generate_synstat.generate_systematic(statute_random)

total_statute_results = {"True Positive": 0, "True Negative": 0,
                         "False Positive": 0, "False Negative": 0, "unclear":0}
total_sentence_results = total_statute_results.copy()
total_num = 0

for run_num in range(args.numruns):
    if 0 < args.max_num <= total_num:
        assert total_num == args.max_num, "should never go over"
        break  # if we go over the total number allowed, stop further calls

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("run_num=", run_num)

    nonce_list = raw_nonce_list.copy()
    statute_random.seed(run_num) # re-seeding right before new shuffle fixes unexpected-reshuffling issues
    statute_random.shuffle(nonce_list)

    abst = generate_synstat.generate_abstract(nonce_list, args.depth, args.width)

    curr_statute = generate_synstat.abstract_to_statute(abst)
    print(curr_statute)
    print("")

    # if we are doing many-statute N-shot prompting, we need to generate the N statutes
    Nshot_statutes = []
    if args.Nshot > 0 and args.Nshot_type in ["many_statute", "many_statute_same_pos"]:
        for i in range(args.Nshot):
            # note that the nonce_list items are .pop()'ed, which prevents reuse
            extra_abstract = generate_synstat.generate_abstract(nonce_list, args.depth, args.width)
            Nshot_statutes.append(extra_abstract)

    if args.do_sentences:
        sentences_form, num_sentences = generate_synstat.abstract_to_sentences(abst, "Sentence {:d}: ")
        print(sentences_form)
        print("")

    all_parts = generate_synstat.extract_all_used_parts(abst)

    statute_results = {"True Positive": 0, "True Negative": 0,
                       "False Positive":0, "False Negative": 0, "unclear":0}

    sentence_results = {"True Positive": 0, "True Negative": 0,
                     "False Positive":0, "False Negative": 0, "unclear":0}

    num_this_run = 0

    if args.subdivs == "leavesonly":
        parts_to_test = [p for p in all_parts if not p.has_children()]
    elif args.subdivs == "noleaves": # we are doing it by numbered sentence, so just store the sentence nums
        parts_to_test = [p for p in all_parts if p.has_children() and not p.parent is None]
    elif args.subdivs == "both":
        parts_to_test = [p for p in all_parts if not p.parent is None]
    else:
        assert False, "not implemented"

    for person_type in all_parts: # iterate over all the possible types for the person
        # iterate over all subsections that we might ask if it applies to the person
        for applies_target in parts_to_test:
            names_set = NAMES.copy() # we will pop names out
            Nshot_random.shuffle(names_set)
            person_name = names_set.pop() # this always used to be Alice but is now randomly selected to avoid bias

            if 0 < args.max_num <= total_num:
                assert total_num == args.max_num, "should never go over"
                break # if we go over the total number allowed, stop further calls

            groundtruth = does_A_apply_to_anyB(applies_target, person_type)

            if args.subdivs == "leavesonly":
                assert not applies_target.has_children()
            elif args.subdivs == "noleaves":
                assert applies_target.has_children()

            if not groundtruth is None:
                if skip_random.random() < args.skip_percent/100.0:
                    print("SKIPPING ", end="")
                    continue

                print("++++++++++++++++++++++++++++++")

                # Build the question to pass into GPT-3
                statute_prompt = ""
                if args.Nshot > 0 and args.Nshot_type in ["many_statute", "many_statute_same_pos"]:
                    statute_prompt = write_multistatute_Nshot_prompt(args, applies_target, Nshot_statutes, names_set)
                statute_prompt += curr_statute

                if args.Nshot > 0 and args.Nshot_type == "1statute":
                    examples = write_1statute_Nshot_prompt(args, applies_target, person_type, all_parts)
                    statute_prompt += examples
                    print("-----")

                statute_question = write_statute_facts_question(person_name, person_type, applies_target)
                if args.Nshot == 0: # We don't add this if we already have examples
                    statute_question += " Let's think step by step."
                statute_prompt += "\n" + statute_question

                SECOND_PROMPT = "\nTherefore, the answer (Yes or No) is"  # cf. Kojima et al. 2022 appendix A.5

                print(statute_question)
                print("-----")

                # Make the GPT-3 calls for the statutory reasoning version
                if not args.noGPT:
                    utils.add_comment("Synthetic applies probe in " + __file__)
                    statute_response = utils.call_gpt3_withlogging(statute_prompt, args.model, max_tokens=2000)
                    utils.add_comment("Synthetic applies probe in " + __file__ + " SECOND PROMPT")
                    second_statute_prompt = statute_prompt + statute_response + SECOND_PROMPT
                    second_statute_response = utils.call_gpt3_withlogging(second_statute_prompt, args.model, max_tokens=2000)
                else:
                    statute_response = second_statute_response = ["No.","No","Yes.", "maybe?"][num_this_run % 4]

                statute_result = "unclear"
                if utils.is_yes(second_statute_response):
                    if groundtruth:
                        statute_result = "True Positive"
                    else:
                        statute_result = "False Positive"
                elif utils.is_no(second_statute_response):
                    if groundtruth:
                        statute_result = "False Negative"
                    else:
                        statute_result = "True Negative"
                statute_results[statute_result] += 1
                utils.add_comment("RESULT is " + statute_result)

                print(statute_response)
                print("-----")
                print(second_statute_response)
                print("-----")
                print("Groundtruth=", groundtruth, "so this is:", statute_result)
                print("-----")

                if args.do_sentences:
                    sentence_question = write_facts(person_name, person_type) + \
                                        " Does sentence " + str(applies_target.sentence_num) + " apply to " + \
                                        person_name + "?"
                    sentence_question += " Let's think step by step."
                    sentence_prompt = sentences_form + "\n" + sentence_question

                    print(sentence_question)
                    print("-----")

                    if not args.noGPT:
                        # Make the GPT-3 calls for the SENTENCE reasoning version
                        utils.add_comment("Synthetic applies probe SENTENCE VERSION in " + __file__ + " person_type=" + person_type.term + " sent_num=" + str(sent_num))
                        sentence_response = utils.call_gpt3_withlogging(sentence_prompt, args.model, max_tokens=2000)
                        utils.add_comment("Synthetic applies probe SENTENCE VERSION in " + __file__ + " SECOND PROMPT")
                        second_sentence_prompt = sentence_prompt + sentence_response + SECOND_PROMPT
                        second_sentence_response = utils.call_gpt3_withlogging(second_sentence_prompt, args.model, max_tokens=2000)
                    else:
                        sentence_response = second_sentence_response = statute_response

                    # now check SENTENCE response
                    sentence_result = "unclear"
                    if utils.is_yes(second_sentence_response):
                        if groundtruth:
                            sentence_result = "True Positive"
                        else:
                            sentence_result = "False Positive"
                    elif utils.is_no(second_sentence_response):
                        if groundtruth:
                            sentence_result = "False Negative"
                        else:
                            sentence_result = "True Negative"
                    sentence_results[sentence_result] += 1
                    utils.add_comment("RESULT is " + sentence_result)

                    print(sentence_response)
                    print("-----")
                    print(second_sentence_response)
                    print("-----")
                    print("Groundtruth=", groundtruth, "so this is:", sentence_result)
                    print("-----")

                num_this_run += 1
                total_num += 1
        print("")

    print("num_this_run=", num_this_run)
    print("This run statute_results:" , statute_results)
    if args.do_sentences:
        print("This run sentence_results: ", sentence_results)

    check_sum = 0
    for t in total_statute_results.keys():
        total_statute_results[t] += statute_results[t]
        total_sentence_results[t] += sentence_results[t]
        check_sum += total_statute_results[t]
    assert check_sum == total_num

    print("total_num=", total_num)
    print("so-far total_statute_results=", total_statute_results)
    statute_correct = (total_statute_results['True Positive'] + total_statute_results['True Negative'])
    print("so-far statute accuracy: {:.2f}".format(statute_correct/float(total_num)),
          "(" + str(statute_correct) + "/" + str(total_num) + ")")
    if args.do_sentences:
        print("so-far total_sentence_results=", total_sentence_results)
        sentence_correct = (total_sentence_results['True Positive'] + total_sentence_results['True Negative'])
        print("so-far sentence accuracy: {:.2f}".format(sentence_correct / float(total_num)),
              "(" + str(sentence_correct) + "/" + str(total_num) + ")")


suggested_filename = args.termtype+"_w"+ str(args.width)+ \
                     "_d"+str(args.depth)+"_"+ \
                     str(args.numruns)+"runs"
if args.do_sentences and args.Nshot == 0:
    suggested_filename += "_dosents"
elif args.Nshot > 0:
    suggested_filename += "_" + str(args.Nshot) + "shot"

suggested_filename += ".txt"

end = datetime.now()
print("End=", end)
print("Time taken=", end-start)

print("Suggested filename:", suggested_filename)
print('\a\a\a\a\a\a') # play sounds