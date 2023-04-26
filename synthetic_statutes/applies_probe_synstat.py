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
runsubset_random = random.Random(42) # used to select which items in a run to select

parser = argparse.ArgumentParser(description='Generate synthetic statutes and questions to pass to GPT3')
parser.add_argument('--width', required=True, type=int,
                    help='Width of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--depth', required=True, type=int,
                    help='Depth of the synthetic tree to generate; 2 or 3 are common')
parser.add_argument('--termtype', required=True, choices=["nonces", "ids"],
                    help='These are the basic types of prompting we handle')
parser.add_argument('--numruns', required=True, type=int,
                    help='These are the basic types of prompting we handle')
parser.add_argument('--perrun', type=int, default=0,
                    help='how many of each (positive & negative) to do each run')
parser.add_argument('--do_sentences', action="store_true",
                    help='do the semantically identical sentences in addition to the statutes')
parser.add_argument('--noGPT', action="store_true",
                    help='for debugging; if passed, just generate statutes but do not actually call GPT')
parser.add_argument('--Nshot', required=False, type=int, default=0,
                    help='Allows for N shot with N examples (must be EVEN, so positive and negative balanced)')
parser.add_argument('--Nshot_type', choices=["1", "N", "N_samepos", "N/2", "N/2_samepos"],
                    help='whether to do N-shot with N questions or N statutes (including same position)')
parser.add_argument('--max_num', type=int, default=0,
                    help='stop after this number of queries')
parser.add_argument('--subdivs', required=True, choices=["leavesonly", "noleaves", "both"],
                    help='which type of subdivisions to consider asking about')
parser.add_argument('--model', default="text-davinci-003",
                    help='which openai model to use')
parser.add_argument('--question_form', type=int, default=0,
                    help='how to phrase question; default is "Does section __ apply to __?"')


args = parser.parse_args()

if args.Nshot_type in ["1", "N/2", "N/2_samepos"] and \
    args.Nshot % 2 == 1:
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
# The below were drawn from the top 15 baby names each at https://www.ssa.gov/oact/babynames/decades/names2000s.html.
MALE_NAMES = [ "Jacob", "Michael", "Joshua", "Matthew", "Daniel", "Christopher", "Andrew",
         "Ethan", "Joseph", "William", "Anthony", "David", "Alexander", "Nicholas", "Ryan"]
FEMALE_NAMES = ["Emily", "Madison", "Emma", "Olivia","Hannah","Abigail","Isabella","Samantha",
            "Elizabeth","Ashley","Alexis","Sarah","Sophia","Alyssa","Grace"  ]
NAMES = MALE_NAMES.copy().extend(FEMALE_NAMES) # merge


def write_multistatute_Nshot_prompt(args, Nshot_statutes, names_set, test_applies_to, test_person_type) -> str:
    if not args.Nshot_type in ["N_samepos", "N/2_samepos"]:
        test_applies_to = test_person_type = None # ensures not used if not appropriate

    rv = ""
    used_sec_nums = set()

    # to ensure balance (not too many false examples or too many true examples)
    # we generate a balanced list of positives and negatives and then randomly shuffle
    list_positive = [True] * int(len(Nshot_statutes) / 2)
    list_positive.extend([False] * int(len(Nshot_statutes) / 2))
    if len(Nshot_statutes) % 2 == 1:
        list_positive.append(Nshot_random.random() < 0.5)
    Nshot_random.shuffle(list_positive)
    assert -1 <= (len([x for x in list_positive if x]) - len([x for x in list_positive if not x])) <= 1

    for abst in Nshot_statutes:
        # randomly generate a statute
        sec_num = None
        while sec_num is None or sec_num in used_sec_nums:
            sec_num = Nshot_random.randint(1010, 9999)
        used_sec_nums.add(sec_num)

        # append the statute to the prompt
        rv += generate_synstat.abstract_to_statute(abst, sec_num=sec_num) + "\n"

        all_parts = generate_synstat.extract_all_used_parts(abst)

        name1 = names_set.pop() # the names set was already randomly shuffled
        name2 = names_set.pop()

        # both questions will have the same applies-to-target.
        if args.Nshot_type in ["N_samepos", "N/2_samepos"]:
            target_subsection = test_applies_to.get_analogous_item(abst)
        else: # choose an applies-to target of a type that might be chosen for actual test case
            if args.subdivs == "leavesonly":
                possible_target_subsection = [p for p in all_parts if not p.has_children()]
            elif args.subdivs == "noleaves":  # we are doing it by numbered sentence, so just store the sentence nums
                possible_target_subsection = [p for p in all_parts if p.has_children() and not p.parent is None]
            elif args.subdivs == "both":
                possible_target_subsection = [p for p in all_parts if not p.parent is None]
            else:
                assert False, "not implemented"
            target_subsection = Nshot_random.choice(possible_target_subsection)

        true_type = \
            Nshot_random.choice([p for p in all_parts if True == does_A_apply_to_anyB(target_subsection, p)])
        false_type = \
            Nshot_random.choice([p for p in all_parts if False == does_A_apply_to_anyB(target_subsection, p)])

        if args.Nshot_type in ["N_samepos", "N/2_samepos"]: # exactly match in position to test type
            if does_A_apply_to_anyB(test_applies_to, test_person_type):
                true_type = test_person_type.get_analogous_item(abst)
            else:
                false_type = test_person_type.get_analogous_item(abst)

        # For N-statute N-shot, generate a true example with 50% chance
        # For N/2-statute N-shot, generate the true example first with 50% chance
        if list_positive.pop():
            rv += write_statute_facts_question(name1, true_type, target_subsection)
            rv += write_explanation_true(name1, true_type, target_subsection)
            if args.Nshot_type.startswith("N/2"): # now generate a false example
                rv += write_statute_facts_question(name2, false_type, target_subsection)
                rv += write_explanation_false(name2, target_subsection)
        else:
            rv += write_statute_facts_question(name1, false_type, target_subsection)
            rv += write_explanation_false(name1, target_subsection)
            if args.Nshot_type.startswith("N/2"): # now generate a true example
                rv += write_statute_facts_question(name2, true_type, target_subsection)
                rv += write_explanation_true(name2, true_type, target_subsection)

    assert len(list_positive) == 0, "Should have exactly used up our list of positives or negatives"
    return rv

# This is used for the N-shot prompting where we have a single statute (which is used for test as well)
# and create N questions (half yes, half no) that come right before the test question.
def write_1statute_Nshot_prompt(args, names_set, test_applies_to, test_person_type, all_parts) -> str:
    assert (args.Nshot % 2) == 0
    already_used_subsections = set()
    rv = ""

    for i in range(0, args.Nshot, 2): # do in pairs
        name1 = names_set.pop()
        name2 = names_set.pop()

        # Both Yes and No questions will be based on the same reference section
        # We randomly choose the reference section.
        example_level = args.depth - 1
        candidate_subsections = []
        for x in all_parts:
            if x.get_level() == example_level:
                if False == does_A_apply_to_anyB(test_applies_to, x) and \
                        False == does_A_apply_to_anyB(x, test_applies_to) and \
                        not x in already_used_subsections:
                    candidate_subsections.append(x)
        assert len(candidate_subsections) > 0, "Too few subsections for N shot with N=" + str(args.Nshot)

        # We ideally want an example section that doesn't apply to the person type
        ideal_candidate_subsections = [x for x in candidate_subsections if False == does_A_apply_to_anyB(x, test_person_type)]
        if len(ideal_candidate_subsections) > 0:
            candidate_subsections = ideal_candidate_subsections

        # We want the example to be as far as possible from both the person type and the target section
        candidate_subsections = select_furthest_items(candidate_subsections, test_person_type, test_applies_to)
        print("For test_person_type=", test_person_type.term, "and ", test_applies_to.stat_defined,
              "candidates:", [x.term + ":" + x.stat_defined for x in candidate_subsections])

        # choose the statute to be used in the few shots
        target_subsection = Nshot_random.choice(candidate_subsections)
        already_used_subsections.add(target_subsection)
        print("  chosen=", target_subsection.term + ":" + target_subsection.stat_defined)

        # choose the true target
        candidates_true = select_furthest_items(target_subsection.get_all_descendants(), test_person_type)
        print("true target candidates are:", [x.term + ":" + x.stat_used for x in candidates_true])
        true_item = Nshot_random.choice(candidates_true)
        print("  true_item chosen=", true_item.term + ":" + true_item.stat_used)
        assert does_A_apply_to_anyB(target_subsection, true_item) == True
        assert does_A_apply_to_anyB(test_applies_to, true_item) == False
        assert test_applies_to != true_item

        # choose the false target from valid ones, so that it is farthest from the test person type and test applies to
        candidates_false = []
        for x in all_parts:
            if False == does_A_apply_to_anyB(target_subsection, x):
                candidates_false.append(x)
        assert len(candidates_false) > 0
        candidates_false = select_furthest_items(candidates_false, test_person_type, test_applies_to)
        ideal_candidate_subsections_false = [x for x in candidates_false if False == does_A_apply_to_anyB(test_applies_to, x)]
        if len(ideal_candidate_subsections_false) > 0:
            candidates_false = ideal_candidate_subsections_false
        print("false target candidates are:", [x.term + ":" + x.stat_used for x in candidates_false])
        false_item = Nshot_random.choice(candidates_false)
        print("  false_item=", false_item.term + ":" + false_item.stat_used)
        assert does_A_apply_to_anyB(target_subsection, false_item) == False
        assert does_A_apply_to_anyB(test_applies_to, false_item) == False or (args.width == 2)
        assert test_applies_to != false_item

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

def write_statute_facts_question(person_name, person_type, target_section, args) -> str:
    rv = write_facts(person_name, person_type)

    if not target_section.stat_defined is None:
        section_name = target_section.stat_defined
    else:
        section_name = target_section.stat_used

    main_term = target_section.get_root().term.lower()
    if person_name in FEMALE_NAMES:
        pronoun = "her"
        assert not person_name in MALE_NAMES
    else:
        pronoun = "him"
        assert person_name in MALE_NAMES

    # Here are the possible options for question phrasing
    # 0. “Does section 1001(b) apply to Alice?” [DEFAULT]
    # 1. “Does section 1001(b) apply to Alice, making her a tammers?”
    # 2. “Does section 1001(b) apply to make Alice a tammers?”
    # 3. “Is Alice a tammers because of section 1001(b)?”
    # 4. "Is Alice a tammers owing to section 1001(b)?"
    # 5. "Is Alice a tammers as per section 1001(b)?"
    if args.question_form == 0:
        rv += " Does " + section_name + " apply to " + person_name + "?"
    elif args.question_form == 1:
        rv += " Does " + section_name + " apply to " + person_name +  ", making " + \
              pronoun + " " + generate_synstat.get_article(main_term) + " " + main_term + "?"
    elif args.question_form == 2:
        rv += " Does " + section_name + " apply to make " + person_name + \
              " " + generate_synstat.get_article(main_term) + " " + main_term + "?"
    elif args.question_form == 3:
        rv += " Is " + person_name + " " + generate_synstat.get_article(main_term) + \
              " " + main_term + " because of " + section_name + "?"
    elif args.question_form == 4:
        rv += " Is " + person_name + " " + generate_synstat.get_article(main_term) + \
              " " + main_term + " owing to " + section_name + "?"
    elif args.question_form == 5:
        rv += " Is " + person_name + " " + generate_synstat.get_article(main_term) + \
              " " + main_term + " as per " + section_name + "?"

    return rv

# This builds up all possible queries.  It returns a 2-tuple, with the first being for
# statutes and the second being sentence based (which is empty if we are not doing sentences in options).
# Each of the items in the 2-tuple is a list of possible prompts, which are stored as 2-tuples
# of (prompt text as a str, ground truth as a bool).
def build_possible_queries(args, all_parts, curr_statute):
    statute_queries = []
    sentence_queries = []

    if args.subdivs == "leavesonly":
        parts_to_test = [p for p in all_parts if not p.has_children()]
    elif args.subdivs == "noleaves": # we are doing it by numbered sentence, so just store the sentence nums
        parts_to_test = [p for p in all_parts if p.has_children() and not p.parent is None]
    elif args.subdivs == "both":
        parts_to_test = [p for p in all_parts if not p.parent is None]
    else:
        assert False, "not implemented"

    for person_type in all_parts: # iterate over all the possible types for the person
        for applies_target in parts_to_test: # iterate over all subsections to ask if applies
            names_set = NAMES.copy() # we will pop names out
            Nshot_random.shuffle(names_set)
            person_name = names_set.pop() # this always used to be Alice but is now randomly selected to avoid bias

            groundtruth = does_A_apply_to_anyB(applies_target, person_type)

            if args.subdivs == "leavesonly":
                assert not applies_target.has_children()
            elif args.subdivs == "noleaves":
                assert applies_target.has_children()

            if not groundtruth is None:

                # Build the question
                statute_prompt = ""
                if args.Nshot > 0 and args.Nshot_type.startswith("N"):
                    statute_prompt = write_multistatute_Nshot_prompt(args,
                                                                     Nshot_statutes,
                                                                     names_set,
                                                                     applies_target,
                                                                     person_type)
                statute_prompt += curr_statute

                if args.Nshot > 0 and args.Nshot_type == "1":
                    examples = write_1statute_Nshot_prompt(args, names_set, applies_target, person_type, all_parts)
                    statute_prompt += "\n" + examples
                    print("-----")

                statute_question = write_statute_facts_question(person_name, person_type, applies_target, args)
                if args.Nshot == 0: # We don't add this if we already have examples
                    statute_question += " Let's think step by step."
                statute_prompt = statute_prompt.rstrip() + "\n\n" + statute_question
                statute_queries.append((statute_prompt, groundtruth))

    return (statute_queries, sentence_queries)

def filter_and_balance_queries(args, possible_queries):
    positive_queries = [x for x in possible_queries if x[1]]
    negative_queries = [x for x in possible_queries if not x[1]]

    assert len(positive_queries) <= len(negative_queries), "When are there more positive examples??  Unexpected!"

    num_each = min(len(positive_queries), len(negative_queries))
    if args.perrun > 0:
        assert args.perrun <= num_each
        num_each = args.perrun

    filtered_pos = runsubset_random.sample(positive_queries, num_each)
    filtered_neg = runsubset_random.sample(negative_queries, num_each)

    filtered_pos.extend(filtered_neg)
    return filtered_pos


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

    # Generate the statute about which we will ask questions
    nonce_list = raw_nonce_list.copy()
    statute_random.seed(run_num) # re-seeding right before new shuffle fixes unexpected-reshuffling issues
    statute_random.shuffle(nonce_list)
    abst = generate_synstat.generate_abstract(nonce_list, args.depth, args.width)
    curr_statute = generate_synstat.abstract_to_statute(abst)

    # if we are doing many-statute N-shot prompting, we need to generate the N statutes used in prompting
    Nshot_statutes = []
    if args.Nshot > 0 and args.Nshot_type in ["N", "N_samepos", "N/2", "N/2_samepos"]:
        num_statutes = args.Nshot
        if args.Nshot_type in ["N/2", "N/2_samepos"]:
            num_statutes = int(args.Nshot / 2)

        for i in range(num_statutes):
            # note that the nonce_list items are .pop()'ed, which prevents reuse
            extra_abstract = generate_synstat.generate_abstract(nonce_list, args.depth, args.width)
            Nshot_statutes.append(extra_abstract)

    # if also doing sentence testing, generate the sentences about which we will ask questions
    if args.do_sentences:
        sentences_form, num_sentences = generate_synstat.abstract_to_sentences(abst, "Sentence {:d}: ")
        print(sentences_form)
        print("")

    all_parts = generate_synstat.extract_all_used_parts(abst)

    statute_results = {"True Positive": 0, "True Negative": 0,
                       "False Positive":0, "False Negative": 0, "unclear":0}
    sentence_results = statute_results.copy()

    num_this_run = 0

    # build up all possible queries
    possible_statute_queries, possible_sentence_queries = \
        build_possible_queries(args, all_parts, curr_statute)

    # filter the queries so that the positive/false are balanced and we have appropriate num
    statute_queries = filter_and_balance_queries(args, possible_statute_queries)

    # run thru and make the actual calls to GPT
    for query in statute_queries:
        print("----------")
        print(query[0]) # this is the text to pass to GPT

        # Make the GPT-3 calls for the statutory reasoning version
        if not args.noGPT:
            utils.add_comment("Synthetic applies probe in " + __file__)
            statute_response = utils.call_gpt3_withlogging(query[0], args.model, max_tokens=1000)

            SECOND_PROMPT = "\nTherefore, the answer (Yes or No) is"  # cf. Kojima et al. 2022 appendix A.5

            # for the N-shot prompting where there are 2 questions after each statute,
            # GPT3 will generally try to answer the first question and then produce and
            # answer a second question!  To address this, we need to construct a second
            # prompt that removes this second question & answer.
            construct_normal_second_prompt = True
            if args.Nshot_type in ["N/2", "N/2_samepos"]:
                if statute_response.count("\n\n") > 1:
                    print("POSSIBLE PROBLEM: More than one double carriage return in response.\n")
                if "\n\n" in statute_response:
                    construct_normal_second_prompt = False # turns off normal construction
                    second_statute_prompt = \
                        query[0] + \
                        statute_response.split("\n\n")[0] + \
                        SECOND_PROMPT

            utils.add_comment("Synthetic applies probe in " + __file__ + " SECOND PROMPT")
            if construct_normal_second_prompt:
                second_statute_prompt = query[0] + statute_response + SECOND_PROMPT
            second_statute_response = utils.call_gpt3_withlogging(second_statute_prompt, args.model, max_tokens=400)
        else:
            statute_response = second_statute_response = ["No.","No","Yes.", "maybe?"][num_this_run % 4]

        statute_result = "unclear"
        if utils.is_yes(second_statute_response):
            if query[1]: # this holds the groundtruth
                statute_result = "True Positive"
            else:
                statute_result = "False Positive"
        elif utils.is_no(second_statute_response):
            if query[1]: # this holds the groundtruth
                statute_result = "False Negative"
            else:
                statute_result = "True Negative"
        statute_results[statute_result] += 1
        utils.add_comment("RESULT is " + statute_result)

        print(statute_response)
        print("-----")
        print(second_statute_response)
        print("-----")
        print("Groundtruth=", query[1], "so this is:", statute_result)
        print("-----")

        # if args.do_sentences:
        #     sentence_question = write_facts(person_name, person_type) + \
        #                         " Does sentence " + str(applies_target.sentence_num) + " apply to " + \
        #                         person_name + "?"
        #     sentence_question += " Let's think step by step."
        #     sentence_prompt = sentences_form + "\n" + sentence_question
        #
        #     print(sentence_question)
        #     print("-----")
        #
        #     if not args.noGPT:
        #         # Make the GPT-3 calls for the SENTENCE reasoning version
        #         utils.add_comment("Synthetic applies probe SENTENCE VERSION in " + __file__ + " person_type=" + person_type.term + " sent_num=" + str(sent_num))
        #         sentence_response = utils.call_gpt3_withlogging(sentence_prompt, args.model, max_tokens=400)
        #         utils.add_comment("Synthetic applies probe SENTENCE VERSION in " + __file__ + " SECOND PROMPT")
        #         second_sentence_prompt = sentence_prompt + sentence_response + SECOND_PROMPT
        #         second_sentence_response = utils.call_gpt3_withlogging(second_sentence_prompt, args.model, max_tokens=400)
        #     else:
        #         sentence_response = second_sentence_response = statute_response
        #
        #     # now check SENTENCE response
        #     sentence_result = "unclear"
        #     if utils.is_yes(second_sentence_response):
        #         if groundtruth:
        #             sentence_result = "True Positive"
        #         else:
        #             sentence_result = "False Positive"
        #     elif utils.is_no(second_sentence_response):
        #         if groundtruth:
        #             sentence_result = "False Negative"
        #         else:
        #             sentence_result = "True Negative"
        #     sentence_results[sentence_result] += 1
        #     utils.add_comment("RESULT is " + sentence_result)
        #
        #     print(sentence_response)
        #     print("-----")
        #     print(second_sentence_response)
        #     print("-----")
        #     print("Groundtruth=", groundtruth, "so this is:", sentence_result)
        #     print("-----")

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
print('\a\a\a\a') # play sounds