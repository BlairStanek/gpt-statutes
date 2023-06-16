# This tests on the simple question of what the text is at a particular section.

import generate_synstat
from synthetic_stat_utils import NOT_PARALLEL, NOT_FOUND, \
    analyze_error, reformat_with_lines, print_stats
import sys, argparse, re
sys.path.append('../')
import utils
import random
from datetime import datetime
from collections import Counter

parser = argparse.ArgumentParser(description='Generate synthetic statutes and questions to pass to GPT')
parser.add_argument('--width', required=True, type=int,
                    help='Width of the synthetic tree to generate; 2 thru 5 are common')
parser.add_argument('--depth', required=True, type=int,
                    help='Depth of the synthetic tree to generate; 2 thru 5 are common')
parser.add_argument('--numruns', default=1, type=int,
                    help='These are the basic types of prompting we handle')
DEFAULT_MODEL = "text-davinci-003"
parser.add_argument('--model', default=DEFAULT_MODEL,
                    help='which openai model to use')
parser.add_argument('--do_lines', action='store_true',
                    help='whether to also try semantically-equivalent line numbers')
args = parser.parse_args()

print("args=", args)
print(datetime.now())

def find_line_num(part, lines, included_term=None):
    rv = None  # returns line number of correct answer
    for idx_line, line in enumerate(lines):
        if part.term.lower() in line and (included_term is None or included_term in line):
            assert rv is None, "Should only appear once"
            rv = idx_line
    assert not rv is None, "Should have found one line with the right answer"
    return rv

###### END OF HELPER FUNCTIONS #######

raw_nonce_list = generate_synstat.read_nonces()

statute_random = random.Random(42) # used for shuffling to get the statute
count_wrong = 0
count_calls = 0
count_overinclusive = 0 # when it returns more than just the expected text
count_wrong_and_overinclusive = 0
count_wrong_last_child = 0 # if wrong and it is the last child
count_forgot_any = 0
histogram_wrong = Counter() # keeps histogram of the subsection where wrong
histogram_relative_loc = Counter()
histogram_absolute_loc = Counter()
histogram_errors = Counter()
count_list_calls = 0 # used only with --do_lines
count_list_wrong = 0 # used only with --do_lines


for idx_run in range(args.numruns):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ idx_run=", idx_run)
    nonce_list = raw_nonce_list.copy()
    statute_random.seed(idx_run)  # re-seeding right before new shuffle fixes unexpected-reshuffling issues
    statute_random.shuffle(nonce_list)
    abst = generate_synstat.generate_abstract(nonce_list, args.depth, args.width)

    if idx_run == 0:
        print_stats(abst)

    # sec_num = statute_random.randint(1010, 9999)

    curr_statute = generate_synstat.abstract_to_statute(abst, keep_compact=True)
    curr_statute_lines = curr_statute.split("\n")
    # curr_statute_lines_standardized = \
    #     [standardize_text(line) for line in curr_statute_lines if len(standardize_text(line)) > 0]

    all_parts = generate_synstat.extract_all_used_parts(abst)
    leaves_only = [p for p in all_parts if not p.has_children()]

    dict_terms = dict() # store terms for later
    for part in all_parts:
        dict_terms[part.term.lower()] = part

    print(curr_statute)

    line_formatted = None
    if args.do_lines: # we do a comparison to simply having the text with line numbers
        line_formatted = reformat_with_lines(curr_statute)
        print(line_formatted)

    for leaf in leaves_only:
        question = "What is the exact text at " + leaf.stat_used + "?"
        query = curr_statute + "\n\n" + question

        messages = [{"role": "user", "content": query}]

        print(question)
        count_calls += 1
        statute_response = utils.call_gpt_withlogging(messages, args.model, max_tokens=1000)

        # sometimes GPT-* returns not just a single line, but multiple lines; find that here
        is_overinclusive = False
        count_within = 0
        for part in all_parts:
            if part.term.lower() in statute_response.lower():
                count_within += 1
        if count_within > 1:
            is_overinclusive = True
            count_overinclusive += 1

        print(statute_response.strip())
        if leaf.term.lower() in statute_response.lower():
            if ("any " + leaf.term.lower()) not in statute_response.lower():
                count_forgot_any += 1
        else:
            print("WRONG!")
            count_wrong += 1
            histogram_absolute_loc.update([leaves_only.index(leaf)]) # store absolute location
            histogram_wrong.update([leaf.stat_used])
            if is_overinclusive:
                count_wrong_and_overinclusive += 1

            # How often are these errors occurring for the last child of the parent
            if leaf == leaf.parent.children[-1]:
                count_wrong_last_child += 1

            # Start analyzing errors
            correct_linenum = find_line_num(leaf, curr_statute_lines)
            response_words = re.split(r'\W+', statute_response.lower())
            returned_parts = set() # actual statute objects
            returned_cites = set() # strings of the citation in question (non-leaf parts have 2 of these)
            returned_linenums = set() # ints of the line where came from
            for idx_word, word in enumerate(response_words):
                if word in dict_terms:
                    part = dict_terms[word]
                    returned_parts.add(part)
                    if part.has_children(): # not a leaf
                        # we must figure out whether it is being referred to in definition or being used
                        add_defines = False
                        add_used = False
                        if (idx_word < len(response_words)-1 and response_words[idx_word+1] == "means") or \
                             (idx_word < len(response_words) - 2 and response_words[idx_word + 2] == "means") or \
                             (idx_word > 0 and response_words[idx_word-1] == "term"):
                            add_defines = True
                        elif idx_word > 0 and response_words[idx_word-1] == "any":
                            add_used = True
                        elif word not in response_words[idx_word+1:]:
                            # In this case, this word does not appear again later in the response with context
                            # that could clarify, so we return BOTH.
                            add_defines = True
                            add_used = True
                            if ("any" in response_words) or ("means" in response_words):
                                print("ODDITY TO INVESTIGATE: unclear whether definition or used")

                        if add_defines:
                            returned_cites.add(part.stat_defined) # when defined "means" appears
                            returned_linenums.add(find_line_num(part, curr_statute_lines, "means"))
                        if add_used:
                            returned_cites.add(part.stat_used) # when used "any" comes before term
                            returned_linenums.add(find_line_num(part, curr_statute_lines, "any"))
                    else: # a leaf -- the simplest case
                        assert part.stat_defined is None
                        returned_cites.add(part.stat_used)
                        returned_linenums.add(find_line_num(part, curr_statute_lines))
            print("returned_cites=", returned_cites)
            print("returned_parts' terms=", [x.term for x in returned_parts])
            print("correct_linenum=", correct_linenum)
            print("returned_linenums=", returned_linenums)

            # find the closest line to the correct one and store relative problem
            rel_loc = None
            for linenum in returned_linenums:
                if rel_loc == None or (abs(linenum - correct_linenum) < abs(rel_loc)):
                    rel_loc = linenum - correct_linenum
            print("rel_loc=", rel_loc)
            histogram_relative_loc.update([rel_loc])

            # analyze the type of error
            minimal_error = None # there can be multiple items returned; choose the closest
            for incorrect_cite in returned_cites:
                errors = analyze_error(leaf.stat_used, incorrect_cite)
                if minimal_error is None:
                    minimal_error = errors
                elif minimal_error == NOT_PARALLEL and errors != NOT_PARALLEL:
                    # we will always prefer errors involving same level (i.e. parallel) to those of different levels
                    minimal_error = errors
                elif minimal_error != NOT_PARALLEL and errors != NOT_PARALLEL and \
                    len(minimal_error) > len(errors):
                    minimal_error = errors

            if minimal_error is None:
                minimal_error_text = NOT_FOUND # text not even found
            elif minimal_error == NOT_PARALLEL:
                minimal_error_text = minimal_error
            else:
                minimal_error_text = ",".join(sorted(minimal_error))
            print("minimal error: ", minimal_error_text)
            histogram_errors.update([minimal_error_text])

        if args.do_lines:
            # find the line label corresponding to this leaf
            matching_line = None
            for line in line_formatted.split("\n"):
                if leaf.term.lower() in line:
                    assert matching_line is None, "should not be two for leaves"
                    matching_line = line
            assert matching_line is not None
            line_label = matching_line[0: matching_line.find(":")]
            question = "What is the exact text at line " + line_label + "?"
            query = line_formatted + "\n\n" + question
            messages = [{"role": "user", "content": query}]
            print(question)
            count_list_calls +=1
            statute_response = utils.call_gpt_withlogging(messages, args.model, max_tokens=1000)
            print("response:", statute_response.strip())
            if leaf.term.lower() not in statute_response:
                print("WRONG on list!")
                count_list_wrong +=1

    print("count_overinclusive=", count_overinclusive)
    print("count_wrong_and_overinclusive=", count_wrong_and_overinclusive)
    print("count_wrong_last_child=", count_wrong_last_child)
    print("count_forgot_any=", count_forgot_any)

    print("\nhistogram_wrong_list: -------------------")
    histogram_wrong_list = list(histogram_wrong.items())
    histogram_wrong_list.sort(key=lambda x: x[0])
    for x in histogram_wrong_list:
        print(x[0], " ", x[1])

    print("histogram_relative_loc_list: -------------------")
    histogram_relative_loc_list = list(histogram_relative_loc.items())
    histogram_relative_loc_list.sort(key=lambda x: x[0])
    for x in histogram_relative_loc_list:
        print(x[0], " ", x[1])

    print("histogram_absolute_loc_list: -------------------")
    histogram_absolute_loc_list = list(histogram_absolute_loc.items())
    histogram_absolute_loc_list.sort(key=lambda x: x[0])
    for x in histogram_absolute_loc_list:
        print(x[0], " ", x[1])

    print("histogram_errors: -------------------")
    histogram_errors_list = list(histogram_errors.items())
    histogram_errors_list.sort(key=lambda x: x[1], reverse=True)  # sort by COUNT, not errors
    for x in histogram_errors_list:
        print("{:5d}".format(x[1]), " ", x[0])
    print("NOTE: len(leaves_only)=", len(leaves_only))

    print("****** count_wrong =", count_wrong, " of ", count_calls,
          " accuracy = {:.2f}".format(float((count_calls-count_wrong)/count_calls)))

    if args.do_lines:
        print("*** LIST STATS: count_list_wrong=",count_list_wrong,
              "count_list_calls=",count_list_calls,
              "accuracy = ", (count_list_calls-count_list_wrong)/float(count_list_calls))

print(datetime.now())

sugg_filename = "textat_d" + str(args.depth) + "w" + str(args.width) + \
      "n" + str(args.numruns) + "_" + args.model
if args.do_lines:
    sugg_filename += "_dolines"
sugg_filename += ".txt"

print("Suggested filename:", sugg_filename)
