# This is a real-statute version of the synthetic-statute defined_at_synstat.py
import xml.etree.ElementTree as ET
from real_stat_utils import StatLine, FLUSH_LANGUAGE, load_statutes
from collections import Counter
import tiktoken
import argparse, random, sys, re, os, numpy, datetime
sys.path.append('../')
import utils

parser = argparse.ArgumentParser(description='Test the ability of GPT-* to find the section')
parser.add_argument('--numcalls', type=int, # required=True,
                    help='how many GPT calls to make')
DEFAULT_MODEL = "text-davinci-003"
parser.add_argument('--model', default=DEFAULT_MODEL,
                    help='which openai model to use')
parser.add_argument('--limitGPT3', default=True,
                    help='whether to limit calls to those that would fit within GPT-3\'s 4000 token limit')
parser.add_argument('--replacenumtitle', action='store_true',
                    help='whether to substitute the actual section number and title with a fake one')

args = parser.parse_args()
print("args=", args)
print(datetime.datetime.now())
order_random = random.Random(42) # used for shuffling


class DefinedLine:
    def __init__(self, list_statlines, linenum, term):
        self.list_statlines = list_statlines
        self.linenum = linenum
        self.term = term

    def get_full_stat_text(self) -> str:
        rv = ""
        for l in self.list_statlines:
            rv += l.text.rstrip() + "\n"
        return rv

# returns list of
def get_candidates() -> list:
    _, list_list_statlines = load_statutes(1000) # 1000 means basically don't collect leaves
    rv = []

    for statute in list_list_statlines:
        for idx_line, line in enumerate(statute):
            if line.identifier != FLUSH_LANGUAGE: # cannot use definitions in flush language
                matches = re.findall('["“]([^"”]+)["”]\smeans', line.text)
                if matches is not None and len(matches) == 1:
                    rv.append(DefinedLine(statute, idx_line, matches))
                elif matches is not None and len(matches) > 1:
                    print("Interesting, more than one match", line.text, matches)
    return rv

def run_tests():
    list_lines = get_candidates()
    order_random.shuffle(list_lines)

    gpt3_tokenizer = tiktoken.encoding_for_model("text-davinci-003")
    curr_tokenizer = tiktoken.encoding_for_model(args.model)

    count_calls = 0
    count_wrong = 0
    histogram_errors = Counter()

    for leaf in list_lines:
        if count_calls >= args.numcalls:
            break # done

        statute_text = leaf.get_full_stat_text()
        cite_text = leaf.get_cite()
        if args.replacenumtitle:
            idx_first_newline = statute_text.find("\n")
            assert idx_first_newline > 0
            idx_first_paren = cite_text.find("(")
            assert idx_first_paren > 0
            statute_text = "Section 1001.  Key provisions." + statute_text[idx_first_newline:]
            cite_text = "1001" + cite_text[idx_first_paren:]

        question = utils.TEXT_AT_STRING + cite_text + "?"
        query = statute_text + "\n" + question

        # This test ensures comparability of GPT-4 and GPT-3, despite different-sized token windows
        GPT3_LIMIT = 4000
        if args.limitGPT3:
            TOKEN_BUFFER = 300
            gpt3_encoding = gpt3_tokenizer.encode(query + leaf.get_line_text())
            num_gpt3_tokens = len(gpt3_encoding)
            if num_gpt3_tokens > (GPT3_LIMIT-TOKEN_BUFFER):
                print("Skipped", leaf.statlines[leaf.linenum].identifier, "due to exceeding gpt-3 capacity")
                continue

        # Here we do calculations to get the correct number to pass for max_tokens
        if args.model == DEFAULT_MODEL:
            MAX_TOKENS = GPT3_LIMIT
        elif "gpt-4" in args.model:
            MAX_TOKENS = 8000
        else:
            assert False, "need to add appropriate MAX_TOKENS"
        max_tokens_back = MAX_TOKENS - len(curr_tokenizer.encode(query))

        # Actually pass to the model
        messages = [{"role": "user", "content": query}]
        print(leaf.statlines[leaf.linenum].identifier, "*****************************")
        print(query)
        response = utils.call_gpt_withlogging(messages, args.model, max_tokens=max_tokens_back)
        # if response == None:
        #     continue # this is the result of being too long; don't count towards stats, since never got actual response
        count_calls += 1

        correct_answer = leaf.get_line_text()
        print("correct:", correct_answer)
        print("response:", response.strip())

        # determine whether this is correct
        has_error = False
        if is_statute_in_response(correct_answer, response, True): # Correct
            list_charlen_correct.append(len(statute_text))
            list_newlinelen_correct.append(len(statute_text.split("\n")))
        else:
            # Do error analysis
            print("WRONG!")
            has_error = True
            list_charlen_wrong.append(len(statute_text))
            list_newlinelen_wrong.append(len(statute_text.split("\n")))
            list_wrong_statutes.append(query + "\n\nRESPONSE:" + response)
            count_wrong+=1
            list_percentiles_wrong.append(leaf.percentile)
            errors = None
            # Example of an actual subset-wrong-way error:
            #   correct: the transferor or a person who bears a relationship to the transferor described in section 267(b) or 707(b), and
            #   response: The exact text at section 304(b)(5)(A)(i)(II) is: "who bears a relationship to the transferor described in section 267(b) or 707(b)."
            subset_wrong_way = False
            if is_statute_in_response(response, leaf.get_line_text_withnum()): # include num, since more likely to identify this error
                subset_wrong_way = True
            qmarks = [idx for idx, c in enumerate(response) if c in ['\"', '\'', '“', "”", "‘", "’"]]
            if len(qmarks) >= 2 and \
                is_statute_in_response(response[qmarks[0]+1:qmarks[-1]], leaf.get_line_text_withnum()): # include num, since more likely to identify this error
                subset_wrong_way = True
            if subset_wrong_way:
                errors = [EXCERPTED_SUBSET]
            else: # so, we don't have an wrong-way-subset error.  Let's figure out the problem.
                # find which lines were actually returned
                returned_lines = set()
                for line in leaf.statlines[1:]: # skip the 0th line, which is the section num and title
                    if is_statute_in_response(line.get_line_text(), response):
                        returned_lines.add(line)
                print("*** len(returned_lines)=", len(returned_lines))
                if len(returned_lines) == 0:
                    errors = [NOT_FOUND]
                    print("FURTHER ANALYSIS REQUIRED - Response not found in statute")
                else:
                    # analyze the type of error
                    for line in returned_lines:
                        print(line)
                        line_errors = analyze_error(leaf.get_identifier(), line.identifier)
                        if errors is None:
                            errors = line_errors
                        elif errors == NOT_PARALLEL and line_errors != NOT_PARALLEL:
                            # we will always prefer errors involving same level (i.e. parallel) to those of different levels
                            errors = line_errors
                        elif errors != NOT_PARALLEL and line_errors != NOT_PARALLEL and \
                                len(errors) > len(line_errors):
                            errors = line_errors

            if errors is None:
                error_text = NOT_FOUND  # text not even found
            elif errors == NOT_PARALLEL:
                error_text = errors
            else:
                error_text = ",".join(sorted(errors)) # sorting ensures consistent ordering
            print("error recorded: ", error_text)
            histogram_errors.update([error_text])

        # print detailed error information only if there was just an error found or we are at end
        if has_error or count_calls == args.numcalls:
            print("percentile errors: -------------------")
            for decile in range(10):
                if decile == 9:
                    count = len([x for x in list_percentiles_wrong if 0.9 <= x])
                else:
                    count = len([x for x in list_percentiles_wrong if 0.1 * decile <= x < 0.1 * (1 + decile)])
                print("{:.1f}-{:.1f}: {:4d}".format(0.1 * decile, 0.1 * (1 + decile), count))

            print("histogram_errors: -------------------")
            histogram_errors_list = list(histogram_errors.items())
            histogram_errors_list.sort(key=lambda x: x[1], reverse=True)  # sort by COUNT, not errors
            for x in histogram_errors_list:
                print("{:5d}".format(x[1]), " ", x[0])

        print("count_calls=", count_calls, " count_wrong=", count_wrong,
              " accuracy={:.3f}".format((count_calls-count_wrong)/float(count_calls)))

    print("SMALLEST NUMBER CHARACTERS ERRORS:")
    list_wrong_statutes.sort(key=lambda x: len(x))
    for i in range(min(5, len(list_wrong_statutes))):
        print("**** i=", i, " num chars=", len(list_wrong_statutes[i]))
        print(list_wrong_statutes[i].split("\n")[0])
    print("SMALLEST NUMBER LINES ERRORS:")
    list_wrong_statutes.sort(key=lambda x: len(x.split("\n")))
    for i in range(min(5, len(list_wrong_statutes))):
        print("**** i=", i, " num newlines=", len(list_wrong_statutes[i].split("\n")))
        print(list_wrong_statutes[i].split("\n")[0])
    print("list_charlen_correct=", list_charlen_correct)
    print("avg = ", numpy.mean(list_charlen_correct))
    print("list_charlen_wrong =", list_charlen_wrong)
    print("avg=", numpy.mean(list_charlen_wrong))
    print("list_newlinelen_correct =", list_newlinelen_correct)
    print("avg=", numpy.mean(list_newlinelen_correct))
    print("list_newlinelen_wrong =", list_newlinelen_wrong)
    print("avg=", numpy.mean(list_newlinelen_wrong))

    print(datetime.datetime.now())

    suggested_filename = "realtextat_n" + str(args.numcalls) + "_minD" + str(args.mindepth)
    if not args.limitGPT3: # if we depart from the comparability baseline, then MENTION it
        suggested_filename += "_NOgptLimit"
    suggested_filename += "_" + args.model + ".txt"
    print("Suggested filename:", suggested_filename)

if __name__ == "__main__":
    run_tests()