# This is designed to test the ability of GPT* to retrieve text from a deposition transcript
import argparse, random, sys, re, datetime
sys.path.append('../')
import utils
import transcript_utils
from collections import Counter

parser = argparse.ArgumentParser(description='Generate synthetic statutes and questions to pass to GPT')
parser.add_argument('--numlines', type=int, required=True,
                    help='number of lines per simulated page')
parser.add_argument('--numruns', type=int, required=True,
                    help='number of simulated pages to run')
DEFAULT_MODEL = "text-davinci-003"
parser.add_argument('--model', default=DEFAULT_MODEL,
                    help='which openai model to use')
args = parser.parse_args()
print("args=", args)
print(datetime.datetime.now())

def standardize_line(line:str) -> str:
    return " ".join(re.split(r'\W+', line)).lower().strip()

def standardize_transcript(lines) -> str:
    rv = ""
    for line in lines:
        rv += standardize_line(line) + " "
    return standardize_line(rv)

# sometimes GPT returns a bracketed line number at the start, so we need to filter that
def strip_bracket(line:str, args) -> str:
    m = re.match("^\[(?P<num>\d+)\](?P<rest>.+)", line)
    if m is not None and (1 <= int(m.group("num")) <= args.numlines):
        return m.group("rest")
    else:
        return line

transcripts = transcript_utils.load_transcripts(args.numlines)
random.seed(42) # ensures reproducability
random.shuffle(transcripts)

count_calls = 0
count_wrong = 0
histogram_errors = Counter() # stores strings describing the errors and counts
histogram_wronglines_relative = Counter() # stores counts of relative position of line RETURNED
histogram_wronglines_absolute = Counter() # stores counts of absolute position of line ASKED FOR

assert args.numruns <= len(transcripts)

for idx_run in range(args.numruns) :
    transcript = transcripts[idx_run]
    print("idx_run=", idx_run)
    question_index = random.randint(0, args.numlines-1)
    query = transcript_utils.list_to_text(transcript)
    query += "\nWhat is the exact text at line " + str(question_index+1) + "?" # use 1-indexing for query itself
    correct_answer = transcript[question_index]

    print("QUERY:")
    print(query)

    messages = [{"role": "user", "content": query}]
    count_calls += 1
    gpt_response = utils.call_gpt_withlogging(messages, args.model, max_tokens=1000)
    print("RESPONSE:", gpt_response.strip())
    print("CORRECT: ", correct_answer.strip())

    # We find a proper match leniently, looking solely at the alphanumerical characters, so
    # punctuation issues or spacing issues are stripped before comparison.  Also, casing is
    # all removed, with everything lower.
    gpt_response_std = " ".join(re.split(r'\W+', strip_bracket(gpt_response, args))).lower().strip()
    correct_answer_std = " ".join(re.split(r'\W+', correct_answer)).lower().strip()

    correct = False
    if gpt_response_std == correct_answer_std:
        correct = True
        print("Correct first time")

    quoted = re.match("\w+.+[\"“‘'](?P<text>.+)[\"”’'][.]?$", gpt_response.strip())
    if not correct and quoted is not None:
        gpt_response_std = " ".join(re.split(r'\W+', strip_bracket(quoted.group("text"), args))).lower().strip()
        if gpt_response_std == correct_answer_std:
            print("Correct upon quote extraction")
            correct = True

    if not correct:
        count_wrong += 1
        print("WRONG!")
        histogram_wronglines_absolute.update([question_index])
        error_description = None
        # Now we need to analyze the error
        # We give priority to identifying it as another line
        other_line_error = False
        for idx_line in range(len(transcript)):
            if idx_line != question_index and \
                    gpt_response_std in standardize_line(transcript[idx_line]):
                other_line_error = True
                histogram_wronglines_relative.update([idx_line - question_index])  # may rarely be more than one; that's OK
        if other_line_error:
            error_description = "wrong line (single)"
        elif correct_answer_std in gpt_response_std:
            error_description = "superset"
        elif gpt_response_std in correct_answer_std:
            error_description = "subset"
        elif gpt_response_std in standardize_transcript(transcript):
            error_description = "wrong line (multiple)"
        else:
            error_description = "not present"
        print("ERROR DESCRIPTION:", error_description)
        histogram_errors.update([error_description])

    print("**** STATS:  count_calls=", count_calls, "count_wrong=", count_wrong,
          "accuracy={:.3f}".format(float(count_calls-count_wrong)/count_calls))
    print("histogram_errors: -------------------")
    histogram_errors_list = list(histogram_errors.items())
    histogram_errors_list.sort(key=lambda x: x[1], reverse=True)  # sort by COUNT, not errors
    for x in histogram_errors_list:
        print("{:5d}".format(x[1]), " ", x[0])
    print("histogram_wronglines_relative: -------------------")
    if len(histogram_wronglines_relative) == 0:
        print("nothing in histogram_wronglines_relative")
    else:
        histogram_wronglines_relative_list = list(histogram_wronglines_relative.items())
        histogram_wronglines_relative_list.sort(key=lambda x: x[0], reverse=True) # sort by relative location, not numbers
        for x in histogram_wronglines_relative_list:
            print("{:5d} {:5d}".format(x[0], x[1]))
    print("histogram_wronglines_absolute: -------------------")
    if len(histogram_wronglines_absolute) == 0:
        print("nothing in histogram_wronglines_absolute")
    else:
        histogram_wronglines_absolute_list = list(histogram_wronglines_absolute.items())
        histogram_wronglines_absolute_list.sort(key=lambda x: x[0], reverse=True) # sort by absolute location, not numbers
        for x in histogram_wronglines_absolute_list:
            print("{:5d} {:5d}".format(x[0], x[1]))


print(datetime.datetime.now())
suggested_filename = "textatline_lines" + str(args.numlines) + "_runs" + str(args.numruns) + "_" + \
    args.model + ".txt"
print("Suggested filename:", suggested_filename)