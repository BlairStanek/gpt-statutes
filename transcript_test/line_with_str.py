# Test whether GPT can retrieve the text at a particular line.
import argparse, random, sys, re, datetime
sys.path.append('../')
import utils
import transcript_utils
import Levenshtein
from collections import Counter

parser = argparse.ArgumentParser(description='Test whether GPT can retrieve the transcript line with some text')
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

transcripts = transcript_utils.load_transcripts(args.numlines)
random.seed(42) # ensures reproducability
random.shuffle(transcripts)

count_calls = 0
count_wrong = 0
histogram_wronglines_relative = Counter() # stores counts of relative position of line RETURNED
histogram_wronglines_absolute = Counter() # stores counts of absolute position of line ASKED FOR

assert args.numruns <= len(transcripts)

for idx_run in range(args.numruns) :
    transcript = transcripts[idx_run]
    query = transcript_utils.list_to_text(transcript)

    print("************ idx_run=", idx_run)
    # we have to choose a line whose text is unique from the others
    possible_idxs = list(range(args.numlines))
    random.shuffle(possible_idxs)
    found_valid_line = False
    for idx in possible_idxs:
        candidate = transcript[idx].lower()
        if len(candidate.split()) >= 3: # only consider lines with at least 3 words
            conflict = False
            for other_idx in range(args.numlines):
                if other_idx != idx:
                    other = transcript[other_idx].lower()
                    LEVENSHTEIN_CUTOFF = 5
                    lev_dist = Levenshtein.distance(other, candidate,
                                                    weights=(1, 0, 1), # small weight for a deletion to eliminate near subsets
                                                    score_cutoff=LEVENSHTEIN_CUTOFF)
                    if candidate in other or lev_dist < LEVENSHTEIN_CUTOFF:
                        conflict = True # we don't use candidate line if it's a subset of another or close
            if not conflict:
                found_valid_line = True
                break # then idx is the one to use
    if not found_valid_line:
        print("WARNING: COULD NOT FIND VALID LINE FOR FOLLOWING TRANSCRIPT:\n", query)
        continue

    text_to_id = transcript[idx]
    query += "\nWhat is the line number above with the text \"" + text_to_id + "\"?"

    print("QUERY:")
    print(query)

    messages = [{"role": "user", "content": query}]
    count_calls += 1
    gpt_response = utils.call_gpt_withlogging(messages, args.model, max_tokens=1000)
    print("CORRECT: ", idx+1) # for passing, we use 1-based indexing
    print("RESPONSE:", gpt_response.strip())

    # if the text to id is present in the response, strip it out
    gpt_response = gpt_response.lower().replace(text_to_id.lower(),"")
    print("Trimmed RESPONSE:", gpt_response.strip())

    num = re.match("\D*(?P<num>\d+).*", gpt_response.strip() )
    response_num = None
    if num is not None:
        response_num = int(num.group("num"))
    print("RESPONSE number extracted=", response_num)
    if response_num != idx+1:
        print("WRONG!")
        count_wrong += 1
        if response_num is None: # i.e., refused to return a number
            histogram_wronglines_relative.update([0]) # this indicates that NO line returned
        else:
            histogram_wronglines_relative.update([response_num-(idx+1)])
        histogram_wronglines_absolute.update([idx+1])

    print("**** STATS:  count_calls=", count_calls, "count_wrong=", count_wrong,
          "accuracy={:.3f}".format(float(count_calls-count_wrong)/count_calls))
    print("histogram_wronglines_relative: -------------------")
    if len(histogram_wronglines_relative) == 0:
        print("nothing in histogram_wronglines_relative")
    else:
        histogram_wronglines_relative_list = list(histogram_wronglines_relative.items())
        histogram_wronglines_relative_list.sort(key=lambda x: x[0]) # sort by relative location, not numbers
        for x in histogram_wronglines_relative_list:
            if x[0] == 0:
                print("No line number returned {:5d}".format(x[1]))
            else:
                print("{:5d} {:5d}".format(x[0], x[1]))
    print("histogram_wronglines_absolute: -------------------")
    if len(histogram_wronglines_absolute) == 0:
        print("nothing in histogram_wronglines_absolute")
    else:
        histogram_wronglines_absolute_list = list(histogram_wronglines_absolute.items())
        histogram_wronglines_absolute_list.sort(key=lambda x: x[0]) # sort by absolute location, not numbers
        for x in histogram_wronglines_absolute_list:
            print("{:5d} {:5d}".format(x[0], x[1]))

print(datetime.datetime.now())
suggested_filename = "linewithstr_lines" + str(args.numlines) + "_runs" + str(args.numruns) + "_" + \
    args.model + ".txt"
print("Suggested filename:", suggested_filename)

