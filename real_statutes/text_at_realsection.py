# Takes a particular section of a real statute and calls GPT-* to see if it can
# identify the text.
import xml.etree.ElementTree as ET
from real_stat_utils import subdivision_types, usc_ns_str, ns, usc_ns_str_curly
from collections import Counter
import tiktoken
import argparse, random, sys, re, os, numpy, datetime
sys.path.append('../')
import utils

parser = argparse.ArgumentParser(description='Test the ability of GPT-* to find the text for a section')
parser.add_argument('--mindepth', required=True, type=int,
                    help='minimum depth of a leaf to consider')
parser.add_argument('--numcalls', required=True,type=int,
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

FLUSH_LANGUAGE = "flush"

# All of the below is similar to what is in the synthetic_statutes/generate_synstat.py
WRONG_SUBSECTION = "wrong subsection"
WRONG_PARAGRAPH = "wrong paragraph"
WRONG_SUBPARAGRAPH = "wrong subparagraph"
WRONG_CLAUSE = "wrong clause"
WRONG_SUBCLAUSE = "wrong subclause"
WRONG_ITEM = "wrong item"
WRONG_SUBITEM = "wrong subitem"
WRONG_SUBSUBITEM = "wrong subsubitem"

NOT_PARALLEL = "not parallel"
NOT_FOUND = "not found"
EXCERPTED_SUBSET = "only subset"
def analyze_error(correct_id:str, incorrect_id:str):
    if incorrect_id == FLUSH_LANGUAGE:
        return NOT_PARALLEL # not clear how to compare flush language to a real cite; several ways to do it, each with problems.

    match_re = "/us/usc/t\d\d?/s" + \
               "(?P<sec>\d[^/]*)" + \
               "(?P<subsec>/\w+)?" + \
               "(?P<para>/\w+)?" + \
               "(?P<subpara>/\w+)?" + \
               "(?P<clause>/\w+)?" + \
               "(?P<subclause>/\w+)?" \
               "(?P<item>/\w+)?" + \
               "(?P<subitem>/\w+)?" + \
               "(?P<subsubitem>/\w+)?$"
    cor = re.match(match_re, correct_id)
    incorr = re.match(match_re, incorrect_id)
    assert cor.group("sec") == incorr.group("sec"), "Should be from same subsection"
    # Check to see if there's a simple failure of parallelism
    for group in ["subsec", "para", "subpara", "clause", "subclause", "item", "subitem", "subsubitem"]:
        if (cor.group(group) is None) != (incorr.group(group) is None):
            return NOT_PARALLEL # return directly rather than as list
    # Now pinpoint the error
    rv = []
    if cor.group("subsec") != incorr.group("subsec"):
        rv.append(WRONG_SUBSECTION)
    if cor.group("para") != incorr.group("para"):
        rv.append(WRONG_PARAGRAPH)
    if cor.group("subpara") != incorr.group("subpara"):
        rv.append(WRONG_SUBPARAGRAPH)
    if cor.group("clause") != incorr.group("clause"):
        rv.append(WRONG_CLAUSE)
    if cor.group("subclause") != incorr.group("subclause"):
        rv.append(WRONG_SUBCLAUSE)
    if cor.group("item") != incorr.group("item"):
        rv.append(WRONG_ITEM)
    if cor.group("subitem") != incorr.group("subitem"):
        rv.append(WRONG_SUBITEM)
    if cor.group("subsubitem") != incorr.group("subsubitem"):
        rv.append(WRONG_SUBSUBITEM)

    return rv

class StatLine:
    def __init__(self, identifier, text):
        self.identifier = identifier
        self.text = text
        assert "\n" not in self.text

    def __str__(self):
        return self.identifier + "," + self.text

    def get_line_text_withnum(self) -> str:
        return self.text

    # returns just the text, with the num stripped off
    def get_line_text(self) -> str:
        rv = self.text.strip()
        if self.identifier == FLUSH_LANGUAGE:
            return rv
        elif rv.startswith("§"):
            assert re.search("/s\d[^/]*$", self.identifier) is not None
            start_id = rv.find(".") # find the period after the section number
            assert start_id >= 0
            return rv[start_id+1:].strip()
        else:
            rv = self.text.strip()
            if rv.startswith("["): # unusual circumstance, e.g. 26 USC 56(c); must handle
                rv = rv[1:]
                assert "repealed" in rv.lower() or \
                       "transferred" in rv.lower() or \
                       "redesignated" in rv.lower() or \
                       self.identifier == '/us/usc/t42/s1396r–8/e/4'
                rv = rv.replace("]", "")
            assert rv.startswith("(" + self.identifier.split("/")[-1] + ")")
            end_id = rv.find(")")
            assert end_id > 0
            return rv[end_id+1:].strip()

class Leaf:
    def __init__(self, statlines, linenum, level):
        assert type(statlines) == list
        for sl in statlines:
            assert type(sl) == StatLine
        self.statlines = statlines
        assert linenum < len(statlines)
        self.linenum = linenum
        self.level = level
        self.percentile = None # This will be used to analyze how far the leaf is into the statute

    def get_line_text_withnum(self):
        return self.statlines[self.linenum].get_line_text_withnum()

    def get_line_text(self):
        return self.statlines[self.linenum].get_line_text()

    def get_identifier(self):
        return self.statlines[self.linenum].identifier

    def __str__(self):
        return "(" + str(self.level) + ", " + self.statlines[self.linenum] + ")"

    # returns the full text of the *entire* statute in which the leaf is contained
    def get_full_stat_text(self) -> str:
        rv = ""
        for l in self.statlines:
            rv += l.text.rstrip() + "\n"
        return rv

    def get_cite(self) -> str:
        identifier = self.statlines[self.linenum].identifier
        non_title_info = identifier.strip("/us/usc/t").split("/s")[1]
        cite_components = non_title_info.split("/")
        rv = cite_components[0]
        for component in cite_components[1:]:
            rv += "(" + component + ")"
        return rv

def is_statute_in_response(statute_raw:str, response_raw:str, print_wrong=False) -> bool:
    statute_tokens = re.split(r'\W+', statute_raw)
    statute_std = " ".join(statute_tokens).lower().strip()
    response_std = " ".join(re.split(r'\W+', response_raw)).lower().strip()
    if statute_std in response_std: # this is the most basic way to be in it
        return True
    # sometimes GPT quotes back without trailing conjunction
    if statute_tokens[-1] in ["or", "and", "but", "yet", "however", "nor"]:
        if " ".join(statute_tokens[:-1]).lower().strip() in response_std:
            return True
    # sometimes GPT quotes back without the title of the subdivison
    # for example for "(1) General rule. In the case of an estate or trust (other than a trust me..."
    # it quoted back only "In the case of an estate or trust (other than a trust me..."
    header_match = re.search("[.]\s+[A-Z]", statute_raw)
    if header_match is not None:
        candidate_subset = statute_raw[header_match.start(0)+1:]
        # We only treat the header as valid if at least this number of words.
        # For example, this once incorrectly matched up 'In general. If—'
        MIN_WORDS_TO_BE_VALID = 5
        if len(candidate_subset.split()) >= MIN_WORDS_TO_BE_VALID:
            if is_statute_in_response(candidate_subset, response_raw):
                return True

    if print_wrong:
        print("statute_std=", statute_std)
        print("response_std=", response_std)
    return False

# This builds up the text in the WESTLAW format (i.e. more compact, and using idents
# that make the most sense).
# The text is built up within the list of statlines
# Then it returns true if x is a leaf (a bool).
# Then it returns a list of all Leaf (a list) under x, IF x was not itself a leaf.
def parse_xml_statute(x:ET.Element, statlines:list, assert_no_subdivisions = False, level = 0):
    x_is_leaf = False
    leaves = [] # this will be used to return a set of leaves UNDER x that can be used

    # We ignore repealed, historical, notes, etc. and focus on just the statute as it is
    if "status" in x.attrib:
        assert x.attrib["status"] in ['repealed', 'transferred']
        return x_is_leaf, leaves
    if "sourceCredit" in x.tag or "notes" in x.tag:
        return x_is_leaf, leaves

    if x.tag in subdivision_types:
        assert not assert_no_subdivisions, "an assumption about no subdivisions under chapeaus, continuations, etc. failed"
        statlines.append(StatLine(x.attrib.get("identifier", ""), "  " * max(0, level-1)))
        x_is_leaf = True # presumed, but can be negated immediately below if contains subdivision
        for subdiv_type in subdivision_types:
            if x.tag != subdiv_type and len(list(x.iter(subdiv_type))) > 0:
                x_is_leaf = False # contains a subdivision, so cannot be leaf

    if len(statlines) == 0:
        statlines.append(StatLine(x.attrib.get("identifier", ""), ""))

    if x.text is not None: # this is the main mechanism for building up the text
        statlines[-1].text += x.text

    for sub in x:
        if sub.tag == (usc_ns_str_curly + "heading"):
            sub_isleaf, _ = parse_xml_statute(sub, statlines, True, 0)
            assert not sub_isleaf
            if len(statlines[-1].text) > 0 and \
                    statlines[-1].text.strip()[-1] in ".-——–:": # some headers already end in punctuation; don't add more
                if not statlines[-1].text[-1].isspace():
                    statlines[-1].text += " " # ensure at least one space
            else:
                statlines[-1].text += ". " # like with Westlaw, have a heading ended with a period, not a newline
        elif sub.tag == (usc_ns_str_curly + "chapeau"):
            sub_isleaf, _ = parse_xml_statute(sub, statlines, True, 0)
            assert not sub_isleaf
        elif sub.tag == (usc_ns_str_curly + "continuation"): # aka "flush language", since flush to header
            statlines.append(StatLine(FLUSH_LANGUAGE, ("  " * max(0, level - 1)))) # like with Westlaw have flush language flush
            sub_isleaf, _ = parse_xml_statute(sub, statlines, True, 0)
            assert not sub_isleaf
            if len(leaves) > 0: # This is crucial: we cannot test for a leaf followed immediately by flush language
                leaves.pop() # removes last leaf
        else:
            sub_isleaf, sub_leaves = \
                parse_xml_statute(sub, statlines, assert_no_subdivisions, level + 1) # recursively build up
            if sub_isleaf:
                assert len(sub_leaves) == 0
                leaves.append(Leaf(statlines, len(statlines)-1, level))
            elif len(sub_leaves) > 0:
                assert not sub_isleaf
                leaves.extend(sub_leaves)

    if x.tail is not None:
        statlines[-1].text += x.tail.replace("\n","") + " " # we handle line-keeping via statlines, not newlines

    if level == 0 and not assert_no_subdivisions: # here we are at the top level
        for idx_line, line in enumerate(statlines):
            assert "\n" not in line.text or len(statlines) == 1 or \
                   "repealed" in line.text.lower() or \
                   "omitted" in line.text.lower() or\
                   "reserved" in line.text.lower(), \
                "The only newlines should be when there are no subdivisions or item repealed, etc."
            if "\n" in line.text:
                line.text = line.text.replace("\n"," ").replace("  ", " ")
            if line.text.strip().startswith("“"): # e.g. 5 USCA § 9507(b)(1).  Don't even load these sections.
                statlines.clear() # remove everything
                return False, [] # do nothing

        # Here we calculate the leaf percentiles
        if len(leaves) > 1: # ignore leaves from sections with just one leaf
            for idx_leaf, leaf in enumerate(leaves):
                leaf.percentile = float(idx_leaf)/(len(leaves)-1)

    if statlines[-1].text.strip().startswith("["): # unusual circumstance, e.g., 26 USC 56(c)
        assert (x.tag not in subdivision_types) or \
               statlines[-1].text.strip().strip(".").strip().endswith("]") or \
               statlines[-1].identifier == '/us/usc/t42/s1396r–8/e/4' # handles odd case
        assert (x.tag not in subdivision_types) or \
               "repealed" in statlines[-1].text.lower() or \
                "transferred" in statlines[-1].text.lower() or \
                "redesignated" in statlines[-1].text.lower() or \
                statlines[-1].identifier == '/us/usc/t42/s1396r–8/e/4'  # handles odd case
        x_is_leaf = False # repealed/transferred/redesignated portions disallowed from being tested as leaves or containing leaves
        leaves = []

    if "\n" in statlines[-1].text:
        assert (x.tag not in subdivision_types) or \
               "repealed" in statlines[-1].text.lower() or \
               "omitted" in statlines[-1].text.lower() or \
               "reserved" in statlines[-1].text.lower() or \
               statlines[-1].identifier == '/us/usc/t42/s1396r–8/e/4'  # handles odd case
        x_is_leaf = False # repealed, etc. portions disallowed from being tested as leaves or containing leaves
        leaves = []

    return x_is_leaf, leaves

def load_leaves() -> list:
    list_leaves = []  # list of tuples of (statutory text:str, Leaf)
    print("Loading Titles ", end="")
    for title in range(1, 55):
    # for title in range(26, 27):  # DEBUG
        print( title, end=" ", flush=True)
        prefix = ""
        if title < 10:
            prefix = "0"
        USC_XMLFILE_ROOT = "xml_uscAll@117-327not263not286"
        filename = USC_XMLFILE_ROOT + "/usc" + prefix + str(title) + ".xml"
        if not os.path.exists(filename):
            continue
        title_tree = ET.parse(filename)
        title_root = title_tree.getroot()
        for s in title_root.iter('{' + usc_ns_str + '}section'):  # this loop builds up the list of possibilities
            num = s.find('{' + usc_ns_str + '}num')
            # if s.attrib.get("identifier", "") == "/us/usc/t26/s482": # debug
            #     print(ET.tostring(s, encoding='utf8').decode('utf8'))

            if num is not None and \
                    num.text is not None and \
                    len(s.attrib.get("status", "")) == 0 and \
                    s.attrib.get("identifier", "").startswith("/us/usc/t"):
                tables = s.iter("{http://www.w3.org/1999/xhtml}table") # HTML-style tables within US Code XML
                layouttables = s.iter(usc_ns_str+"layout") # XML-style tables wtihin US Code XML, called <layout>
                if len(list(tables)) == 0 and len(list(layouttables)) == 0: # skip all sections with tables
                    statlines = []
                    _, leaves = parse_xml_statute(s, statlines)
                    # for line in statlines:
                    #     print("{:32s} |".format(line.identifier), line.text)
                    #     if "\n" in line.text:
                    #         print("+++++++++ Newline above")
                    if len(leaves) > 1: # ignore leaves from sections with just one valid leaf
                        for leaf in leaves:
                            if leaf.level >= args.mindepth:
                                list_leaves.append(leaf)

    print("len(list_leaves)=", len(list_leaves))
    order_random.shuffle(list_leaves)
    return list_leaves

def run_tests():
    list_leaves = load_leaves()
    list_wrong_statutes = []
    list_charlen_correct = []
    list_charlen_wrong = []
    list_newlinelen_correct = []
    list_newlinelen_wrong = []

    gpt3_tokenizer = tiktoken.encoding_for_model("text-davinci-003")
    curr_tokenizer = tiktoken.encoding_for_model(args.model)

    count_calls = 0
    count_wrong = 0
    list_percentiles_wrong = [] # percentile of leaf in [0,1.0] for all leaves incorrectly identified
    histogram_errors = Counter()

    for leaf in list_leaves:
        if count_calls >= args.numcalls:
            break # done

        # def get_full_stat_text(self) -> str:
        # def get_cite(self) -> str:

        statute_text = leaf.get_full_stat_text()
        cite_text = leaf.get_cite()
        if args.replacenumtitle:
            idx_first_newline = statute_text.find("\n")
            assert idx_first_newline > 0
            idx_first_paren = cite_text.find("(")
            assert idx_first_paren > 0
            statute_text = "Section 1001.  Key provisions." + statute_text[idx_first_newline:]
            cite_text = "1001" + cite_text[idx_first_paren:]

        question = "What is the exact text at section " + cite_text + "?"
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
        if is_statute_in_response(correct_answer, response, True): # Correct
            list_charlen_correct.append(len(statute_text))
            list_newlinelen_correct.append(len(statute_text.split("\n")))
        else:
            # Do error analysis
            print("WRONG!")
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

            # print detailed error information only if there was just an error found
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
    # correct = "the number of Federal public bridges within each such State; bears to"
    # response = "The exact text at section 204(b)(1)(A)(iv)(I) is:\n\n\"(I) the number of Federal public bridges within each such State;\""
    # rv = is_statute_in_response(response, correct, True)
    # print(rv)
