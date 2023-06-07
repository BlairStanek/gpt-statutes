# Takes a particular section of a real statute and calls GPT-* to see if it can
# identify the text.
import xml.etree.ElementTree as ET
from real_stat_utils import subdivision_types, usc_ns_str, ns, usc_ns_str_curly
import tiktoken
import argparse, random, sys, re
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
args = parser.parse_args()

order_random = random.Random(42) # used for shuffling


class Leaf:
    def __init__(self, identifier, text, level):
        self.identifier = identifier
        self.text = text
        self.level = level

    def __str__(self):
        return "(" + self.identifier + ", " + str(self.level) + ", " + self.text.strip() + ")"

class StatLine:
    def __init__(self, identifier, text):
        self.identifier = identifier
        self.text = text
        assert "\n" not in self.text

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
        assert x.attrib["status"] == 'repealed'
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
            if len(statlines[-1].text) > 0 and statlines[-1].text[-1] == ".": # some headers already end in period; don't add a second one
                statlines[-1].text += " "
            else:
                statlines[-1].text += ". " # like with Westlaw, have a heading ended with a period, not a newline
        elif sub.tag == (usc_ns_str_curly + "chapeau"):
            sub_isleaf, _ = parse_xml_statute(sub, statlines, True, 0)
            assert not sub_isleaf
        elif sub.tag == (usc_ns_str_curly + "continuation"): # aka "flush language", since flush to header
            statlines.append(StatLine("flush", ("  " * max(0, level - 1)))) # like with Westlaw have flush language flush
            sub_isleaf, _ = parse_xml_statute(sub, statlines, True, 0)
            assert not sub_isleaf
            if len(leaves) > 0: # This is crucial: we cannot test for a leaf followed immediately by flush language
                leaves.pop() # removes last leaf
        else:
            sub_isleaf, sub_leaves = \
                parse_xml_statute(sub, statlines, assert_no_subdivisions, level + 1) # recursively build up
            if sub_isleaf:
                assert len(sub_leaves) == 0
                leaves.append(Leaf(sub.attrib.get("identifier", ""), statlines[-1].text, level))
            elif len(sub_leaves) > 0:
                assert not sub_isleaf
                leaves.extend(sub_leaves)

    if x.tail is not None:
        statlines[-1].text += x.tail.replace("\n","") + " " # we handle line-keeping via statlines, not newlines

    if level == 0: # The only newlines should be when there are no subdivisions
        for line in statlines:
            assert "\n" not in line.text or len(statlines) == 1

    return x_is_leaf, leaves


########## Start main code #############
irc_tree = ET.parse("xml_uscAll@117-327not263not286/usc26.xml")
irc_root = irc_tree.getroot()
list_leaves = []  # list of tuples of (statutory text:str, Leaf)
for s in irc_root.iter('{' + usc_ns_str + '}section'):  # this loop builds up the list of possibilities
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
            for line in statlines:
                print("{:32s} |".format(line.identifier), line.text)
                if "\n" in line.text:
                    print("+++++++++ Newline above")
            # print(text.strip())
            # for leaf in leaves:
            #     if leaf.level >= args.mindepth:
            #         list_leaves.append((text.strip(), leaf))

exit(1)

print("len(list_leaves)=", len(list_leaves))
order_random.shuffle(list_leaves)


gpt3_tokenizer = tiktoken.encoding_for_model("text-davinci-003")
curr_tokenizer = tiktoken.encoding_for_model(args.model)

count_calls = 0
count_wrong = 0

for leaf_info in list_leaves:
    if count_calls >= args.numcalls:
        break # done

    identifier = leaf_info[1].identifier
    cite_components = identifier.strip("/us/usc/t26/s").split("/")
    cite = cite_components[0]
    for component in cite_components[1:]:
        cite += "(" + component + ")"

    question = "What is the exact text at section " + cite + "?"
    query = leaf_info[0] + "\n\n" + question

    # This test ensures comparability of GPT-4 and GPT-3, despite different-sized token windows
    if args.limitGPT3:
        GPT3_LIMIT = 4000
        TOKEN_BUFFER = 300
        gpt3_encoding = gpt3_tokenizer.encode(query + leaf_info[1].text.strip())
        num_gpt3_tokens = len(gpt3_encoding)
        if num_gpt3_tokens > (GPT3_LIMIT-TOKEN_BUFFER):
            print("Skipped", identifier, "due to exceeding gpt-3 capacity")
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
    print(identifier, "*****************************")
    print(query)
    response = utils.call_gpt_withlogging(messages, args.model, max_tokens=max_tokens_back)
    # if response == None:
    #     continue # this is the result of being too long; don't count towards stats, since never got actual response
    count_calls += 1

    correct_answer = leaf_info[1].text.strip()
    print("correct:", correct_answer)
    print("response:", response.strip())

    # determine whether this is correct
    assert correct_answer.find(")") > 0
    correct_trimmed = correct_answer[correct_answer.find(")")+1:]
    correct_std = (" ".join(re.split(r'\W+', correct_trimmed))).lower()
    response_std = (" ".join(re.split(r'\W+', response))).lower()
    if correct_std not in response_std:
        print("WRONG!")
        print("correct_std:", correct_std)
        print("response_std:", response_std)
        count_wrong+=1

    print("count_calls=", count_calls, " count_wrong=", count_wrong,
          " accuracy={:.3f}".format((count_calls-count_wrong)/float(count_calls)))
