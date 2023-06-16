import re
from generate_synstat import statute_part

# This is related to the function of the same name in real_statues/statute_stats.py, but that one
# is applied to real statutes (encoded in XML), whereas this applies to the synthetic statutes.
def get_stats_recursive(x:statute_part, cur_depth = 0):
    max_depth = cur_depth
    max_width = 0
    count_leaves = 0
    count_nonleaves = 0
    total_depth_leaves = 0 # will be used to calculate average depth of a leaf
    total_branching_nonleaves = 0 # will be used to calculate average branching for nonleaves
    branching = 0
    if x.has_grandchildren():
        # then we need to count the non-leaf and leaves in the "General rule"
        count_nonleaves += 1 # "General rule."
        total_branching_nonleaves += len(x.children)
        count_leaves += len(x.children) # one for each of the children there
        total_depth_leaves += (len(x.children) * (cur_depth+2))
        branching += 1 # there is a branching for the "General rule"

    if x.has_children():
        for child in x.children:
            branching += 1
            temp_depth, temp_max_width, temp_count_leaves, \
                temp_count_nonleaves, temp_total_depth_leaves, temp_total_branching_nonleaves = \
                   get_stats_recursive(child, cur_depth+1)
            if temp_depth > max_depth:
                max_depth = temp_depth
            if temp_max_width > max_width:
                max_width = temp_max_width
            count_leaves += temp_count_leaves
            count_nonleaves += temp_count_nonleaves
            total_depth_leaves += temp_total_depth_leaves
            total_branching_nonleaves += temp_total_branching_nonleaves

    if branching > max_width:
        max_width = branching # then this is the maximum branching

    if branching == 0: # ie, x is a leaf
        count_leaves = 1
        total_depth_leaves = cur_depth
    else: # ie, x is NOT a leaf
        count_nonleaves += 1
        total_branching_nonleaves += branching # add in the branching we had here

    return max_depth, max_width, count_leaves, \
           count_nonleaves, total_depth_leaves, total_branching_nonleaves

def print_stats(abst):
    max_depth, max_width, count_leaves, \
    count_nonleaves, total_depth_leaves, total_branching_nonleaves = \
        get_stats_recursive(abst)
    print("max_depth=", max_depth, "\t\t",
          "max_width=", max_width, "\t\t",
          "count_leaves=", count_leaves, "\t\t",
          "count_nonleaves=", count_nonleaves, "\t\t",
          "average_leaf_depth={:.2f}".format(float(total_depth_leaves) / count_leaves), "\t\t",
          "total_branching_nonleaves=", total_branching_nonleaves, "\t\t",
           "average_branching={:.2f}".format(total_branching_nonleaves / float(count_nonleaves)))

WRONG_SUBSECTION = "wrong subsection"
WRONG_PARAGRAPH = "wrong paragraph"
WRONG_SUBPARAGRAPH = "wrong subparagraph"
WRONG_CLAUSE = "wrong clause"
WRONG_SUBCLAUSE = "wrong subclause"
NOT_PARALLEL = "not parallel"
NOT_FOUND = "not found"
def analyze_error(correct_cite:str, incorrect_cite:str):
    match_re = "section \d+" + \
               "(?P<subsec>\(\w+\))" + \
               "(?P<para>\(\w+\))?" + \
               "(?P<subpara>\(\w+\))?" + \
               "(?P<clause>\(\w+\))?" + \
               "(?P<subclause>\(\w+\))?$"
    cor = re.match(match_re, correct_cite)
    incorr = re.match(match_re, incorrect_cite)
    # Check to see if there's a simple failure of parallelism
    for group in ["subsec", "para", "subpara", "clause", "subclause"]:
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
    return rv

def num_to_letter(x:int) -> str:
    letter = chr(ord('A') + (x % 26))
    return letter * (1+(x // 26))

# This is used as a baseline for GPT*'s ability to retrieve by line number.
def reformat_with_lines(statute_text:str) -> str:
    in_lines = statute_text.split("\n")
    rv = ""
    for idx, in_line in enumerate(in_lines):
        if len(in_line.strip()) > 0:
            unnum_text = re.match("\s*\(\w+\)\s*(?P<text>\w.*)$", in_line)
            if unnum_text is not None:
                new_text = unnum_text.groups("text")[0].strip()
            else:
                new_text = in_line.strip()
            rv += num_to_letter(idx) + ": " + new_text + "\n"
            # rv += "Line " + str(idx+1) + ": " + new_text + "\n"
    return rv



