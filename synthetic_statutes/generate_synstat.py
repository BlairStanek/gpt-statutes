# Generates synethetic statutes as well as semantically-identical prose equivalents
import re

class statute_part:
    def __init__(self, term:str):
        self.term = term  # e.g. "vilihick" -- the term being referenced
        self.parent = None # None for root, otherwise reference
        self.children = None # starts out as none; add later using function below
        self.stat_used = None # e.g. 1001(a)(3)(A)
        self.stat_defined = None # e.g. 1001(a)(3)(A)(iii)
        self.sentence_num = None # int.  useful for creating numbered prose sentences.

    def add_child(self, new_child):
        if new_child is None:
            return
        if self.children is None:
            self.children = []
        for c in self.children:
            assert c.term != new_child.term
        self.children.append(new_child)
        new_child.parent = self

    def has_children(self):
        return self.children is not None and len(self.children) > 0

    def has_grandchildren(self):
        return self.has_children() and self.children[0].has_children()

    def get_all_descendants(self) -> list:
        rv = [self]
        if self.has_children():
            for c in self.children:
                rv.extend(c.get_all_descendants())
        return rv

    def print_statute_info_recursive(self):
        used = "--"
        if not self.stat_used is None:
            used = self.stat_used
        defined = "--"
        if not self.stat_defined is None:
            defined = self.stat_defined

        print("{0:<25s} {1:<25s}".format(used, defined), self.term)
        if not self.children is None:
            for child in self.children:
                child.print_statute_info_recursive()

    def print_statute_info(self):
        print("{0:<25s} {1:<25s}".format("stat_used", "stat_defined"))
        self.print_statute_info_recursive()

    def get_level(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_level()

    # gets distance, in tree edges, from x
    def get_dist(self, x):
        self_walker = self
        self_count = 0
        while self_walker is not None:
            x_walker = x
            x_count = 0
            while x_walker is not None:
                if self_walker == x_walker:
                    return self_count + x_count
                x_walker = x_walker.parent
                x_count += 1
            self_walker = self_walker.parent
            self_count += 1
        assert False, "Got two that are not in the same tree"

    # Find the item in the other_statute that is in the exact same position
    # as self.  The assumption is that the source statute for self is
    # different from other_statute; otherwise, it just returns self.
    def get_analogous_item(self, other_statute):
        tree_branches = []
        x = self
        while x.parent is not None:
            tree_branches.append(x.parent.children.index(x))
            x = x.parent
        y = other_statute
        while len(tree_branches) > 0:
            y = y.children[tree_branches.pop()]
        return y

    def get_root(self):
        x = self
        while x.parent is not None:
            x = x.parent
        return x

# This generates the abstract representation of the synthetic statute.
def generate_abstract(stack_names:list, tree_depth:int, branch_factor:int, cur_depth=0):
    if cur_depth > tree_depth:
        return None
    if cur_depth == 0:
        rv = statute_part("") # assign name later when giving None for parent
    else:
        rv = statute_part(stack_names.pop())

    for i in range(branch_factor):
        rv.add_child(generate_abstract(stack_names, tree_depth, branch_factor, cur_depth+1))
    if cur_depth == 0:
        rv.term = stack_names.pop()
        assert rv. parent is None

    return rv

# Extracts a list of all terms that appear as the first item the 2-tuples in the tree
# generated by generate_abstract().
def extract_all_used_terms(abst) -> list:
    rv = [abst.term.lower()] # we want lower case versions
    if abst.has_children():
        for child in abst.children:
            rv.extend(extract_all_used_terms(child))
    return rv

# Extracts a list of all terms that appear as the first item the 2-tuples in the tree
# generated by generate_abstract().
def extract_all_used_parts(abst) -> list:
    rv = [abst] # we want lower case versions
    if abst.has_children():
        for child in abst.children:
            rv.extend(extract_all_used_parts(child))
    return rv


# Reads from a file filled with nonces generated by https://www.soybomb.com/tricks/words/
def read_nonces() -> list:
    with open("nonces.txt", "r") as f:
        nonce_txt = f.read()
    lowercase_nonce_list = nonce_txt.split()
    # verify none are duplicated, which would cause statutory problems
    rv = []
    for nonce in lowercase_nonce_list:
        candidate = nonce[0].upper() + nonce[1:].lower()
        if candidate[-1:] == "s": # remove ending s, which indicates plural and might confuse language models
            candidate = candidate[:-1]
        assert not candidate in rv

        # make sure there is no overlap of part of one term over another
        overlap = False
        for already_in in rv:
            if candidate.lower() in already_in.lower() or \
               already_in.lower() in candidate.lower():
                overlap = True
        if not overlap:
            rv.append(candidate)

    for x in rv:
        for y in rv:
            if x != y:
                assert not x.lower() in y.lower()
                assert not y.lower() in x.lower()

    return rv

# This generates a random set of names like "M11" and "Z66"
def generate_systematic(rand_gen) -> list:
    rv = []
    for idx_letter in range(26):
        for idx_num in range(10):
            rv.append(chr(ord('A')+idx_letter) + str(idx_num) + str(idx_num))
    rand_gen.shuffle(rv)
    return rv

# Used to generate roman numerals, which are used for clause and subclause numbering
def int_to_roman(num):
    if 1 <= num <= 3:
        return "i" * num
    elif num == 4:
        return "iv"
    elif 5 <= num <= 8:
        return "v" + ("i" * (num-5))
    else:
        assert False, "not implemented"

def level_num_label(level:int, num:int):
    rv = "  " * level + "("
    if level == 0: # subsection   (a)
        rv += chr(ord('a') + num)
    elif level == 1: # paragraph  (1)
        rv += str(num+1)
    elif level == 2: # subparagraph (A)
        rv += chr(ord('A') + num)
    elif level == 3: # clause (i)
        rv += int_to_roman(num+1).lower()
    elif level == 4: # subclause (I)
        rv += int_to_roman(num+1).upper()
    elif level == 5: # NOT using
        rv += ""
    else:
        assert False, "not implemented"
    return rv + ")"

# Creates appropriate separator between parts of a statute.
def sep(index, len_list) -> str:
    if index == len_list - 2:
        return ", or\n"
    elif index == len_list - 1:
        return ".\n"
    else:
        return ",\n"

def simple_sep(index, len_list) -> str:
    if index == len_list - 2:
        if len_list == 2:
            return " or"
        else:
            return ", or"
    elif index == len_list - 1:
        return "."
    else:
        return ","

# Takes an abstract representation and creates a statute (recursively).
# Also fills in the citation.
# keep_compact -- used to minimize the number of lines in a normal statute
# collapse_leaves -- makes all leaf terms appear in a single line with their parent
def abstract_to_statute(abst,
                        level = 0,
                        context = None,
                        sec_num = 1001,
                        keep_compact=False,
                        collapse_leaves=False) -> str:
    if collapse_leaves:
        assert keep_compact, "They go together"
    rv = ""
    if level == 0:
        rv  = "Section " + str(sec_num) + ".  Definition of " + abst.term +".\n"
        context = "section " + str(sec_num)
    if not abst.has_grandchildren(): # simple; definition in terms of leaf nodes
        if not collapse_leaves:
            if not keep_compact:
                rv += "  " * (level - 1)
            rv += "The term \"" + abst.term.lower() + "\" means-\n"
            abst.stat_defined = context
            for i, child in enumerate(abst.children):
                used_label = level_num_label(level, i)
                child.stat_used = context.strip() + used_label.strip()
                rv += used_label + " any " + child.term.lower() + sep(i, len(abst.children))
        else: # if here, we are collapsing the children into a single sentence on one line
            rv += "The term \"" + abst.term.lower() + "\" means"
            abst.stat_defined = context
            for i, child in enumerate(abst.children):
                used_label = level_num_label(level, i)
                child.stat_used = context.strip() + used_label.strip()
                rv += " any " + child.term.lower() + simple_sep(i, len(abst.children)) # no newline
            rv += "\n"
    else:
        def_label = level_num_label(level, 0)
        rv +=  def_label + " General rule"
        if keep_compact:
            rv += ". "
        else:
            rv += "\n" + "  " * (level)
        if not collapse_leaves:
            rv += "The term \"" + abst.term.lower() + "\" means-\n"
            abst.stat_defined = context.strip()
            for i, child in enumerate(abst.children):
                used_label = level_num_label(level + 1, i)
                rv += used_label + " any " + child.term.lower() + sep(i, len(abst.children))
                child.stat_used = abst.stat_defined.strip() + def_label.strip() + used_label.strip()
        else:
            rv += "The term \"" + abst.term.lower() + "\" means"
            abst.stat_defined = context.strip()
            for i, child in enumerate(abst.children):
                used_label = level_num_label(level + 1, i)
                rv += " any " + child.term.lower() + simple_sep(i, len(abst.children))
                child.stat_used = abst.stat_defined.strip() + def_label.strip() + used_label.strip()
            rv += "\n"

        for i, child in enumerate(abst.children):
            head_label = level_num_label(level, i+1)
            rv += head_label
            if collapse_leaves:
                if child.has_grandchildren():
                    rv += " " + child.term + "\n"
                else:
                    rv += " "
            else:
                rv += " " + child.term
                if keep_compact and not child.has_grandchildren():
                    rv += ". "
                else:
                    rv += "\n"
            rv += abstract_to_statute(child,
                                      level + 1,
                                      context.strip() + head_label.strip(),
                                      keep_compact=keep_compact,
                                      collapse_leaves=collapse_leaves)
    return rv

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


# This will produce text in a non-statutory format to use as a benchmark
# Also sets the sentence numbers
def abstract_to_sentences(abst, sentence_num_format=None, num_sentence=1):
    assert (num_sentence != 1) or (abst.parent is None)
    prose_text = ""
    if not sentence_num_format is None:
        prose_text += sentence_num_format.format(num_sentence)

    prose_text += "The term \"" + abst.term.lower() + "\" means "
    abst.sentence_num = num_sentence # save for creating probes

    num_sentence += 1
    for i, child in enumerate(abst.children):
        prose_text += "any " + child.term.lower()
        if len(abst.children) > 2 and i < len(abst.children)-1:
            prose_text += ", "
        if i == len(abst.children)-2:
            if not prose_text[-1].isspace():
                prose_text += " "
            prose_text += "or "
    prose_text += ".\n"

    if abst.has_grandchildren():
        for child in abst.children:
            text_back, num_sentence = abstract_to_sentences(child, sentence_num_format, num_sentence)
            prose_text += text_back

    return prose_text, num_sentence

# Gets dict where keys are sentence nums and values are the statute_parts whose terms
# are defined in that sentence
def get_dict_of_sentence_definitions(abst):
    rv = dict()
    if not abst.sentence_num is None:
        rv[abst.sentence_num] = abst
    if abst.has_children():
        for child in abst.children:
            rv.update(get_dict_of_sentence_definitions(child))
    return rv


# This can be used for the output of abstract_to_sentences to number the lines, to
# allow references for precise reasoning.
def add_line_numbers(text:str) -> str:
    rv = ""
    for idx, line in enumerate(text.split("\n")):
        if len(line) > 0 and not line.isspace():
            rv += "(" + str(idx+1) + ") " + line + "\n"
    return rv

# For most words, this will be "a".  But for words starting with
# a vowel or some abbreviations, it will be "an"
def get_article(word):
    if word[0].isalpha() and word[1].isalpha():
        if word[0].lower() in "aeiou":
            return "an"
    elif word[0].isalpha() and word[1].isnumeric():
        if word[0].lower() in "aefhilmnorsx":
            # an A-plus, an E-class, an F-U, an H14, an I9, an L4, an M16, an N99, an O-ring, an R5, an S22, an X-ray
            return "an"
    else:
        assert False, "not implemented"
    return "a" # the default


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

if __name__ == "__main__":
    nonce_list = read_nonces()
    import random
    rand_gen = random.Random(42)
    rand_gen.shuffle(nonce_list)
    abst = generate_abstract(nonce_list, 3, 3)
    print(abstract_to_statute(abst))
    abst.print_statute_info()
    print("\n" + abstract_to_sentences(abst, "Sentence {:d}: ")[0])

    for num, part in get_dict_of_sentence_definitions(abst).items():
        print(num, part.term)

    used_parts = extract_all_used_parts(abst)
    print([l.term for l in used_parts])

    # print("\n", extract_all_used_terms(abst))
    # auncles = get_auncles(abst)
    # for stat, auncles in auncles:
    #     print("{:<15s}  auncles:".format(stat.term), end=" ")
    #     for auncle in auncles:
    #         print("{:<15s} {:<20s} sent{:3d}".format(auncle.term, auncle.stat_defined, auncle.sentence_num), end=" ")
    #     print("")
