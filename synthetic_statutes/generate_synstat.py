# Generates synethetic statutes as well as semantically-identical prose equivalents

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
        capitalize_first = nonce[0].upper() + nonce[1:].lower()
        assert not capitalize_first in rv
        rv.append(capitalize_first)
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

# Takes an abstract representation and creates a statute (recursively)
# Also fills in the citation
def abstract_to_statute(abst, level = 0, context = None, sec_num = 1001) -> str:
    rv = ""
    if level == 0:
        rv  = "Section " + str(sec_num) + ".  Definition of " + abst.term +".\n"
        context = "section " + str(sec_num)
    if not abst.has_grandchildren(): # simple; definition in terms of leaf nodes
        rv += "  " * (level-1) + "The term \"" + abst.term.lower() + "\" means-\n"
        abst.stat_defined = context
        for i, child in enumerate(abst.children):
            used_label = level_num_label(level, i)
            child.stat_used = context.strip() + used_label.strip()
            rv += used_label + " any " + child.term.lower() + sep(i, len(abst.children))
    else:
        def_label = level_num_label(level, 0)
        rv +=  def_label + " General rule\n"
        rv += "  " * (level) + "The term \"" + abst.term.lower() + "\" means-\n"
        abst.stat_defined = context.strip()
        for i, child in enumerate(abst.children):
            used_label = level_num_label(level + 1, i)
            rv += used_label + " any " + child.term.lower() + sep(i, len(abst.children))
            child.stat_used = abst.stat_defined.strip() + def_label.strip() + used_label.strip()

        for i, child in enumerate(abst.children):
            head_label = level_num_label(level, i+1)
            rv += head_label + " " + child.term + "\n"
            rv += abstract_to_statute(child, level + 1, context.strip() + head_label.strip())

    return rv

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
