# Takes a particular section of a real statute and calls GPT-* to see if it can
# identify the text.

import xml.etree.ElementTree as ET
from real_stat_utils import subdivision_types, usc_ns_str, ns, usc_ns_str_curly

irc_tree = ET.parse("xml_uscAll@117-327not263not286/usc26.xml")
irc_root = irc_tree.getroot()

# These are derived from usctitle.css downloaded from https://uscode.house.gov/download/resources/schemaandcss.zip
# def get_indent(class_str) -> int:
#     left_margin = 0
#     indent = 0
#
#     for style in class_str.split():
#         if style == "firstIndent-4":
#             indent = -4
#         elif style == "firstIndent-3":
#             indent = -3
#         elif style == "firstIndent-2":
#             indent = -2
#         elif style == "firstIndent-1":
#             indent = -1
#         elif style == "firstIndent0":
#             indent = 0
#         elif style == "firstIndent2":
#             indent = 2
#         elif style == "indent0":
#             indent += 1
#             pass
#         elif style == "indent1":
#             left_margin += 1
#             indent += 1
#         elif style == "indent2":
#             left_margin += 2
#             indent += 1
#         elif style == "indent3":
#             left_margin += 3
#             indent += 1
#         elif style == "indent4":
#             left_margin += 4
#             indent += 1
#         elif style == "indent5":
#             left_margin += 5
#             indent += 1
#         elif style == "indent6":
#             left_margin += 6
#             indent += 1
#         elif style == "indent7":
#             left_margin += 7
#             indent += 1
#
#     return max(0, left_margin + indent)
#
# # This builds up the text (in the same way as the House website).
# def print_text_exactly(x:ET.Element, top_level = True) -> str:
#     # We ignore historical, notes, etc. and focus on just the statute
#     if "status" in x.attrib:
#         assert x.attrib["status"] == 'repealed'
#         return ""
#     if "sourceCredit" in x.tag or "notes" in x.tag:
#         return ""
#
#     rv = ""
#     if "class" in x.attrib and "indent" in x.attrib["class"].lower():
#         # an indent* attribute indicates that it is a newline
#         rv += "\n" + ("  " * get_indent(x.attrib["class"]))
#
#     if x.text is not None:
#         rv += x.text
#
#     for sub in x:
#         rv += print_text_exactly(sub, False) # recursively build up
#
#     if x.tail is not None:
#         rv += x.tail + " "
#     return rv.rstrip()

class Leaf:
    def __init__(self, identifier, text, level):
        self.identifier = identifier
        self.text = text
        self.level = level

    def __str__(self):
        return "(" + self.identifier + ", " + str(self.level) + ", " + self.text.strip() + ")"


# This builds up the text in the WESTLAW format (i.e. more compact, and using idents
# that make the most sense).  This text is the first return value (a str).
# Then it returns true if x is a leaf (a bool).
# Then it returns a list of all Leaf (a list) under x, IF x was not itself a leaf.
def print_text_exactly(x:ET.Element, assert_no_subdivisions = False, level = 0):
    text = ""
    x_is_leaf = False
    leaves = [] # this will be used to return a set of leaves UNDER x that can be used

    # We ignore repealed, historical, notes, etc. and focus on just the statute as it is
    if "status" in x.attrib:
        assert x.attrib["status"] == 'repealed'
        return text, x_is_leaf, leaves
    if "sourceCredit" in x.tag or "notes" in x.tag:
        return text, x_is_leaf, leaves

    if x.tag in subdivision_types:
        assert not assert_no_subdivisions, "an assumption about no subdivisions under chapeaus, continuations, etc. failed"
        text += "\n" + ("  " * max(0, level-1))
        x_is_leaf = True # presumed, but can be negated immediately below if contains subdivision
        for subdiv_type in subdivision_types:
            if x.tag != subdiv_type and len(list(x.iter(subdiv_type))) > 0:
                x_is_leaf = False # contains a subdivision, so cannot be leaf

    if x.text is not None: # this is the main mechanism for building up the text
        text += x.text

    for sub in x:
        if sub.tag == (usc_ns_str_curly + "heading"):
            sub_text, sub_isleaf, _ = print_text_exactly(sub, True, 0)
            assert not sub_isleaf
            text += sub_text
            if len(text) > 0 and text[-1] == ".": # some headers already end in period; don't add a second one
                text += " "
            else:
                text += ". " # like with Westlaw, have a heading ended with a period, not a newline
        elif sub.tag == (usc_ns_str_curly + "chapeau"):
            sub_text, sub_isleaf, _ = print_text_exactly(sub, True, 0)
            assert not sub_isleaf
            text += sub_text
        elif sub.tag == (usc_ns_str_curly + "continuation"): # aka "flush language", since flush to header
            text += "\n" + ("  " * max(0, level - 1)) # like with Westlaw have flush language flush
            sub_text, sub_isleaf, _ = print_text_exactly(sub, True, 0)
            assert not sub_isleaf
            text += sub_text
            if len(leaves) > 0: # This is crucial: we cannot test for a leaf followed immediately by flush language
                leaves.pop() # removes last leaf
        else:
            sub_text, sub_isleaf, sub_leaves = \
                print_text_exactly(sub, assert_no_subdivisions, level + 1) # recursively build up
            text += sub_text
            if sub_isleaf:
                assert len(sub_leaves) == 0
                leaves.append(Leaf(sub.attrib.get("identifier", ""), sub_text, level))
            elif len(sub_leaves) > 0:
                assert not sub_isleaf
                leaves.extend(sub_leaves)

    if x.tail is not None:
        text += x.tail + " "

    if len(text) > 0 and text[-1] == " ": # the only right whitespace we want to preserve is plain spaces (i.e. not newline)
        text = text.rstrip() + " "
    else:
        text = text.rstrip()

    return text, x_is_leaf, leaves


########## Start actual code #############
for s in irc_root.iter('{' + usc_ns_str + '}section'):
    num = s.find('{' + usc_ns_str + '}num')
    # print(ET.tostring(s, encoding='utf8').decode('utf8'))

    if not s.attrib.get("identifier", "").endswith("s66"):
        continue # DEBUG

    if num is not None and \
            num.text is not None and \
            len(s.attrib.get("status", "")) == 0 and \
            s.attrib.get("identifier", "").startswith("/us/usc/t"):
        tables = s.iter("{http://www.w3.org/1999/xhtml}table")
        if len(list(tables)) == 0: # skip all sections with tables
            text, _, leaves = print_text_exactly(s)
            print(text.strip())
            for leaf in leaves:
                print(leaf)
            exit(0) # for debugging

# def recursive_tag_print(s):
#     print(s.tag)
#     for sub in s:
#         recursive_tag_print(sub)
# recursive_tag_print(irc_sec)
#
# tables = s.iter("{http://www.w3.org/1999/xhtml}table")
# print("table:", len(list(tables)))





















































