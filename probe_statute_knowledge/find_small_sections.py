# Created 9 Feb 2023
# Aims to find tiny sections of the U.S. Code

import xml.etree.ElementTree as ET
import pickle, re, sys, statistics
import os.path
sys.path.append('../')
import utils
import random
random.seed(42) # ensure reproducability

usc_ns_str = "http://xml.house.gov/schemas/uslm/1.0"
ns = {"usc" : usc_ns_str}

# Gets all non-header text from NON-repealed sections
def get_IRC_text_recursive(x:ET.Element, top_level = True) -> str:
    rv = ""
    if x.text is not None:
        rv += x.text + " "
    for sub in x:
        if "status" not in sub.attrib:
            if "sourceCredit" not in sub.tag and \
                    "notes" not in sub.tag and \
                    (not top_level or ("num" not in sub.tag and "heading" not in sub.tag)):
                rv += get_IRC_text_recursive(sub, False)
        else:
            # Count of statuses in all of IRC was the following: {'': 519201, 'repealed': 32}
            # Thus we are making the assumption asserted below
            assert sub.attrib["status"] in ["repealed" , 'transferred']
    if x.tail is not None:
        rv += x.tail + " "
    return rv

USC_DIRECTORY = "xml_uscAll@117-327not263not286"

def compact_statute(orig_text:str) -> str:
    rv = ""
    for line in orig_text.split("\n"):
        if not line.isspace():
            rv += line.rstrip() + "\n"
    rv = rv.replace("\n\n", "\n")
    return rv

def get_sections_from_title(title_num:int, min_len = 1, max_len = 100) -> list:
    prefix = ""
    if title_num < 10:
        prefix = "0"
    filename = USC_DIRECTORY + "/usc" + prefix + str(title_num) + ".xml"
    if not os.path.exists(filename):
        return None

    identifier_prefix = "/us/usc/t" + str(title_num) + "/s"

    # Load the Code itself (downloaded from https://uscode.house.gov/download/download.shtml earlier)
    title_tree = ET.parse(filename)
    title_root = title_tree.getroot()

    sections = [] # list of ALL tuples of (section number, text, word length) that qualify

    for s in title_root.iter('{' + usc_ns_str + '}section'):
        if "identifier" in s.attrib and s.attrib.get("status","") != "repealed":
            num = s.attrib["identifier"]
            assert num.startswith(identifier_prefix)
            num_minus_prefix = num[len(identifier_prefix):]
            # must exclude sections with dashes or letters; will be asking GPT3 for arabic numeral sections
            if num_minus_prefix.isnumeric():
                sect_text = get_IRC_text_recursive(s)
                sect_text_compact = compact_statute(sect_text)
                words_in_section = len(sect_text_compact.split())
                if min_len <= words_in_section <= max_len:
                    sections.append((int(num_minus_prefix),sect_text_compact, words_in_section))
    return sections

if __name__ == "__main__":

    smallest_sections = [0] * 1000

    for title in range(1, 55):
        # if title > 15:
        #     break
        print("Title", title, end="\t")
        sections = get_sections_from_title(title)
        if sections is None:
            print("None")
        else:
            for s in sections:
                if s[2] <= 20:
                    print(title, " USC ", s[0], ": ", s[1])
                smallest_sections[s[2]] += 1

    # for i in range(0, 100):
    #     print(i, "\t", smallest_sections[i])
