# Created 6 Feb 2023
# Probes whether GPT3 can *identify* arbitrary sections of the U.S. Code based on their text.

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

def compact_statute(orig_text:str) -> str:
    rv = ""
    for line in orig_text.split("\n"):
        if not line.isspace():
            rv += line.rstrip() + "\n"
    rv = rv.replace("\n\n", "\n")
    return rv

USC_DIRECTORY = "xml_uscAll@117-327not263not286"

def get_sections_from_title(title_num:int, min_len = 100, max_len = 1000) -> list:
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

# Returns the identified title and identified section (i.e. a 2-tuple of ints)
# But returns None if cannot find
def GPT3_id(section, title_num):
    section_num = section[0]
    section_text = section[1]
    print("Start GPT3_id() for", title_num, "USC sec", section_num, "****************************")

    utils.add_comment("Seeing if GPT3 can id the text of " + str(title_num) + " USC sec " + str(section_num) + \
                      " in " + __file__)

    INITIAL_PROMPT = "Where is the text above from?"
    USC_PROMPT = "\nSo is it from the U.S. Code? The answer (Yes or No) is"
    TITLE_PROMPT = "\nWhat title of the U.S. Code is it from? The answer (arabic numerals) is"
    SECTION_PROMPT = "\nWhat section of title " + str(title_num) + \
                     " of the U.S. Code is it from? The answer (arabic numerals) is"

    initial_prompt = section_text + INITIAL_PROMPT
    initial_response = utils.call_gpt3_withlogging(initial_prompt, "text-davinci-003", max_tokens=300)
    usc_prompt = initial_prompt + initial_response + USC_PROMPT
    usc_response = utils.call_gpt3_withlogging(usc_prompt, "text-davinci-003", max_tokens=300)
    if utils.is_no(usc_response):
        print("FAILURE")
        print(usc_prompt + usc_response)
        return None, None
    title_prompt = usc_prompt + usc_response + TITLE_PROMPT
    title_response = utils.call_gpt3_withlogging(title_prompt, "text-davinci-003", max_tokens=300)
    title_num_text = title_response.strip().strip(".").lower().strip("title").strip()
    if not title_num_text.isnumeric():
        if len(re.findall(r"\d+", title_num_text)):
            extracted_title_num = int(re.findall(r"\d+", title_num_text)[0])
            print("WARNING NONNUMERIC TITLE:", title_num_text, "but extracted", extracted_title_num)
        else:
            print("FAILURE TO GET NUMERIC TITLE")
            print(title_prompt + title_response)
            return None, None
    else:
        extracted_title_num = int(title_num_text)

    section_prompt = title_prompt + title_response + SECTION_PROMPT
    section_response = utils.call_gpt3_withlogging(section_prompt, "text-davinci-003", max_tokens=300)
    section_num_text = section_response.strip().strip(".").lower().strip("section").strip()
    if not section_num_text.isnumeric():
        extracted_section_num = None
        print("WARNING NONNUMERIC SECTION:", section_response)
        for cand in section_num_text.split(): # if both title and section, we will extract the latest (i.e. section)
            cand_clean = cand.strip("§").strip(",").strip(".")
            if cand_clean.find("(") > 0:
                cand_clean = cand_clean[:cand_clean.find("(")]
            if cand_clean.isnumeric():
                extracted_section_num = int(cand_clean)

        if extracted_section_num is None:
            print("FAILURE TO GET NUMERIC SECTION (but did get title, so returning it)")
            print(section_prompt + section_response)
            return extracted_title_num, None
    else:
        extracted_section_num = int(section_num_text)

    print(section_prompt,section_response)
    return extracted_title_num, extracted_section_num

if __name__ == "__main__":

    total_sections = 0
    total_NOtitle = 0
    total_titlewrong = 0
    total_titleright_NOsection = 0
    total_titleright_sectionwrong = 0
    list_titleright_wrong_and_right_sections = []
    total_bothright = 0

    dict_len_all = {}
    dict_len_titleright = {}
    dict_len_bothright = {}

    for title in range(1, 55):
        # if title > 15:
        #     break
        print("Title", title, end="\t")
        sections = get_sections_from_title(title)
        if sections is None:
            print("None")
        else:
            print("num sections =", len(sections))
            for sect in random.sample(sections, k=10): # using sample instead of choice ensures no replacement
                total_sections += 1
                extracted_title, extracted_section = GPT3_id(sect, title) # KEY CALL!
                num_words = len(sect[1].split())
                if num_words not in dict_len_all:
                    dict_len_all[num_words] = 0
                dict_len_all[num_words] += 1
                assert 100 <= num_words <= 1000, "Was predone?"
                print("extracted:", extracted_title, extracted_section)
                if extracted_title is None:
                    assert extracted_section is None, "expected both"
                    print("None, None")
                    total_NOtitle += 1
                else:
                    if extracted_title != title:
                        print("title wrong got", extracted_title, "expected", title)
                        total_titlewrong += 1
                    else:
                        if num_words not in dict_len_titleright: # for building a histogram
                            dict_len_titleright[num_words] = 0
                        dict_len_titleright[num_words] += 1

                        if extracted_section is None:
                            total_titleright_NOsection += 1
                            print("section", extracted_section)
                        elif extracted_section != sect[0]:
                            total_titleright_sectionwrong += 1
                            print("section wrong got", extracted_section, "expected", sect[0])
                            list_titleright_wrong_and_right_sections.append((sect[0], extracted_section))
                        else:
                            print("CORRECT!  :-)")
                            total_bothright += 1
                            if num_words not in dict_len_bothright:  # for building a histogram
                                dict_len_bothright[num_words] = 0
                            dict_len_bothright[num_words] += 1


        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("PROGRESS:")
        print("total_sections =", total_sections)
        print("total_NOtitle = ", total_NOtitle)
        print("total_titlewrong = ", total_titlewrong)
        print("total_titleright_NOsection = ", total_titleright_NOsection)
        print("total_titleright_sectionwrong = ", total_titleright_sectionwrong)
        print("list_titleright_wrong_and_right_sections = ", list_titleright_wrong_and_right_sections)
        print("total_bothright = ", total_bothright)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("HISTOGRAM DETAILS")
    for stat_len in range(min(dict_len_all.keys()), 1+max(dict_len_all.keys())):
        format_string = "{:4d} Total={:3d}  TitleRight={:3d} {:.3f}  BothRight={:3d} {:.3f}"
        if stat_len in dict_len_all:
            print(format_string.format(stat_len,
                                       dict_len_all[stat_len],
                                       dict_len_titleright.get(stat_len, 0),
                                       dict_len_titleright.get(stat_len, 0)/float(dict_len_all[stat_len]),
                                       dict_len_bothright.get(stat_len, 0),
                                       dict_len_bothright.get(stat_len, 0)/float(dict_len_all[stat_len])))


    if len(list_titleright_wrong_and_right_sections) > 0:
        print("median distance of section off=",
              statistics.median([abs(ss[0] - ss[1]) for ss in list_titleright_wrong_and_right_sections]))
    else:
        print("ALAS, len(list_titleright_wrong_and_right_sections)=0")

    with open("data_re_USC_knowledge.pkl", "wb") as f:
        pickle.dump((dict_len_all, dict_len_titleright, dict_len_bothright, list_titleright_wrong_and_right_sections), f)