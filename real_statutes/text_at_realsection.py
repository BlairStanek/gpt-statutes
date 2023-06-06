# Takes a particular section of a real statute and calls GPT-* to see if it can
# identify the text.

import xml.etree.ElementTree as ET

usc_ns_str = "http://xml.house.gov/schemas/uslm/1.0"

ns = {"usc" : usc_ns_str}

irc_tree = ET.parse("xml_uscAll@117-327not263not286/usc26.xml")
irc_root = irc_tree.getroot()

# These are derived from usctitle.css downloaded from https://uscode.house.gov/download/resources/schemaandcss.zip
def get_indent(class_str) -> int:
    left_margin = 0
    indent = 0

    for style in class_str.split():
        if style == "firstIndent-4":
            indent = -4
        elif style == "firstIndent-3":
            indent = -3
        elif style == "firstIndent-2":
            indent = -2
        elif style == "firstIndent-1":
            indent = -1
        elif style == "firstIndent0":
            indent = 0
        elif style == "firstIndent2":
            indent = 2
        elif style == "indent0":
            indent += 1
            pass
        elif style == "indent1":
            left_margin += 1
            indent += 1
        elif style == "indent2":
            left_margin += 2
            indent += 1
        elif style == "indent3":
            left_margin += 3
            indent += 1
        elif style == "indent4":
            left_margin += 4
            indent += 1
        elif style == "indent5":
            left_margin += 5
            indent += 1
        elif style == "indent6":
            left_margin += 6
            indent += 1
        elif style == "indent7":
            left_margin += 7
            indent += 1

    return max(0, left_margin + indent)

# This builds up the text (in the same way as the House website).
def print_text_exactly(x:ET.Element, top_level = True) -> str:
    # We ignore historical, notes, etc. and focus on just the statute
    if "status" in x.attrib:
        assert x.attrib["status"] == 'repealed'
        return ""
    if "sourceCredit" in x.tag or "notes" in x.tag:
        return ""

    rv = ""
    if "class" in x.attrib and "indent" in x.attrib["class"].lower():
        # an indent* attribute indicates that it is a newline
        rv += "\n" + ("  " * get_indent(x.attrib["class"]))

    if x.text is not None:
        rv += x.text

    for sub in x:
        rv += print_text_exactly(sub, False) # recursively build up

    if x.tail is not None:
        rv += x.tail + " "
    return rv.rstrip()



########## Start actual code #############
sec_num = "66" # debug
irc_sec = None
for s in irc_root.iter('{' + usc_ns_str + '}section'):
    if "identifier" in s.attrib and \
            s.attrib["identifier"].lower() == "/us/usc/t26/s" + sec_num:
        assert irc_sec is None, "Should be only one match"
        irc_sec = s
        assert s.find("usc:num", ns).attrib["value"].lower() == sec_num

print(ET.tostring(irc_sec, encoding='utf8').decode('utf8'))

print(print(print_text_exactly(irc_sec).strip()))


# def recursive_tag_print(s):
#     print(s.tag)
#     for sub in s:
#         recursive_tag_print(sub)
# recursive_tag_print(irc_sec)

tables = irc_sec.iter("{http://www.w3.org/1999/xhtml}table")
print("table:", len(list(tables)))





















































