import os, re
import xml.etree.ElementTree as ET

usc_ns_str = "http://xml.house.gov/schemas/uslm/1.0"
usc_ns_str_curly = "{" + usc_ns_str + "}"
ns = {"usc" : usc_ns_str}

FLUSH_LANGUAGE = "flush"

subdivision_types = [usc_ns_str_curly + "subsection",
                usc_ns_str_curly + "paragraph",
                usc_ns_str_curly + "subparagraph",
                usc_ns_str_curly + "clause",
                usc_ns_str_curly + "subclause",
                usc_ns_str_curly + "item",
                usc_ns_str_curly + "subitem",
                usc_ns_str_curly + "subsubitem"]

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

# returns a 2-tuple:  a list of leaves, and a list of lists of StatLines
def load_statutes(min_depth:int):
    list_leaves = []  # list of Leaf's
    list_list_statlines = [] # list of list of StatLine's (each list corresponds to a statute)
    print("Loading Titles ", end="")
    # for title in range(1, 55):
    for title in range(26, 27):  # for debug
        print(title, end=" ", flush=True)
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
                if len(list(tables)) == 0 and len(list(layouttables)) == 0: # skip ALL sections with tables; they make queries ill-defined
                    statlines = []
                    _, leaves_raw = parse_xml_statute(s, statlines)
                    list_list_statlines.append(statlines) # save the statute

                    # for line in statlines:
                    #     print("{:32s} |".format(line.identifier), line.text)
                    #     if "\n" in line.text:
                    #         print("+++++++++ Newline above")

                    if len(leaves_raw) > 1: # ignore leaves from sections with just one valid leaf
                        # first filter by min-depth
                        filtered_leaves = []
                        for leaf in leaves_raw:
                            if leaf.level >= min_depth:
                                filtered_leaves.append(leaf)
                        # ignore leaves from sections with zero or one leaf sufficiently deep
                        if len(filtered_leaves) > 1:
                            # calculate leaf percentiles
                            for idx_leaf, leaf in enumerate(filtered_leaves):
                                leaf.percentile = float(idx_leaf) / (len(filtered_leaves) - 1)
                            list_leaves.extend(filtered_leaves) # actually add to the list we return

    print("len(list_leaves)=", len(list_leaves))
    print("len(list_list_statlines)=", len(list_list_statlines))
    return list_leaves, list_list_statlines
