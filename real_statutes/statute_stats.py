# Calculates statistics about real statutes, drawing from their XML representation
import xml.etree.ElementTree as ET
from collections import Counter
import os

usc_ns_str = "http://xml.house.gov/schemas/uslm/1.0"
usc_ns_str_curly = "{" + usc_ns_str + "}"
ns = {"usc" : usc_ns_str}

subdivision_types = [usc_ns_str_curly + "subsection",
                usc_ns_str_curly + "paragraph",
                usc_ns_str_curly + "subparagraph",
                usc_ns_str_curly + "clause",
                usc_ns_str_curly + "subclause",
                usc_ns_str_curly + "item",
                usc_ns_str_curly + "subitem",
                usc_ns_str_curly + "subsubitem"]

def get_stats_recursive(x, cur_depth = 0):
    max_depth = cur_depth
    max_width = 0
    count_leaves = 0
    count_nonleaves = 0
    total_depth_leaves = 0 # will be used to calculate average depth of a leaf
    total_branching_nonleaves = 0 # will be used to calculate average branching for nonleaves
    branching = 0
    for sub in x:
        if sub.tag in subdivision_types:
            branching += 1
            temp_depth, temp_max_width, temp_count_leaves, \
                temp_count_nonleaves, temp_total_depth_leaves, temp_total_branching_nonleaves = \
                get_stats_recursive(sub, cur_depth+1)
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
        # print(x.attrib.get("identifier", ""), "\t\tbranching=", branching, "\t\ttotal_branching_nonleaves=", total_branching_nonleaves)

    return max_depth, max_width, count_leaves, count_nonleaves, total_depth_leaves, total_branching_nonleaves


USC_XMLFILE_ROOT = "xml_uscAll@117-327not263not286" # downloaded from https://uscode.house.gov/download/download.shtml

status_counter = Counter()

print("{:10s}, {:20s}, {:5s}, {:5s}, {:5s}, {:5s}, {:3s}, {:4s}" \
      .format("text", "id", "branc", "leafD", "Nleaf", "Nnonl", "mxD", "maxW"))

# called once per section
def print_stats(text, id,
                count_leaves, count_nonleaves,
                max_depth, max_width,
                total_depth_leaves, total_branching_nonleaves):
    avg_branching = 0
    if count_nonleaves > 0:
        avg_branching = total_branching_nonleaves/float(count_nonleaves)
    avg_leaf_depth = total_depth_leaves / float(count_leaves) # there is always >0 leaves
    print("{:10s}, {:20s}, {:5.2f}, {:5.2f}, {:5d}, {:5d}, {:3d}, {:4d}" \
          .format(text, id, avg_branching, avg_leaf_depth,
                  count_leaves, count_nonleaves, max_depth, max_width))


for title in range(1, 55):
# for title in range(26, 27): # DEBUG
    # print("******** START TITLE", title)
    prefix = ""
    if title < 10:
        prefix = "0"
    filename = USC_XMLFILE_ROOT + "/usc" + prefix + str(title) + ".xml"
    if not os.path.exists(filename):
        continue

    title_tree = ET.parse(filename)
    title_root = title_tree.getroot()

    for s in title_root.iter('{' + usc_ns_str + '}section'):
        num = s.find('{' + usc_ns_str + '}num')
        if num is not None and \
            num.text is not None and \
            len(s.attrib.get("status","")) == 0 and \
            s.attrib.get("identifier","").startswith("/us/usc/t"):
            # if not s.attrib.get("identifier","").endswith("s7481"):  # DEBUG
            #     continue
            assert len(num.text) > 0
            max_depth, max_width, count_leaves, \
               count_nonleaves, total_depth_leaves, total_branching_nonleaves = \
                get_stats_recursive(s)

            print_stats(num.text, s.attrib.get("identifier", ""),
                        count_leaves, count_nonleaves,
                        max_depth, max_width,
                        total_depth_leaves, total_branching_nonleaves)

            # print(num.text, "\t\t",
            #       s.attrib.get("identifier", ""), "\t\t",
            #       "max_depth=", max_depth, "\t\t",
            #       "max_width=", max_width, "\t\t",
            #       "count_leaves=", count_leaves, "\t\t",
            #       "count_nonleaves=", count_nonleaves, "\t\t",
            #       "average_leaf_depth={:.2f}".format(float(total_depth_leaves)/count_leaves), "\t\t"
            #       "total_branching_nonleaves=", total_branching_nonleaves, "\t\t"
            #       "average_branching={:.2f}".format(average_branching))

            # print(ET.tostring(s, encoding='utf8').decode('utf8'))  DEBUG

            status_counter.update([s.attrib.get("status","none provided")])

print(status_counter)