# small helper file to get the XML of a particular section for further inspection/debugging
import xml.etree.ElementTree as ET
from real_stat_utils import subdivision_types, usc_ns_str, ns, usc_ns_str_curly

TITLE = "34"
SECTION = "20942"
title_tree = ET.parse("xml_uscAll@117-327not263not286/usc" + TITLE + ".xml")
title_root = title_tree.getroot()

for s in title_root.iter('{' + usc_ns_str + '}section'):  # this loop builds up the list of possibilities
    num = s.find('{' + usc_ns_str + '}num')
    if s.attrib.get("identifier", "") == "/us/usc/t" + TITLE + "/s" + SECTION:
        print(ET.tostring(s, encoding='utf8').decode('utf8'))
        exit(1)
