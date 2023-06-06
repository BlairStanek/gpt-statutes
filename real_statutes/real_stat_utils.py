

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
