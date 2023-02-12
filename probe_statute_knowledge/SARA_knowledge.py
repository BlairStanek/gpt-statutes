# Created 6 Feb 2023
# Aims to probe whether GPT3 knows about the statutes in SARA

import sys, os
sys.path.append('../')
import utils

PROMPT = "\nWhere is the text above from?"
STATUTES_DIR = "../statutes_compacted"

for filename in sorted(os.listdir(STATUTES_DIR)):
    utils.add_comment("probe GPT3 knowledge of SARA " + filename + " in " + __file__)
    with open(STATUTES_DIR + "/" + filename, "r") as f:
        statute_text = f.read()
    assert statute_text[0] == "§"
    # cut off the title line
    statute_text = statute_text[statute_text.find("\n") + 1:].strip()
    full_prompt = statute_text + PROMPT
    statute_response = utils.call_gpt3_withlogging(full_prompt, "text-davinci-003", max_tokens=2000)
    print(filename, "--------------------------")
    print(statute_response)
