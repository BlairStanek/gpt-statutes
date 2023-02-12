# Created 30 Jan 2023
# Provides helper functions for doing SARA tests against GPT3

import os, openai, time
from datetime import datetime

GPT3_LOGFILE = "gpt3_log.txt"

def add_comment(comment:str):
    f = open(GPT3_LOGFILE, "a")
    f.write(datetime.now().strftime("%A %d-%B-%Y %H:%M:%S") + "  COMMENT:" + comment + "\n")
    f.flush()
    f.close()

def call_gpt3_withlogging(prompt:str,
                          engine:str,
                          temperature=0.0,
                          max_tokens=256,
                          top_p=1.0,
                          frequency_penalty=0.0,
                          presence_penalty=0.0) -> str:
    openai.api_key = os.getenv("GPT3_API_KEY")

    f = open(GPT3_LOGFILE, "a")
    f.write("************************\n")

    f.write(datetime.now().strftime("%A %d-%B-%Y %H:%M:%S") + "\n")
    f.write("engine=" + engine + " temp={:.2f}".format(temperature) +
            " max_tokens=" + str(max_tokens) + " top_p={:.2f}".format(top_p) +
            " freq_pen={:.2f}".format(frequency_penalty) +
            " pres_pen={:.2f}".format(presence_penalty) + "\n")
    f.write(prompt + "\n")
    f.write("------- (prompt above/response below)\n")

    worked = False
    while not worked:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            response_text = response['choices'][0]['text']
            worked = True
        except openai.error.ServiceUnavailableError:
            print("ServiceUnavailableError error, retrying in 2s.", end="")
            time.sleep(2)
        except openai.error.RateLimitError:
            print("RateLimitError error, retrying in 2s.", end="")
            time.sleep(2)
        except openai.error.APIConnectionError:
            print("APIConnectionError error, retrying in 2s.", end="")
            time.sleep(2)
        except openai.error.APIError:
            print("APIError error, retrying in 2s.", end="")
            time.sleep(2)
        except openai.error.Timeout:
            print("Timeout error, retrying in 2s.", end="")
            time.sleep(2)




    assert worked

    f.write(response_text + "\n")
    f.write("************************\n")
    f.flush()
    f.close()

    return response_text

def get_cases(test_or_train:str, exclude_dollars=False, only_tax_cases=False) -> list:
    rv = []
    f = open("./sara_v2/splits/" + test_or_train, "r")
    for l in f.readlines():
        if only_tax_cases and not l.startswith("tax_case"):
            continue

        f_casefile = open("./sara_v2/cases/" + l.strip() + ".pl", "r")

        text = ""
        line = f_casefile.readline()
        assert line.strip() == "% Text"
        line = f_casefile.readline()
        while line.startswith("% "):
            text += line[len("% "):]
            line = f_casefile.readline()

        question = ""
        line = f_casefile.readline()
        assert line.strip() == "% Question"
        line = f_casefile.readline()
        while line.startswith("% "):
            question += line[len("% "):]
            line = f_casefile.readline()

        remaining_text = ""
        for line in f_casefile.readlines():
            remaining_text += line

        if not exclude_dollars or not ("$" in text or "$" in question):
            rv.append((l.strip(), text.strip(), question.strip(), remaining_text.strip()))

    return rv

def print_case_breakdown():
    for split in ["train", "test"]:
        print("SARA", split, ":")
        total = get_cases(split)
        nodollar = get_cases(split, exclude_dollars=True)
        tax_only = get_cases(split, exclude_dollars=False, only_tax_cases=True)
        print("     nodollar =", len(nodollar))
        print("dollar entail =", len(total) - len(nodollar) - len(tax_only))
        print("      taxonly =", len(tax_only))
        print("        TOTAL =", len(total))

def is_entail(text) -> bool:
    return text.lower().replace(".", "").strip() == "entailment"

def is_contra(text) -> bool:
    return text.lower().replace(".", "").strip() == "contradiction"

def is_entail_or_contra(text) -> bool:
    return is_contra(text) or is_entail(text)

# Even when given the second prompt "Therefore, the answer (Yes or No) is", GPT3 sometimes
# gives answers with lots of random punctuation other than just "Yes" or "No".
# Annoyingly this sometimes includes a whole long sentence after the "Yes" or "No"
def is_match(response:str, query:str) -> bool:
    clean = response.replace(".","").replace(":","").replace("-","").replace(":","").strip()
    return clean.lower().startswith(query)
def is_yes(response:str) -> bool:
    return is_match(response, "yes")
def is_no(response:str) -> bool:
    return is_match(response, "no")

AMBIGUOUS_WORDS = ["depend", "depends", "dependent", "may", "maybe", "if", "but"] # a suggestion of problem words
def warning_if_problem_words(list_problem_words, target:str, context:str) -> str:
    rv = ""
    for problem_word in list_problem_words:
        if problem_word.lower() in target.lower().split():
            rv += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            rv += "!WARNING: found problem word " + problem_word + " in: " + target + "\n"
            rv += "!Context: " + context + "\n"
            rv += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    return rv

def reformat_case(text, first, second, third, skip_third=False, add_cite_before_section=False) -> str:
    if add_cite_before_section: # used for testing without the statutes in the prompt
        text = text.replace("section", "I.R.C. section")
        text = text.replace("Section", "I.R.C. section")

    assert len(text.split("\n")) == 2
    first_part = text.split("\n")[0]
    if not skip_third:
        last = text.split("\n")[1].split()[-1]
        assert is_entail_or_contra(last)
    else:
        last = ""
    second_part = text.split("\n")[1]
    second_part = second_part[0:len(second_part) - len(last)].strip() # off with "Entailment" etc

    return first + first_part + "\n" + \
            second + second_part + "\n" + \
            third + last


def print_confusion_matrix(entail_cor, contra_cor, entail_answercontra, contra_answerentail):
    print("                ", "      Predicted         ")
    print("                ", "  entail      contrad.  ")
    print("Actual  entail  ", "  {:6d} ".format(entail_cor), "   {:6d} ".format(entail_answercontra))
    print("        contrad.", "  {:6d} ".format(contra_answerentail), "   {:6d} ".format(contra_cor))
    total = entail_cor+ contra_cor+ entail_answercontra+ contra_answerentail
    corr = entail_cor+ contra_cor
    if total > 0:
        print("Accuracy:", corr, "/", total, "=", corr/float(total))
    else:
        assert corr == 0
        print("Accuracy:", 0, "/", 0, "=", 0)


def remove_statute_whitespace(orig_text) -> str:
    rv = orig_text.replace("\n\n", "\n")
    return rv

# if __name__ == "__main__":
#     print("TEST")
#     tests = get_cases("test", True)
#     for t in tests:
#         print(t)
#         assert t[2].endswith("Contradiction") or t[2].endswith("Entailment")
#     print("TRAIN")
#     train = get_cases("train", True)
#     for t in train:
#         print(t)
#         assert t[2].endswith("Contradiction") or t[2].endswith("Entailment")
#
#     print_case_breakdown()
