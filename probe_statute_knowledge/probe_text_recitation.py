# Probes whether GPT3 can *recite* arbitrary sections of the U.S. Code based on the cite.
import numpy

import USC_knowledge
import sys, random, numpy, os, tqdm, pickle
sys.path.append('../')
from sacrebleu.metrics import BLEU
import utils

random.seed(42)

results = [] # will be pickled
actual_recalled_at = []
percentile_actual_recall = []
list_raw_section_distance = [] # raw section distance from actual section and one with best BLEU
recall_at1_by_title = [0] * 55

NUM_PER_TITLE = 10

# SARA_sections = [1, 2, 63, 68, 151, 152, 3301, 3306, 7703]
for title in range(1, 55):
    # if title > 15:
    #     break
    print("Title", title, end="\t")
    all_title_sections = USC_knowledge.get_sections_from_title(title, 1, 1000000000)
    if all_title_sections is None: # some titles are empty; just continue
        continue
    assert len(all_title_sections) > NUM_PER_TITLE, "Expected enough"
    print("num sections =", len(all_title_sections))
    for s, _, _ in random.sample(all_title_sections, k=NUM_PER_TITLE):  # using sample instead of choice ensures no replacement
        print("title", title, "section", s, "******************************************")
        gpt_fileloc = "gpt_output/" + str(title) + "usc" + str(s) + ".txt"
        if os.path.exists(gpt_fileloc): # if already queried, don't call GPT3 again!
            print("Loading preexisting", gpt_fileloc)
            with open(gpt_fileloc, "r") as f:
                response = f.read()
        else:
            prompt = "The text of " + str(title) + " U.S. Code section " + str(s) + " is:"
            print("Calling GPT3:", prompt)
            response = utils.call_gpt3_withlogging(prompt, engine="text-davinci-003", max_tokens=3000)
            with open(gpt_fileloc, "w") as f_out:
                f_out.write(response)

        # Go over all the sections in the title and calculate the non-penalized BLEU score
        bleu = BLEU()
        bleu_against_all_actual = []
        for actual_section in tqdm.tqdm(all_title_sections):
            actual_section_text = actual_section[1]
            bleu_score = bleu.corpus_score([response], [[actual_section_text]])
            if bleu_score.score == 0 or bleu_score.bp == 0:
                unpenalized_bleu = 0
            else:
                unpenalized_bleu = bleu_score.score / bleu_score.bp
            assert bleu_score.bp <= 1
            assert 0 <= unpenalized_bleu <= 100
            bleu_against_all_actual.append((actual_section[0], unpenalized_bleu))
        print("")

        # Sort based on bleu score
        bleu_against_all_actual.sort(key=lambda x: x[1], reverse=True)
        found_actual = False
        for raw_actual_recalled_atN in range(len(bleu_against_all_actual)):
            if bleu_against_all_actual[raw_actual_recalled_atN][0] == s:
                found_actual = True
                actual_section_bleu = bleu_against_all_actual[raw_actual_recalled_atN][1]
                break
        assert found_actual
        actual_recalled_atN = 1 + raw_actual_recalled_atN # recall@N is 1-based not 0
        if actual_recalled_atN == 1:
            recall_at1_by_title[title] += 1
        print("Actual section was recalled at", actual_recalled_atN, "with unpenalized-BLEU value", bleu_against_all_actual[raw_actual_recalled_atN][1])
        print("Top 10:", bleu_against_all_actual[:10])
        print("Average BLEU against all sections in entire title:", numpy.mean([x[1] for x in bleu_against_all_actual]))
        raw_section_distance = abs(bleu_against_all_actual[0][0] - s)
        print("Raw distance:", raw_section_distance)
        list_raw_section_distance.append(raw_section_distance)
        percentile_actual_recall.append(raw_actual_recalled_atN/float(len(all_title_sections)))
        print("percentile actual recall (reverse):", percentile_actual_recall[-1])

        actual_recalled_at.append(actual_recalled_atN)

        results.append({"title": title, "section": s,
                        "actual_recalled_atN": actual_recalled_atN, "actual_section_bleu": actual_section_bleu,
                        "best_section": bleu_against_all_actual[0][0], "best_section_bleu": bleu_against_all_actual[0][1],
                        "title_num_sections": len(all_title_sections)})

print("actual_recalled_at=", actual_recalled_at)
print("TOTAL =", len(actual_recalled_at))
print("recall @ 1:", len([x for x in actual_recalled_at if x == 1]))
print("recall @ 5:", len([x for x in actual_recalled_at if x <= 5]))
print("MRR=", numpy.mean([1/x for x in actual_recalled_at]))

print("raw_section_distance=", list_raw_section_distance)
print("median(raw_section_distance)=", numpy.median(list_raw_section_distance))
print("median(section distances where not 0)=", numpy.median([x for x in list_raw_section_distance if x>0]))
print("mean(percentile_actual_recall)=", numpy.mean(percentile_actual_recall))

for idx, recalled_at1 in enumerate(recall_at1_by_title):
    if recalled_at1 > 0:
        print("Title", idx, "recall@1 for", recalled_at1)

print(results)

with open("probe_text_recitation_results.pkl", "wb") as f:
    pickle.dump(results, f)