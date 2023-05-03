# gpt-statutes
This is the code corresponding to our paper [*Can GPT-3 Perform Statutory Reasoning?*](
https://doi.org/10.48550/arXiv.2302.06100), where we probe GPT-3's ability to perform statutory reasoning.  The paper has been accepted to the 2023 International Conference on Artificial Intelligence and Law (ICAIL).

The directory **sara_run** contains the code and data corresponding to section 3 of our paper.  The directory **probe_statute_knowledge** contains the code for section 4 of our paper.  The directory **synthetic_statutes** contains the code and inputs for section 5 of our paper.

You store your OpenAI API key in an environment variable called GPT_API_KEY, which utils.py reads and uses for calls. 
