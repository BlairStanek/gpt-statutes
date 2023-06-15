# does basic work of loading/preprocessing transcripts
import re, os

DEPOSITION_DIRECTORY = "RawDepositions"

# Note that all the transcripts were expert witness depositions since 2015 from
# the Lexis expert witness deposition database, randomly selected, with the
# non-substantive fluff at the start and end edited out.
def load_bracket_transcript(filename) -> list:
    f_in = open(os.path.join(DEPOSITION_DIRECTORY,filename), "r")
    rv = []
    for line in f_in.readlines():
        # Basically, as long as there is a single alphanumerical character (\w) after
        # the [number], then we assume it's a valid line.
        m = re.match("\[\d+\](?P<txt>.*\w.*)$", line.strip())
        if m is not None:
            text = m.group("txt").strip()
            # replace weird pagination
            text = re.sub("\[\*\d+\]", "", text)
            # replace weird whitespace with standard spaces
            text = re.sub("\s+", " ", text)
            rv.append(text)
    return rv

def list_to_text(lines:list) -> str:
    rv = ""
    for idx, line in enumerate(lines):
        rv += "[" + str(idx + 1) + "] " + line.strip() + "\n"
    return rv

# This returns a list of lists of strings of the specified length.
# They are drawn from all the raw deposition text we have.
def load_transcripts(num_lines:int) -> list:
    rv = []
    for deposition_file in sorted(os.listdir(DEPOSITION_DIRECTORY)):
        if deposition_file.endswith(".txt"):
            file_lines = load_bracket_transcript(deposition_file)
            for idx_run in range(len(file_lines) // num_lines):
                rv.append(file_lines[idx_run*num_lines:(idx_run+1)*num_lines])
    return rv

if __name__ == "__main__":
    transcripts = load_transcripts(20)
    for trans in transcripts:
        print("*****************")
        print(list_to_text(trans))
