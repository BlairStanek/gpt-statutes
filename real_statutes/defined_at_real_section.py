# This is a real-statute version of the synthetic-statute defined_at_synstat.py
import xml.etree.ElementTree as ET
from real_stat_utils import subdivision_types, usc_ns_str, ns, usc_ns_str_curly, StatLine, FLUSH_LANGUAGE
from collections import Counter
import tiktoken
import argparse, random, sys, re, os, numpy, datetime
sys.path.append('../')
import utils

