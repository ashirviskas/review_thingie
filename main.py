import re
import os
import numpy as np
from naive_bayes import NaiveBayes

negative_words = {}
positive_words = {}
P_WN = {}
P_WP = {}
P_W_IS_NEG = {}


def regex_words_out_of_line(line):
    return re.findall(r"[a-z0-9_$\']+", line)


def read_reviews_in_file(filename, start=0, end=-1):
    content = list()
    with open(os.path.dirname(__file__) + filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word_list = regex_words_out_of_line(line)  # re.findall(r"[A-Za-z0-9_$\'\"]+", line)#line.split()
            content.append(word_list)
    # print(content)
    # print(len(content))
    return content[start:end]


def main():
    neg_revs = read_reviews_in_file("/rt-polaritydata/rt-polarity.neg")
    pos_revs = read_reviews_in_file("/rt-polaritydata/rt-polarity.pos")
    nb = NaiveBayes(neg_revs, pos_revs, val_split=0.2)
    nb.evaluate_naive_bayes()


if __name__ == "__main__":
    main()

