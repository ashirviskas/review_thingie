import re
import os
import numpy as np

negative_words = {}
positive_words = {}
P_WN = {}
P_WP = {}
P_W_IS_NEG = {}

# Number of words to take that are the most interesting (furthest from 0.5 probability of being negative)
INTERESTING_WORDS = 50
NEGATIVE_THRESHOLD = 0.98  # if review gets more = is negative
UNSEEN = 0.4


def get_count_of_word(word):
    count = 0
    if word in negative_words:
        count += negative_words[word]
    if word in positive_words:
        count += positive_words[word]
    return count


def populate_wn(total_neg_reviews):
    for k, v in negative_words.items():
        P_WN[k] = v / total_neg_reviews


def populate_wp(total_pos_reviews):
    for k, v in positive_words.items():
        P_WP[k] = v / total_pos_reviews


def populate_np(neg_coeff):
    for k, v in negative_words.items():
        if k in positive_words:
            P_W_IS_NEG[k] = (P_WN[k]) / (P_WN[k] + P_WP[k])
        else:
            P_W_IS_NEG[k] = 0.99

    for k, v in positive_words.items():
        if k not in P_W_IS_NEG:
            P_W_IS_NEG[k] = 0.01


def populate_probabilities(total_neg, total_pos):
    populate_wn(total_neg)
    populate_wp(total_pos)
    neg_coeff = total_neg / (total_neg + total_pos)
    populate_np(neg_coeff)


def how_far_from_normal(word):
    if word in P_W_IS_NEG:
        return abs(0.5 - P_W_IS_NEG[word])
    else:
        return 0.5 - UNSEEN


def get_score_for_words(words):
    p_w = {}
    words = sorted(words, key=how_far_from_normal, reverse=True)  # Sorting words by how far from middle (0.5) they are
    words = words[:INTERESTING_WORDS]  # Taking only the most interesting words
    for word in words:
        if word in P_W_IS_NEG:
            p_w[word] = P_W_IS_NEG[word]
        else:
            p_w[word] = UNSEEN
    neg_prob = 1
    for pn in p_w.values():
        neg_prob = neg_prob * pn
    opp = 1
    for p in p_w.values():
        opp = opp * (1 - p)
    if opp == 0:
        return 0
    neg_prob = neg_prob / (neg_prob + opp)
    return neg_prob


def read_and_populate_reviews(filename, populate_dictionary):
    content = read_reviews_in_file(filename, 0, -200)
    for review in content:
        for word in review:
            if word in populate_dictionary:
                populate_dictionary[word] += 1
            else:
                populate_dictionary[word] = 1
    return len(content)


def regex_words_out_of_line(line):
    return re.findall(r"[a-z0-9_$\']+", line)


def read_reviews_in_file(filename, start = 0, end = -1):
    content = list()
    with open(os.path.dirname(__file__) + filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word_list = regex_words_out_of_line(line)  # re.findall(r"[A-Za-z0-9_$\'\"]+", line)#line.split()
            content.append(word_list)
    print(content)
    print(len(content))
    return content[start:end]


def evaluate_naive_bayes():
    neg_reviews = read_reviews_in_file("/rt-polaritydata/rt-polarity.neg", -200, -1)
    pos_reviews = read_reviews_in_file("/rt-polaritydata/rt-polarity.pos", -200, -1)
    neg_evals = np.zeros((len(neg_reviews)), dtype=np.float)
    pos_evals = np.zeros((len(pos_reviews)), dtype=np.float)

    for i, review in enumerate(pos_reviews):
        pos_evals[i] = get_score_for_words(review)

    for i, review in enumerate(neg_reviews):
        neg_evals[i] = get_score_for_words(review)

    mse_pos = (np.square(pos_evals - np.zeros(pos_evals.shape))).mean(axis=0)
    mse_neg = (np.square(neg_evals - np.ones(pos_evals.shape))).mean(axis=0)
    print("MSE for positive validation reviews: ", mse_pos)
    print("MSE for negative validation reviews: ", mse_neg)


def main():
    total_neg = read_and_populate_reviews("/rt-polaritydata/rt-polarity.neg", negative_words)
    total_pos = read_and_populate_reviews("/rt-polaritydata/rt-polarity.pos", positive_words)
    populate_probabilities(total_neg, total_pos)
    print(P_WN)
    print(P_WP)
    print(P_W_IS_NEG)
    evaluate_naive_bayes()


if __name__ == "__main__":
    main()

