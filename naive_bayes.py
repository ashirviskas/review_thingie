import numpy as np
import random


class NaiveBayes:
    def __init__(self, negative_reviews, positive_reviews, val_split=0.2):
        self.negative_reviews = np.array(negative_reviews)
        self.positive_reviews = np.array(positive_reviews)
        self.P_WN = dict()
        self.P_WP = dict()
        self.P_W_IS_NEG = dict()
        self.negative_words = dict()
        self.positive_words = dict()
        self.INTERESTING_WORDS = 50
        self.NEGATIVE_THRESHOLD = 0.98  # if review gets more = is negative
        self.UNSEEN = 0.4
        # self.val_split = val_split
        # self.REVIEWS_FOR_EVALUATION = 200
        # splitting train and validation sets:
        self.negative_reviews_train, self.negative_reviews_val = self.split_train_val(self.negative_reviews, val_split)
        self.positive_reviews_train, self.positive_reviews_val = self.split_train_val(self.positive_reviews, val_split)

        self.populate_everything()

    def populate_everything(self):
        print(len(self.negative_reviews))
        print(len(self.negative_reviews_train))

        self.populate_word_dicts(self.negative_reviews_train, self.negative_words)
        self.populate_word_dicts(self.positive_reviews_train, self.positive_words)

        self.populate_probabilities(len(self.negative_reviews_train), len(self.negative_reviews_train))



    @staticmethod
    def populate_word_dicts(content, populate_dictionary):
        for review in content:
            for word in review:
                if word in populate_dictionary:
                    populate_dictionary[word] += 1
                else:
                    populate_dictionary[word] = 1

    @staticmethod
    def split_train_val(reviews, val_split):
        # reviews = np.array(reviews_)
        reviews_n = len(reviews)
        taking_sample = np.random.choice(reviews_n, int(reviews_n * (1-val_split)), replace=False)
        train_reviews = reviews[taking_sample]
        val_reviews_sample = np.unique(np.concatenate((taking_sample, np.arange(reviews_n))))
        val_reviews = reviews[val_reviews_sample]
        return train_reviews, val_reviews

    @staticmethod
    def read_and_populate_reviews(reviews, populate_dictionary):
        for review in reviews:
            for word in review:
                if word in populate_dictionary:
                    populate_dictionary[word] += 1
                else:
                    populate_dictionary[word] = 1

    def populate_wn(self, total_neg_reviews):
        for k, v in self.negative_words.items():
            self.P_WN[k] = v / total_neg_reviews

    def populate_wp(self, total_pos_reviews):
        for k, v in self.positive_words.items():
            self.P_WP[k] = v / total_pos_reviews

    def populate_np(self, neg_coeff):
        for k, v in self.negative_words.items():
            if k in self.positive_words:
                # main naive bayes formula
                self.P_W_IS_NEG[k] = (self.P_WN[k] * neg_coeff) / ((self.P_WN[k] * neg_coeff) + ((1 - neg_coeff) * self.P_WP[k]))
            else:
                self.P_W_IS_NEG[k] = 0.99

        for k, v in self.positive_words.items():
            if k not in self.P_W_IS_NEG:
                self.P_W_IS_NEG[k] = 0.01

    def populate_probabilities(self, total_neg, total_pos):
        self.populate_wn(total_neg)
        self.populate_wp(total_pos)
        neg_coeff = total_neg / (total_neg + total_pos)
        self.populate_np(neg_coeff)

    def how_far_from_normal(self, word):
        if word in self.P_W_IS_NEG:
            return abs(0.5 - self.P_W_IS_NEG[word])
        else:
            return 0.5 - self.UNSEEN

    def get_score_for_words(self, words):
        p_w = {}
        for word in words:
            if word in self.P_W_IS_NEG:
                p_w[word] = self.P_W_IS_NEG[word]
            else:
                p_w[word] = self.UNSEEN
        neg_prob = 1
        for pn in p_w.values():
            neg_prob = neg_prob * pn
        opposite_neg_prob = 1
        for p in p_w.values():
            opposite_neg_prob = opposite_neg_prob * (1 - p)
        if opposite_neg_prob == 0:
            return 0
        neg_prob = neg_prob / (neg_prob + opposite_neg_prob)
        return neg_prob

    def evaluate_naive_bayes(self):
        print("Native bayes evaluation")
        neg_reviews = self.negative_reviews_val
        pos_reviews = self.positive_reviews_val
        neg_evals = np.zeros((len(neg_reviews)), dtype=np.float)
        pos_evals = np.zeros((len(pos_reviews)), dtype=np.float)

        for i, review in enumerate(pos_reviews):
            pos_evals[i] = self.get_score_for_words(review)

        for i, review in enumerate(neg_reviews):
            neg_evals[i] = self.get_score_for_words(review)

        mse_pos = (np.square(pos_evals - np.zeros(pos_evals.shape))).mean(axis=0)
        mse_neg = (np.square(neg_evals - np.ones(pos_evals.shape))).mean(axis=0)
        correct_predictions = np.concatenate((np.where(pos_evals < 0.5, 1, 0), np.where(neg_evals > 0.5, 1, 0))).sum()
        total_predictions = len(neg_evals) + len(pos_evals)
        print("Accuracy: ", correct_predictions / total_predictions)
        print()
        # print("MSE total: ", (mse_pos + mse_neg) / 2)
        # print("MSE for positive validation reviews: ", mse_pos)
        # print("MSE for negative validation reviews: ", mse_neg)
        return correct_predictions / total_predictions
