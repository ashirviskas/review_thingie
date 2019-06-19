import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math


class LogisticRegression:

    def __init__(self, negative_reviews, positive_reviews, val_split=0.2):

        self.negative_reviews = np.array(negative_reviews)
        self.positive_reviews = np.array(positive_reviews)

        self.model = None

        self.positive_words = dict()
        self.negative_words = dict()

        # self.vectorised_positive_revs = self.vectorise_multiple_reviews(self.positive_reviews)
        # self.vectorised_negative_revs = self.vectorise_multiple_reviews(self.negative_reviews)

        self.pos_rev_train, self.pos_rev_test = self.split_train_val(self.positive_reviews, val_split=val_split)
        self.neg_rev_train, self.neg_rev_test = self.split_train_val(self.negative_reviews, val_split=val_split)

        self.populate_word_dicts(self.negative_reviews, self.negative_words)
        self.populate_word_dicts(self.positive_reviews, self.positive_words)
        # self.all_words, self.all_word_tf_idf = self.calculate_all_tf_idf()

        # self.pos_rev_train_vect = self.vectorise_multiple_reviews(self.pos_rev_train)
        # self.neg_rev_train_vect = self.vectorise_multiple_reviews(self.neg_rev_train)

        # self.neg_rev_test_vect = self.vectorise_multiple_reviews(self.neg_rev_test)
        # self.pos_rev_test_vect = self.vectorise_multiple_reviews(self.pos_rev_test)
        # train_x = np.concatenate((self.neg_rev_train_vect, self.pos_rev_train_vect))

        train_y = np.concatenate((np.ones(len(self.pos_rev_train)), np.zeros(len(self.neg_rev_train))))
        all_reviews = np.concatenate((self.pos_rev_train, self.neg_rev_train))
        vectorizer = TfidfVectorizer("english")
        all_reviews = [" ".join(review) for review in all_reviews]
        train_x = vectorizer.fit_transform(all_reviews)
        test = vectorizer.transform(all_reviews)
        # train_x =
        random_sample = np.random.choice(len(train_y), len(train_y), replace=False)
        train_x = train_x[random_sample]
        train_y = train_y[random_sample]
        self.train_model(train_x, train_y)


    def vectorise_multiple_reviews(self, reviews):
        # reviews_joined = np.array([" ".join(review) for review in reviews])
        reviews_words = list()
        for r in reviews:
            for w in r:
                if w not in reviews_words:
                    reviews_words.append(w)
        reviews_words = {k: v for v, k in enumerate(reviews_words)}
        # print(len(reviews_words))
        vectorised_reviews = csr_matrix((len(reviews), len(reviews_words)))
        for i, review in enumerate(reviews):
            for word in review:
                tf= self.calculate_tf(word, review)
                idf = self.calculate_idf(word, reviews)
                tf_idf = tf * idf
                vectorised_reviews[i, reviews_words[word]] = tf_idf
                print(tf_idf)


        return vectorised_reviews, reviews_words

    # def vectorise_review(self, review):
    #     review_matrix = np.zeros((2, len(self.all_words)), dtype=np.float)
    #     for w in review:
    #         word_id = self.all_words[w]
    #         review_matrix[:, word_id] = self.all_word_tf_idf[word_id]

        # vectorizer = TfidfVectorizer("english")
        # message_mat = vectorizer.fit_transform([" ".join(review), "sveiki sveiki ar and the lol"])
        # print(message_mat)
        # return review_matrix.ravel()

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
        taking_sample = np.random.choice(reviews_n, int(reviews_n * (1 - val_split)), replace=False)
        train_reviews = reviews[taking_sample]
        val_reviews_sample = np.unique(np.concatenate((taking_sample, np.arange(reviews_n))))
        val_reviews = reviews[val_reviews_sample]
        return train_reviews, val_reviews

    def train_model(self, train_x, train_y):
        self.model = lr(solver='liblinear', penalty='l1')
        self.model.fit(train_x, train_y)

    # def calculate_all_tf_idf(self):
    #     all_word_dict = dict()
    #     word_i = 0
    #     all_word_tf_idf = list()
    #     total_words = [sum([v for v in self.positive_words.values()]), sum([v for v in self.negative_words.values()])]
    #     for w in self.negative_words:
    #         all_word_dict[w] = word_i
    #         word_i += 1
    #         tf_idf = self.calculate_tf_idf(w, [self.negative_words, self.positive_words], total_words)
    #         all_word_tf_idf.append(tf_idf)
    #
    #     for w in self.positive_words:
    #         if w not in all_word_dict:
    #             all_word_dict[w] = word_i
    #             word_i += 1
    #             tf_idf = self.calculate_tf_idf(w, [self.negative_words, self.positive_words], total_words)
    #             all_word_tf_idf.append(tf_idf)
    #     return all_word_dict, all_word_tf_idf
    #
    #

    @staticmethod
    def calculate_tf(word, document):
        return document.count(word) / len(document)

    @staticmethod
    def calculate_idf(word, documents):
        occurences = sum([1 for d in documents if word in d])
        n_docs = len(documents)
        idf = math.log(n_docs/occurences)
        return idf

    # def calculate_tf_idf(self, word, documents, total_words):
    #     tf = list()
    #     tf.append(self.calculate_tf(word, documents[0], total_words[0]))
    #     tf.append(self.calculate_tf(word, documents[1], total_words[1]))
    #     idf = self.calculate_idf(word, documents)
    #     return tf[0] * idf, tf[1] * idf

    def evaluate_logistic_regression(self):
        pred = self.model.predict(self.vectorise_multiple_reviews(self.pos_rev_test))
        print("Logistic regression evaluation")
        accuracy = accuracy_score(np.ones(len(pred)), pred)
        print("Accuracy: ", accuracy)
        return accuracy
