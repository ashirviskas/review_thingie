import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, negative_reviews, positive_reviews, val_split=0.2, lr=0.1, num_inter=10, treshold = 0.5):
        self.lr = lr
        self.num_iter = num_inter
        self.threshold = treshold

        self.negative_reviews = np.array(negative_reviews)
        self.positive_reviews = np.array(positive_reviews)

        # Splitting review data for training and testing
        self.pos_rev_train, self.pos_rev_test = self.split_train_val(self.positive_reviews, val_split=val_split)
        self.neg_rev_train, self.neg_rev_test = self.split_train_val(self.negative_reviews, val_split=val_split)

        # Concatenating negative and positive review data
        self.all_rev_train = np.concatenate((self.neg_rev_train, self.pos_rev_train))
        self.all_rev_test = np.concatenate((self.neg_rev_test, self.pos_rev_test))

        # Building word dictionary of training review data
        word_dict = self.build_dictionary(self.all_rev_train)

        # Making vectorised representations of training and testing data
        vectorisations_train = self.vectorize_reviews(self.all_rev_train, word_dict)
        vectorisations_test = self.vectorize_reviews(self.all_rev_test, word_dict)

        self.train_x = vectorisations_train

        # Making labels for training data
        self.train_y = np.concatenate((np.ones(len(self.neg_rev_train)), np.zeros(len(self.pos_rev_train))))

        # Mixing train data
        random_sample = np.random.choice(len(self.train_x), len(self.train_x), replace=False)
        self.train_x = self.train_x[random_sample]
        self.train_y = self.train_y[random_sample]

        # Training
        train_results = self.train(self.train_x, self.train_y)

        # Assigning test data and making labels
        self.test_x = vectorisations_test
        self.test_y = np.concatenate((np.ones(len(self.neg_rev_test)), np.zeros(len(self.pos_rev_test))))

    @staticmethod
    def add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self, X, y):
        print("Training: ")
        losses = []
        X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        print(self.theta)

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            loss = self.loss(h, y)
            losses.append(loss)
            if i % 10 == 0:
                print(f'loss: {loss} \t', f'epoch: {i}')
        plt.plot(losses)
        plt.show()
        return True

    def predict(self, X):
        X = self.add_intercept(X)
        prob = self.sigmoid(np.dot(X, self.theta))
        return prob >= self.threshold

    def evaluate(self, test_x, test_y):
        y_predicted = self.predict(test_x)
        correct = 0
        for i, y in enumerate(test_y):
            if y == 0:
                y = False
            else:
                y = True
            if y == y_predicted[i]:
                correct = correct + 1
        total = y_predicted.size

        return (correct / total) * 100

    @staticmethod
    def build_dictionary(reviews):
        # reviews_joined = np.array([" ".join(review) for review in reviews])
        reviews_words = list()
        for r in reviews:
            for w in r:
                if w not in reviews_words:
                    reviews_words.append(w)
        reviews_words = {k: v for v, k in enumerate(reviews_words)}
        return reviews_words

    @staticmethod
    def vectorize_reviews(reviews, word_dict):
        vectorisations = np.zeros((len(reviews), len(word_dict)), dtype=int)
        for i, review in enumerate(reviews):
            for word in review:
                try:
                    index = word_dict[word]
                    vectorisations[i, index] += 1
                except KeyError:
                    pass
        return vectorisations


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


    def evaluate_logistic_regression(self):
        # pred = self.evaluate()
        print("Logistic regression evaluation")
        accuracy = accuracy = self.evaluate(self.test_x, self.test_y)
        print("Accuracy: ", accuracy)
        return accuracy
