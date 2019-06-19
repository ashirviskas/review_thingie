import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class LogisticRegression:

    def __init__(self, negative_reviews, positive_reviews, val_split=0.2, lr=0.1, num_inter=10, treshold = 0.5):
        self.lr = lr
        self.epochs = num_inter
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
        self.train_y = np.zeros((len(self.neg_rev_train) + len(self.pos_rev_train), 2))
        self.train_y[0:len(self.neg_rev_train), 0] = 1
        self.train_y[len(self.neg_rev_train):len(self.neg_rev_train) + len(self.pos_rev_train), 1] = 1

        # Mixing train data
        random_sample = np.random.choice(len(self.train_x), len(self.train_x), replace=False)
        self.train_x = self.train_x[random_sample]
        self.train_y = self.train_y[random_sample]


        # Assigning test data and making labels
        self.test_x = vectorisations_test
        self.test_y = np.zeros((len(self.neg_rev_test) + len(self.pos_rev_test), 2))
        self.test_y[0:len(self.neg_rev_test), 0] = 1
        self.test_y[len(self.neg_rev_test):len(self.neg_rev_test) + len(self.pos_rev_test), 1] = 1

        n = len(word_dict)
        self.init_tensorflow_variables(n, lr)
        self.train(self.train_x, self.train_y, self.test_x, self.test_y)

    def init_tensorflow_variables(self, n, lr):
        self.X = tf.placeholder(tf.float32, [None, n])

        # Since this is a binary classification problem,
        # Y can take only 2 values.
        self.Y = tf.placeholder(tf.float32, [None, 2])

        # Trainable Variable Weights
        self.W = tf.Variable(tf.zeros([n, 2]))

        # Trainable Variable Bias
        self.b = tf.Variable(tf.zeros([2]))

        # Hypothesis
        self.Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(self.X, self.W), self.b))

        # Sigmoid Cross Entropy Cost Function
        self.cost = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.Y_hat, labels=self.Y)

        # Gradient Descent Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(self.cost)

        # Global Variables Initializer
        self.init = tf.global_variables_initializer()

    def train(self, train_x, train_y, test_x, test_y):

        with tf.Session() as sess:

            # Initializing the Variables
            sess.run(self.init)

            # Lists for storing the changing Cost and Accuracy in every Epoch
            cost_history, accuracy_history = [], []

            # Iterating through all the epochs
            for epoch in range(self.epochs):
                cost_per_epoch = 0

                # Running the Optimizer
                sess.run(self.optimizer, feed_dict={self.X: train_x, self.Y: train_y})

                # Calculating cost on current Epoch
                c = sess.run(self.cost, feed_dict={self.X: train_x, self.Y: train_y})

                # Calculating accuracy on current Epoch
                correct_prediction = tf.equal(tf.argmax(self.Y_hat, 1),
                                              tf.argmax(self.Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))

                # Storing Cost and Accuracy to the history
                cost_history.append(sum(sum(c)))
                accuracy_history.append(accuracy.eval({self.X: train_x, self.Y: train_y}) * 100)

                # Displaying result on current Epoch
                if epoch % 10 == 0:
                    print("Epoch " + str(epoch) + " Cost: "
                          + str(cost_history[-1]))

            Weight = sess.run(self.W)  # Optimized Weight
            Bias = sess.run(self.b)  # Optimized Bias

            # Final Accuracy
            correct_prediction = tf.equal(tf.argmax(self.Y_hat, 1),
                                          tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32))
            accuracy_history.append(accuracy.eval({self.X: train_x, self.Y: train_y}) * 100)

            accuracy_history.append(accuracy.eval({self.X: test_x, self.Y: test_y}) * 100)

            # Test accuracy:

            print("\nTrain Accuracy:", accuracy_history[-2], "%")
            print("\nTest Accuracy:", accuracy_history[-1], "%")
            print(accuracy)

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
