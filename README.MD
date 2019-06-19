## Task
Classify movie reviews into positive and negative. Classification task should be done using two different classification methods (e.g. logistic regression and Naive Bayes).


Requirements

Use Python programming language.
High accuracy is not required for this task.


Dataset

[http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)


Questions

Describe text processing pipeline you have selected.
Why you have selected these two classification methods?
Compare selected classification methods. Which one is better? Why?


Results

Everything (source code, answers to questions, etc.) should be packed into single github repository.

## Chosen methods
**Naive Bayes** and **Logistic Regression**

## Answers
**Describe text processing pipeline you have selected.**

For both methods, initial text processing removed all non english word characters. Recasing isn't done, because text is already lowercase.

For Naive Bayes words are then put to dictionaries as keys with values being the repetition of a certain word in the dataset.

For Logistic Regression words were then tokenized to simple integer indices. Then each review is converted to a vectorized representation matrix consisting of counts of words. It is later used for training and evaluating.

**Why you have selected these two classification methods?**

No particular reason, it's just because they were given as example methods

**Compare selected classification methods. Which one is better? Why?**

In my case, Naive Bayes was faster and more accurate. It didn't need long training like this particular logistic regression implementation did, which took more than a few minutes (9 minutes for 1000 epochs) versus just a few seconds using Naive Bayes method.
Logistic regression is favorable when high accuracy is valued over training time which is generally a case with discriminative learning (http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf).

If I wanted a quick evaluation, I would use Naive Bayes method, as it is quite accurate and fast, but if I had more time and computing power, I would compare these both methods and use the one that is better suited for the particular task.


## Results

Results were calculated using 20% of reviews as a validation set.

**Naive Bayes result:**

 - Accuracy:  0.915%

**Logistic Regression result:**

**Implementation from scratch:**

Trained for 1000 epochs, 9 minutes:

 - Accuracy:  84.249%

Trained for 3000 epochs, 26 minutes:

 - Accuracy: 89.54%

 **Just for fun – implementation using Tensorflow:**

Trained for 200 epochs (10 minutes):

 - Accuracy: 74.67%
