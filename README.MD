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
For both methods, initial text processing removed all non word characters. Recasing isn't done, because text is already lowercase.

For Naive Bayes this is all that was done, tokenizing wasn't implemented.

For Logistic Regression words were tokenized to simple integer indices. Then each review is converted to a vectorized representation matrix consisting of counts of words. It is later used for training and evaluating.

**Why you have selected these two classification methods?**

No particular reason, it's just because they were given as example methods

**Compare selected classification methods. Which one is better? Why?**

In my case, Naive Bayes was faster and more accurate. It didn't need long training like this particular logistic regression implementation did, which took more than a few minutes (9 minutes for 1000 epochs) versus a few seconds.

## Results

Results were calculated using 20% of reviews as a validation set.

**Naive Bayes result:**

 - Accuracy:  0.915%

**Logistic Regression result:**

**Implementation from scratch**

Trained for 1000 epochs, 9 minutes:

 - Accuracy:  84.249%

Trained for 3000 epochs, 26 minutes:

 - Accuracy: 89.54%

 **Just for fun – implementation using Tensorflow**

Trained for 200 epochs (10 minutes):

 - Accuracy: 74.67%
