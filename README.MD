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

For Logistic Regression words were tokenized to simple integer indices. Then each review is converted to a vectorized representation matrix of if-idf values for each word that is later used for training and evaluating.

**Why you have selected these two classification methods?**

No particular reason, it's just because they were given as example methods

**Compare selected classification methods. Which one is better? Why?**

Idk yet lol

## Results
Results were calculated using 20% of reviews as a validation set.
**Naive Bayes result:**

 - Accuracy:  0.9154784240150093
 - MSE total:  0.20090065595346646

**Logistic Regression result:**
