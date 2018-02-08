# COMS 4705 - NLP - Prof. Daniel Bauer

## Homework 1 - Building a Trigram Language Model

This is a small homework that I did in the context of an NLP course at Columbia University.
The script _trigram_model.py_ contains a class Trigram Model that builds a trigram model over a training corpus, and can be used to get the [perplexity](https://en.wikipedia.org/wiki/Perplexity) of a test corpus, which is a metric that evaluates how well the model predicts each word (low perplexity is better).

### Model

When training, the model sets all words with count 1 in the training courpus as unknown tokens ('UNK'), which allows to approximate the probability of truly unseen tokens on the test corpus (which are also set as 'UNK').

The model uses linear interpolation to cope with unseen trigrams: the probability of a trigram is a linear function of the unigram and bigram it contains, therefore the probabilities are smoothed and no trigram gets a probability of 0.

## Predictions

We use the dataset to perform binary classification over a dataset of essays which can have a "high score" or "low score":
We first train a trigram model on a training dataset of low scores, and a second trigram model on a training dataset of high scores.

To predict the score of an unknown essay, we compute its complexity with respect to each model. Perplexity being a metric that describes how well words of a corpus are predicted by an n-gram model, and for which lowest complexity is better, we assign the essay to the class which model returns the lowest complexity.

### Data

The dataset used for this project being proprietary, it has not been added to this repo.
Yet a small dataset used to debug the models can be found in the _data_ folder.