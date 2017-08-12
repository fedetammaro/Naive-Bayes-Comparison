from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

print(__doc__)


def reuters():
    print('Temporary placeholder.')


def twenty_newsgroups():
    remove = ()  # TODO: remove header, footer and quotes or not?
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=remove)
    plot_graph(twenty_train.data, twenty_train.target)  # We pass documents first, then all categories they belong to


def plot_graph(documents, categories):
    x_train_countvect = CountVectorizer().fit_transform(documents)  # Learns a vocabulary dictionary and returns
    # a term-document matrix
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_countvect)  # Now, word frequencies instead of occurrences
    print('(N_samples, N_features_new): ', x_train_tfidf.shape, )  # (N_samples, N_features_new)

    dots, k_fold = plot_preparation()

    # TODO: implement timer to compare performances?

    plot_curve(MultinomialNB(), 'Multinomial Naive Bayes', x_train_tfidf, categories, k_fold, dots)
    plot_curve(BernoulliNB(), 'Bernoulli Naive Bayes', x_train_tfidf, categories, k_fold, dots)

    plt.show()


def plot_curve(estimator, title, X, y, k_fold, dots):
    plt.figure()
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Number of training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = k_fold,
                                                            n_jobs=2, train_sizes=np.linspace(.1, 1.0, dots))
    train_scores_mean = np.mean(train_scores, axis=1)  # NumPy method which computes arithmetic mean along the
    # specified axis
    train_scores_std = np.std(train_scores, axis=1)  # NumPy method which computes the standard deviation along the
    # specified axis
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()  # Matplotlib method which turns on the axes grid

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
      train_scores_mean + train_scores_std, alpha=0.1, color='r')  # Matplotlib method which makes filled polygons
    # between two curves. Creates a PolyCollection and fills the region between the two points on the vertical axis
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='b')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label=('Training score'))  # Matplotlib method which plots
    #  lines to the axes.
    plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label=('Cross-validation score'))

    plt.legend(loc='best')  # Matplotlib which draws a legend associated with axes
    return plt


def plot_preparation():
    while True:  # Until the user gives a correct input...
        print('Please input the number of desired dots for the graph (at least 2): ')
        try:
            dots = int(raw_input())
            if dots <= 1:
                print('Please input a number of dots of at least 2.')
            else:
                break
        except ValueError:
            print('Please input a numerical value (at least 2).')

    while True:  # Until the user gives a correct input...
        print('Please input the desired k for k-fold cross-validation: ')
        try:
            k_fold = int(raw_input())
            if k_fold == "" or k_fold < 2:
                print('Invalid k-fold number specified. Setting it by default to None.')
                return dots, None  # This must be done to avoid setting the k-fold value to an illegal value (it must be
                # either None or >= 2
            break
        except ValueError:
            print('Please input a numerical value.')

    return dots, k_fold


if __name__=='__main__':
    print("""
    Comparison between Multinomial Naive Bayes and Bernoulli Naive Bayes implementations.
    Which dataset would you like to use for this comparison?
    Type 1 for Reuters, 2 for 20newsgroups or 3 to exit this script.""")
    while True:
        #try:
        i = int(raw_input())
        if i is 1:
            reuters()
            print('Reuters execution completed.')
        elif i is 2:
            twenty_newsgroups()
            print('20newsgroups execution completed.')
        elif i is 3:
            break
        else:
            print('Unrecognised input. Please type 1 for Reuters, 2 for 20newsgroups or 3 to exit this script.')
        #except ValueError:
            #print('Please enter a numeric value. Type 1 for Reuters, 2 for 20newsgroups or 3 to exit this script.')