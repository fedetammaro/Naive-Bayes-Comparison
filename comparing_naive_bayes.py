from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import learning_curve

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

import reuters_utilities

__author__ = 'Federico Tammaro'
__email__ = 'federico.tammaro@stud.unifi.it'

"""
Most frequent topics/IDs found on "Information Retrieval Technology" and from McCallum-Nigam paper, ordered from the
least frequent to the most frequent
"""
reuters_topics_list = [('corn', 10), ('wheat', 9), ('ship', 8), ('interest', 7), ('trade', 6), ('crude', 5),
                       ('grain', 4), ('money-fx', 3), ('acq', 2), ('earn', 1)]


def filter_documents(documents_iterator, content, selected_topics):

    """
    Cycles through the entire Reuters dataset to filter documents, returning only those who belong to one of the 10 most
    used categories.

    documents_iterator is the iterator of the dataset, used to cycle through all documents
    content is the format of the content we would like to retrieve from all the documents (e.g. title and body)
    topic_list is the list of categories we would like to retrieve

    Returns two lists, one containing all the documents which have one of the selected categories, the other containing
    the corresponding category of each returned document
    """

    reuters_documents = []
    reuters_topics = []
    for document in itertools.islice(documents_iterator, 21578):
        category = filter_topics(document, selected_topics)
        if category:
            reuters_documents.append(content.format(**document))
            reuters_topics.append(category)
    return reuters_documents, reuters_topics


def filter_topics(document, selected_topics):

    """
    Checks if the selected document has at least one of the categories we want to retrieve.

    doc is the current document
    topic_list is the list of categories we would like to retrieve

    Returns the ID of a category if the document has one of the desired categories, returns false otherwise
    """

    for topic in selected_topics:
        if topic[0] in document['topics']:
            return topic[1]  # Since it checks topics by order, it will return the first
    return False  # No topics of interest found from the given list


def reuters():

    """
    Asks which parts of the document we want to consider, then calls plot_graph passing documents and categories.
    """

    while True:
        print("""
Which parts of the documents would you like to consider?
1 - Title and body
2 - Body only
        """)
        input_value = int(raw_input())
        if input_value == 1:
            content = u'{title}\n\n{body}'
            break
        elif input_value == 2:
            content = u'{body}'
            break
        else:
            print('Unrecognised input.')

    iterator = reuters_utilities.stream_reuters_documents()  # Iterator over parsed Reuters files
    reuters_documents, reuters_topics = filter_documents(iterator, content, reuters_topics_list)

    plot_graph(reuters_documents, reuters_topics)


def twenty_newsgroups():

    """
    Asks which categories we want to retrieve the documents from and if we want any tag to be stripped away from the
    documents. Then it fetches all the 20newsgroups document of the selected category/categories and it passes documents
    and corresponding categories to the plot_graph function.
    """

    while True:
        print("""
Which document categories would you like to use?
1 - All categories
2 - comp.*
3 - rec.*
4 - sci.*
5 - talk.*
        """)
        input_value = int(raw_input())
        if input_value == 1:
            newsgroups_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                                     'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                                     'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                                     'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                                     'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                                     'talk.politics.misc', 'talk.religion.misc']
            break
        elif input_value == 2:
            newsgroups_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                                     'comp.sys.mac.hardware', 'comp.windows.x']
            break
        elif input_value == 3:
            newsgroups_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
            break
        elif input_value == 4:
            newsgroups_categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
            break
        elif input_value == 5:
            newsgroups_categories = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                                     'talk.religion.misc']
            break
        else:
            print('Unrecognised input.')

    while True:
        print("""
Would you like to remove any tag?
1 - Don't remove anything
2 - Remove headers
3 - Remove footers
4 - Remove quotes
5 - Remove headers, footers and quotes
        """)
        input_value = int(raw_input())
        if input_value == 1:
            remove_tags = ()
            break
        elif input_value == 2:
            remove_tags = ('headers')
            break
        elif input_value == 3:
            remove_tags = ('footers')
            break
        elif input_value == 4:
            remove_tags = ('quotes')
            break
        elif input_value == 5:
            remove_tags = ('headers', 'footers', 'quotes')
            break
        else:
            print('Unrecognised input.')

    newsgroups_documents = fetch_20newsgroups(subset='all', shuffle=True, random_state=42, remove=remove_tags,
                                              categories=newsgroups_categories)
    plot_graph(newsgroups_documents.data, newsgroups_documents.target)


def plot_graph(documents, categories):

    """
    First, transforms the set of documents in a sparse matrix of occurrences, then makes a matrix of tfidf. Finally,
    invokes plot_curve twice, each time with one of the two estimators, and shows the corresponding graphs.

    documents contains the set of documents we are using to train and test each estimator
    categories contains the corresponding categories of the documents we are using
    """

    x_countvect = CountVectorizer(stop_words='english').fit_transform(documents)  # Learns a vocabulary dictionary and
    # returns a list of word occurrences
    x_tfidf = TfidfTransformer().fit_transform(x_countvect)  # Since we need word frequencies instead of
    # word occurrences
    print('Number of documents: ' + str(x_tfidf.shape[0]))
    print('Dictionary size: ' + str(x_tfidf.shape[1]) + '\n')

    dots, k_fold = plot_preparation()

    start = time.time()
    plot_curve(MultinomialNB(), 'Multinomial Naive Bayes', x_tfidf, categories, k_fold, dots)
    print('Multinomial Naive Bayes execution time: ' + str(round(time.time() - start, 5)) + ' seconds.')

    start = time.time()
    plot_curve(BernoulliNB(), 'Bernoulli Naive Bayes', x_tfidf, categories, k_fold, dots)
    print('Bernoulli Naive Bayes execution time: ' + str(round(time.time() - start, 5)) + ' seconds.')

    plt.show()


def plot_curve(estimator, title, x_tfidf, categories, k_fold, dots):

    """
    Using learning_curve(), we get the results of the current estimator: the train sizes from 10% to 100% the size of
    the dataset, train scores and test scores of the estimator; also, performs k-fold cross-validation to obtain those
    values. Then, calculates mean and standard deviation of those values, used to plot the graph afterwards.

    estimator is one between MultinomialNB and BernuolliNB, used to get the learning curve
    title will be used as title of the graph
    x_tfidf is the sparse matrix of tfidf values
    categories contains the dataset categories corresponding to each document in the tfidf matrix
    k_fold is the number of k-fold to perform in cross-validation
    dots is the number of dots we want to show in our graph

    The function returns a graph (plot)
    """

    plt.figure()
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Number of training examples')
    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(estimator, x_tfidf, categories, cv=k_fold, n_jobs=4,
                                                            train_sizes=np.linspace(0.1, 1.0, dots))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    print_results(estimator, train_scores_mean, train_scores_std, True)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print_results(estimator, test_scores_mean, test_scores_std, False)

    plt.grid()  # Matplotlib method which turns on the axes grid

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='g')  # Matplotlib method which makes filled
    # polygons between two curves
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='m')

    plt.plot(train_sizes, train_scores_mean, '.-', color='g', label='Training score')
    plt.plot(train_sizes, test_scores_mean, '.-', color='m', label='Cross-validation score')

    plt.legend(loc='best')
    return plt


def print_results(estimator, score, deviation, train):

    """
    Prints scores and deviation on the console for better understanding of the results.

    estimator indicates which estimator we're currently printing the results of
    score is the list of scores returned by the learning_curve function
    deviation is the list of standard deviation values we have for each score returned by learning_curve
    train is a boolean indicating whether the score has been obtained in training (True) or testing (False)
    """

    title = ''

    if type(estimator) is MultinomialNB:
        title += 'Multinomial '
    else:
        title += 'Bernoulli '

    if train:
        title += 'train scores:'
    else:
        title += 'test scores:'

    print(title)

    print_string = ''
    for index in range(0, len(score)):
        print_string += str(round(score[index], 5))
        print_string += ' +/- ' + str(round(deviation[index], 5))
        print_string += '\n'

    print(print_string)


def plot_preparation():

    """
    Asks the user the number of dots he wants on the graph and the number of k-fold to use in the cross-validation
    process.

    The function returns the number of dots and the number of k-folds
    """

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
                print('Invalid k-fold number specified. Setting it by default to 3.')
                return dots, 3  # This must be done to avoid setting the k-fold value to an illegal value (it must be
                # either None or >= 2)
            break
        except ValueError:
            print('Please input a numerical value.')

    return dots, k_fold


if __name__ == '__main__':
    print_prompt = True
    while True:
        print("""
Comparison between Multinomial Naive Bayes and Bernoulli Naive Bayes implementations.
Which dataset would you like to use for this comparison?
Type 1 for Reuters, 2 for 20newsgroups or 3 to exit this script.
        """)

        input_value = int(raw_input())
        if input_value is 1:
            reuters()
            print('Reuters execution completed.')
        elif input_value is 2:
            twenty_newsgroups()
            print('20newsgroups execution completed.')
        elif input_value is 3:
            break
        else:
            print('Unrecognised input.')
