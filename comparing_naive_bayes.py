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

# Most frequent topics/IDs found on "Information Retrieval Technology" and from McCallum-Nigan paper, ordered from the
# least frequent to the most frequent
reuters_topics_list = [('corn', 10), ('wheat', 9), ('ship', 8), ('interest', 7), ('trade', 6), ('crude', 5),
                       ('grain', 4), ('money-fx', 3), ('acq', 2), ('earn', 1)]


def get_documents_batch(documents_iterator, content, topic_list):
    data = [(content.format(**doc), check_topics(doc, topic_list))  # Makes a list of tuples (document text, topic id)
            for doc in itertools.islice(documents_iterator, 21578)  # Splits documents, returning the first 21578 items
            if check_topics(doc, topic_list)]  # Removes documents with False (which do not belong to any
    # of the above topic)
    if not len(data):  # In case there are no documents with the selected topics...
        return np.asarray([], dtype=int), np.asarray([], dtype=int)  # ...it returns an empty list
    x_test, topics = zip(*data)  # Makes two list out of a list of tuples. x_test contains the documents and y contains
    # the topic which they belong to
    return x_test, np.asarray(topics, dtype=int)


def check_topics(doc, topic_list):  # Verifies if doc contains at least one of the chosen topics
    for topic in topic_list:
        if topic[0] in doc['topics']:
            return topic[1]  # Since it checks topics by order, it will return the first
    return False  # No topics of interest found from the given list


def reuters():
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
    reuters_train, reuters_topics = get_documents_batch(iterator, content, reuters_topics_list)

    plot_graph(reuters_train, reuters_topics)


def twenty_newsgroups():
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

    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=remove_tags,
                                      categories=newsgroups_categories)
    plot_graph(twenty_train.data, twenty_train.target)  # We pass documents first, then all categories they belong to


def plot_graph(documents, categories):
    x_train_countvect = CountVectorizer().fit_transform(documents)  # Learns a vocabulary dictionary and returns a list
    # of word occurrences
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_countvect)  # Since we need word frequencies instead of
    # word occurrences
    print('(N_samples, N_features): ', x_train_tfidf.shape, )

    dots, k_fold = plot_preparation()

    start = time.time()
    plot_curve(MultinomialNB(), 'Multinomial Naive Bayes', x_train_tfidf, categories, k_fold, dots)
    print('Multinomial Naive Bayes execution time: ' + str(round(time.time() - start, 5)) + ' seconds.')

    start = time.time()
    plot_curve(BernoulliNB(), 'Bernoulli Naive Bayes', x_train_tfidf, categories, k_fold, dots)
    print('Bernoulli Naive Bayes execution time: ' + str(round(time.time() - start, 5)) + ' seconds.')

    plt.show()


def plot_curve(estimator, title, X, y, k_fold, dots):
    plt.figure()
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Number of training examples')
    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=k_fold, n_jobs=2,
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

    plt.plot(train_sizes, train_scores_mean, '.-', color='g', label='Training score')  # Matplotlib method which plots
    # lines to the axes.
    plt.plot(train_sizes, test_scores_mean, '.-', color='m', label='Cross-validation score')

    plt.legend(loc='best')
    return plt


def print_results(estimator, score, deviation, train):
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
                # either None or >= 2
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
