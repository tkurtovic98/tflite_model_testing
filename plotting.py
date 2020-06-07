import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np


def classification_graph(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f)

    classification_counter = res['counts']

    labels = [str(class_name).upper() for class_name in classification_counter.keys()]

    values = list(classification_counter.values())
    true_class = []
    false_class = []

    for val in values:
        true_class.append(val.true)
        false_class.append(val.false)

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, true_class, width, label='Točno klasificirano')
    rects2 = ax.bar(x + width / 2, false_class, width, label='Netočno klasificirano')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Broj klasifikacija')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def auto_label(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    auto_label(rects1)
    auto_label(rects2)

    fig.tight_layout()

    plt.show()


def skikit():
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    import numpy as np

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # # Add noisy features
    random_state = np.random.RandomState(0)

    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    #
    # # Limit to the two first classes, and split into training and test
    # X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
    #                                                     test_size=.5,
    #                                                     random_state=random_state)
    #
    # # Create a simple classifier
    # classifier = svm.LinearSVC(random_state=random_state)
    # classifier.fit(X_train, y_train)
    # y_score = classifier.decision_function(X_test)

    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]

    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                        random_state=random_state)

    # We use OneVsRestClassifier for multi-label prediction
    from sklearn.multiclass import OneVsRestClassifier

    # Run classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
    classifier.fit(X_train, Y_train)
    y_score = classifier.decision_function(X_test)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

    smt = 5


# classification_graph('results.pickle')
classification_graph('results_0.75.pickle')
