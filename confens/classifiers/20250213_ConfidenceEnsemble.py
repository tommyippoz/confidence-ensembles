import copy
from collections.abc import Iterable

import numpy
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels

from confens.classifiers.Classifier import Classifier
from confens.utils.general_utils import predict_confidence


def define_proba_thr(probs, target: float = None, delta: float = 0.01) -> float:
    """
    Method for finding a confidence threshold based on the expected contamination (iterative)
    :param probs: probabilities to find threshold of
    :param target: the quantity to be used as reference for gettng to the threshold
    :param delta: the tolerance to stop recursion
    :return: a float value to be used as threshold for updating weights in boosting
    """
    target_cont = target
    p_thr = 0.5
    left_bound = 0.5
    right_bound = 1
    actual_cont = numpy.average(probs[:, 0] < p_thr)
    while abs(actual_cont - target_cont) > delta and abs(right_bound - left_bound) > 0.01:
        if actual_cont < target_cont:
            left_bound = p_thr
            p_thr = (p_thr + right_bound) / 2
        else:
            right_bound = p_thr
            p_thr = (p_thr + left_bound) / 2
        actual_cont = numpy.average(probs[:, 0] < p_thr)
    return p_thr


class ConfidenceEnsemble(Classifier):
    """
    Class for creating confidence ensembles
    """

    def __init__(self, clf, n_base: int = 10, conf_thr: float = None, perc_decisors: float = None,
                 n_decisors: int = None, weighted: bool = False):
        """
        Constructor
        :param clf: the algorithm(s) to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param conf_thr: float value for confidence threshold
        :param perc_decisors: percentage of base learners to be used for prediction
        :param n_decisors: number of base learners to be used for prediction
        :param weighted: True if prediction has to be computed as a weighted sum of probabilities
        """
        super().__init__(clf)
        self.clf_list = []
        if is_classifier(clf):
            self.clf_list = [clf]
        elif isinstance(clf, Iterable):
            self.clf_list = []
            for clf_item in clf:
                if is_classifier(clf_item):
                    self.clf_list.append(clf_item)
                else:
                    print("Cant recognize object s a classifier")
            if len(self.clf_list) == 0:
                self.clf_list = [RandomForestClassifier(n_estimators=10)]
                print("clf is not a classifier. Using a 10-tree Random Forest as Base estimator")
        else:
            self.clf_list = [RandomForestClassifier(n_estimators=10)]
            print("clf is not a classifier. Using a 10-tree Random Forest as Base estimator")
        self.weighted = weighted
        if n_base > 1:
            self.n_base = n_base
        else:
            print("Ensembles have to be at least 2")
            self.n_base = 10

        self.proba_thr = None
        self.conf_thr = conf_thr
        self.contamination = clf.contamination if hasattr(clf, 'contamination') else None

        self.perc_decisors = perc_decisors
        if perc_decisors is not None and 0 < perc_decisors <= 1:
            if n_decisors is not None and 0 < n_decisors <= self.n_base:
                print('Both perc_decisors and n_decisors are specified, prioritizing perc_decisors')
            self.n_decisors = int(self.n_base * perc_decisors) if int(self.n_base * perc_decisors) > 0 else 1
        elif n_decisors is not None and 0 < n_decisors <= self.n_base:
            self.n_decisors = n_decisors
        elif self.conf_thr is None:
            self.n_decisors = self.n_base
        else:
            self.n_decisors = None
        self.estimators_ = []

    def fit(self, X, y=None):
        """
        Training function for the classifier
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        self.classes_ = unique_labels(y) if y is not None else [0, 1]
        self.fit_ensemble(X, y)
        self.proba_thr = define_proba_thr(target=self.contamination, probs=self.predict_proba(X)) \
            if self.contamination is not None else 0.5

        # Compliance with SKLEARN and PYOD
        self.X_ = X[[0, 1], :]
        self.y_ = y
        self.feature_importances_ = self.compute_feature_importances()

    def fit_ensemble(self, X, y=None):
        """
        Training function for the confidence ensemble: TO BE OVERRIDDEN
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        pass

    def predict_proba(self, X):
        # Scoring probabilities and confidence
        proba_array = []
        conf_array = []
        for i in range(0, self.n_base):
            if hasattr(self, "feature_sets"):
                # ConfBag, each estimator uses a subset of features
                predictions = self.estimators_[i].predict_proba(X[:, self.feature_sets[i]])
            else:
                # ConfBoost, all estimators use all features
                predictions = self.estimators_[i].predict_proba(X)
            proba_array.append(predictions)
            conf_array.append(numpy.max(predictions, axis=1))
        # 3d matrix (clf, row, probability for class)
        proba_array = numpy.asarray(proba_array)
        # 2dim matrix (clf, confidence for row)
        conf_array = numpy.asarray(conf_array)

        # Compute final probabilities
        proba = numpy.zeros(proba_array[0].shape)
        # Adjust probabilities with confidence if weighted
        if self.weighted:
            for i in range(0, X.shape[0]):
                proba_array[:, i, :] = (proba_array[:, i, :].T * conf_array[:, i]).T

        if self.n_decisors is not None:
            # Option 1: either n_decisors or perc_decisors is set, or conf_thr is None
            if self.n_decisors > self.n_base or self.n_decisors <= 0:
                # Checks for strange n_decisors values
                self.n_decisors = self.n_base
            if self.n_decisors == self.n_base:
                # Easy case, all contribute, no need to additional computation
                proba = numpy.average(proba_array, axis=0)
                proba = proba / numpy.sum(proba, axis=1).reshape(-1, 1)
            else:
                # Here it requires sorting confidences for understanding the "best" base-learners
                conf_array = conf_array.transpose()
                all_conf = -numpy.sort(-conf_array, axis=1)
                conf_thrs = all_conf[:, self.n_decisors - 1]
                for i in range(0, X.shape[0]):
                    proba[i] = numpy.average(proba_array[numpy.where(conf_array[i] >= conf_thrs[i]), i, :], axis=1)
        else:
            # Option 2: conf_thr is not None, neither n_decisors nor perc_decisors are set
            # Thus, base-learners contribute if they are confident at least 'conf_thr'
            conf_array = conf_array.transpose()
            for i in range(0, X.shape[0]):
                proba[i] = numpy.average(proba_array[numpy.where(conf_array[i] >= self.conf_thr), i, :], axis=1)

        # Final averaged Result
        return proba

    def draw_samples(self, X, y, samples_n: int, weights = None):
        """
        Returns samples of a labeled set (X, y), y may be None
        :param weights: may be none, weights for sampling data (needed mostly for ConfBoost)
        :param X: the set to sample
        :param y: the labels to sample
        :param samples_n: numer of samples
        :return: a subset of (X, y)
        """
        if weights is None:
            indexes = numpy.random.choice(X.shape[0], samples_n, replace=False, p=None)
        else:
            indexes = numpy.random.choice(len(weights), samples_n, replace=False, p=weights)
        sample_x = numpy.asarray(X[indexes, :])
        # If data is labeled we also have to refine labels
        if y is not None and hasattr(self, 'classes_') and self.classes_ is not None and len(self.classes_) > 1:
            sample_y = y[indexes]
            sample_labels = unique_labels(sample_y)
            missing_labels = [item for item in self.classes_ if item not in sample_labels]
            # And make sure that there is at least a sample for each class of the problem
            if missing_labels is not None and len(missing_labels) > 0:
                # For each missing class
                for missing_class in missing_labels:
                    miss_class_indexes = numpy.asarray(numpy.where(y == missing_class)[0])
                    new_sampled_index = numpy.random.choice(miss_class_indexes, None, replace=False, p=None)
                    X_missing_class = X[new_sampled_index, :]
                    sample_x = numpy.append(sample_x, [X_missing_class], axis=0)
                    sample_y = numpy.append(sample_y, missing_class)
        else:
            sample_y = None
        return sample_x, sample_y

    def predict_confidence(self, X, y_proba = None):
        """
        Method to compute the confidence in predictions of a classifier
        :param X: the test set
        :return: array of confidence scores
        """
        conf = numpy.zeros(X.shape[0])
        for clf in self.estimators_:
            conf += predict_confidence(clf, X)
        return conf / self.n_base

    def get_feature_importances(self):
        """
        Placeholder, to be implemented if possible
        :return: feature importances (to be tested)
        """
        fi = []
        for clf in self.estimators_:
            c_fi = clf.get_feature_importances()
            fi.append(c_fi)
        fi = numpy.asarray(fi)
        return numpy.average(fi, axis=1)

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :param X: the test set
        :return: array of predicted class
        """
        proba = self.predict_proba(X)
        if self.is_unsupervised():
            return 1 * (proba[:, 0] < self.proba_thr)
        else:
            return self.classes_[numpy.argmax(proba, axis=1)]
