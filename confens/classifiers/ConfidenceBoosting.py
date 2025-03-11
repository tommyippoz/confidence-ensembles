import copy

import numpy

from confens.classifiers.ConfidenceEnsemble import ConfidenceEnsemble
from confens.utils.classifier_utils import get_classifier_name, predict_confidence


def define_conf_thr(confs, target: float = None, delta: float = 0.01) -> float:
    """
    Method for finding a confidence threshold based on the expected contamination (iterative)
    :param confs: confidences to find threshold of
    :param target: the quantity to be used as reference for gettng to the threshold
    :param delta: the tolerance to stop recursion
    :return: a float value to be used as threshold for updating weights in boosting
    """
    target_thr = target
    left_bound = min(confs)
    right_bound = max(confs)
    c_thr = (right_bound + left_bound) / 2
    a = numpy.average(confs < 0.6)
    b = numpy.average(confs < 0.9)
    actual_thr = numpy.average(confs < c_thr)
    while abs(actual_thr - target_thr) > delta and abs(right_bound - left_bound) > 0.001:
        if actual_thr < target_thr:
            left_bound = c_thr
            c_thr = (c_thr + right_bound) / 2
        else:
            right_bound = c_thr
            c_thr = (c_thr + left_bound) / 2
        actual_thr = numpy.average(confs < c_thr)
    return c_thr


class ConfidenceBoosting(ConfidenceEnsemble):
    """
    Class for creating Confidence Boosting ensembles
    """

    def __init__(self, clf, n_base: int = 10, learning_rate: float = None,
                 sampling_ratio: float = 0.5, relative_boost_thr: float = 0.8, static_boost_thr: float = None,
                 conf_thr: float = None, perc_decisors: float = None,
                 n_decisors: int = None, weighted: bool = False):
        """
        Constructor
        :param clf: the algorithm to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param learning_rate: learning rate for updating dataset weights
        :param sampling_ratio: percentage of the dataset to be used at each iteration
        :param boost_thr: threshold of acceptance for confidence scores. It is the percentile of confidence scores of the base estimator that are considered "confident enough"
        :param static_boost_thr: static threshold of acceptance for confidence scores. Lower confidence means untrustable result
        :param conf_thr: float value for confidence threshold
        :param perc_decisors: percentage of base learners to be used for prediction
        :param n_decisors: number of base learners to be used for prediction
        :param weighted: True if prediction has to be computed as a weighted sum of probabilities
        """
        super().__init__(clf, n_base, conf_thr, perc_decisors, n_decisors, weighted)
        self.proba_thr = None

        # Boosting thresholds
        self.relative_boost_thr = relative_boost_thr
        self.static_boost_thr = static_boost_thr
        self.actual_boost_thr_list = None

        # Other ConfBoost parameters
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 2
        if sampling_ratio is not None:
            self.sampling_ratio = sampling_ratio
        else:
            self.sampling_ratio = 1 / n_base ** (1 / 2)

    def fit_ensemble(self, X, y=None):
        """
        Training function for the confidence boosting ensemble
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        train_n = len(X)
        samples_n = int(train_n * self.sampling_ratio)
        weights = numpy.full(train_n, 1 / train_n)

        # If static boosting treshold provided, we use it. Otherwise, we use the relative, computed afterwards
        if self.static_boost_thr is not None and 0 < self.static_boost_thr < 1:
            self.actual_boost_thr_list = [self.static_boost_thr for _ in self.clf_list]
        else:
            self.actual_boost_thr_list = [None for _ in self.clf_list]
            self.relative_boost_thr = 0.8 if self.relative_boost_thr is None else self.relative_boost_thr
        for learner_index in range(0, self.n_base):
            # Draw samples
            sample_x, sample_y = self.draw_samples(X, y, samples_n, weights)
            # Train learner
            learner = copy.deepcopy(self.clf_list[learner_index % len(self.clf_list)])
            learner.fit(sample_x, sample_y)
            if hasattr(learner, "X_"):
                learner.X_ = None
            if hasattr(learner, "y_"):
                learner.y_ = None

            y_conf = predict_confidence(learner, X)
            # Computing actual boosting thresholds if not already computed (only first time for each base estimator)
            if self.actual_boost_thr_list[learner_index % len(self.clf_list)] is None:
                actual_thr = y_conf[int(self.relative_boost_thr*len(y_conf))] if y_conf is not None else 0.8
                self.actual_boost_thr_list[learner_index % len(self.clf_list)] = actual_thr

            p_thr = define_conf_thr(target=self.actual_boost_thr_list[learner_index % len(self.clf_list)],
                                    confs=y_conf)
            self.estimators_.append(learner)
            # Update Weights
            update_flag = numpy.where(y_conf >= p_thr, 0, 1)
            weights = weights * (1 + self.learning_rate * update_flag)
            weights = weights / sum(weights)

    def classifier_name(self):
        """
        Gets classifier name as string
        :return: the classifier name
        """
        clf_name = get_classifier_name(self.clf)
        if self.weighted:
            return "ConfidenceBoosterWeighted(" + str(clf_name) + "-" + \
                   str(self.n_base) + "-" + str(self.relative_boost_thr) + "-" + str(self.static_boost_thr) + "-" + \
                   str(self.learning_rate) + "-" + str(self.sampling_ratio) + "-" + \
                   str(self.conf_thr) + "-" + str(self.perc_decisors) + "-" + str(self.n_decisors) + ")"
        else:
            return "ConfidenceBooster(" + str(clf_name) + "-" + \
                   str(self.n_base) + "-" + str(self.relative_boost_thr) + "-" + str(self.static_boost_thr) + "-" + \
                   str(self.learning_rate) + "-" + str(self.sampling_ratio) + "-" + \
                   str(self.conf_thr) + "-" + str(self.perc_decisors) + "-" + str(self.n_decisors) + ")"
