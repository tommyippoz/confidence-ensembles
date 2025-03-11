from collections.abc import Iterable

import numpy
from pyod.models.base import BaseDetector
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted, check_array


def predict_proba(clf, X):
    """
    Function to predict probabilities of a classifier
    Needed to overcome issues in pyod's predict_proba
    :param clf:
    :param X:
    :return:
    """
    if isinstance(clf, BaseDetector):
        return predict_uns_proba(clf, X)
    else:
        return clf.predict_proba(X)

def predict_uns_proba(uns_clf, X):
    """
    Method to compute probabilities of predicted classes.
    It has to e overridden since PYOD's implementation of predict_proba is wrong
    :return: array of probabilities for each classes
    """

    # Check if fit has been called
    check_is_fitted(uns_clf)
    X = check_array(X)
    probs = numpy.zeros((X.shape[0], 2))
    pred_score = uns_clf.decision_function(X)
    if numpy.isfinite(pred_score).all():
        if isinstance(uns_clf.contamination, (float, int)) and numpy.isfinite(uns_clf.threshold_):
            pred_thr = pred_score - uns_clf.threshold_
        else:
            pred_thr = pred_score
        min_pt = min(pred_thr)
        max_pt = max(pred_thr)
        anomaly = pred_thr > 0
        cont = numpy.asarray([pred_thr[i] / max_pt if anomaly[i] else (pred_thr[i] / min_pt if min_pt != 0 else 0.2)
                              for i in range(0, len(pred_thr))])
        probs[:, 0] = 0.5 + cont / 2
        probs[:, 1] = 1 - probs[:, 0]
        probs[anomaly, 0], probs[anomaly, 1] = probs[anomaly, 1], probs[anomaly, 0]
    else:
        probs[:, 0] = 0.999
        probs[:, 1] = 0.001
    return probs

def get_classifier_name(clf_object):
    """
    Gets a string representing the classifier name
    :param clf_object: the object meant to be a classifier
    :return: a string
    """
    clf_name = ""
    if clf_object is not None:
        if is_classifier(clf_object) or isinstance(clf_object, BaseDetector):
            clf_name = get_single_classifier_name(clf_object)
        elif isinstance(clf_object, Iterable):
            for clf_item in clf_object:
                clf_name = clf_name + (get_single_classifier_name(clf_item) if is_classifier(clf_item) else "?") + "@"
            clf_name = clf_name[0:-1]
        else:
            clf_name = str(clf_object)
    return clf_name


def get_single_classifier_name(clf_object):
    """
    Gets a string representing the classifier name, assuming the object contains a single classifier
    :param clf_object: the object meant to be a classifier
    :return: a string
    """
    if hasattr(clf_object, "classifier_name") and callable(clf_object.classifier_name):
        clf_name = clf_object.classifier_name()
        if clf_name == 'Pipeline':
            keys = list(clf_object.named_steps.keys())
            clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()
    else:
        clf_name = clf_object.__class__.__name__
    return clf_name


def predict_confidence(clf, X):
    """
    Method to compute the confidence in predictions of a classifier
    :param clf: the classifier
    :param X: the test set
    :return: array of confidence scores
    """
    c_conf = None
    if is_classifier(clf) or isinstance(clf, BaseDetector):
        if hasattr(clf, 'predict_confidence') and callable(clf.predict_confidence):
            c_conf = clf.predict_confidence(X)
        else:
            y_proba = predict_proba(clf, X)
            c_conf = numpy.max(y_proba, axis=1)
    return c_conf