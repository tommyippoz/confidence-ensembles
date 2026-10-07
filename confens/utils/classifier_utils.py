import inspect
from collections.abc import Iterable

import numpy
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted, check_array


def top_level_package(obj) -> str | None:
    """
    Gets the string copntaining the top level package of an object
    :param obj:
    :return:
    """
    mod = getattr(type(obj), "__module__", None)
    return mod.split(".")[0] if mod else None

def is_pyod(clf):
    """
    True if the classifier is a PYOD object (unsupervised binary classifier)
    :param clf:
    :return:
    """
    return top_level_package(clf) == "pyod"

def predict_proba(clf, X, get_base:bool = False):
    """
    Function to predict probabilities of a classifier
    Needed to overcome issues in pyod's predict_proba
    :param get_base: Tue if predictions of base-learners have to be returned as well
    :param clf: the classifier to be used
    :param X: the test set
    :return:
    """
    if is_pyod(clf):
        return predict_uns_proba(clf, X)
    elif 'get_base' in inspect.getfullargspec(clf.predict_proba)[0]:
        return clf.predict_proba(X, get_base=get_base)
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
        if is_classifier(clf_object) or is_pyod(clf_object):
            clf_name = get_single_classifier_name(clf_object)
        elif isinstance(clf_object, Iterable):
            if len(clf_object) < 5:
                for clf_item in clf_object:
                    clf_name = clf_name + (get_single_classifier_name(clf_item) if is_classifier(clf_item) else "?") + "@"
                clf_name = clf_name[0:-1]
            else:
                clf_name = clf_name + str(len(clf_object)) + "_algs"
        else:
            clf_name = str(clf_object)
        if hasattr(clf_object, "base_estimator") and hasattr(clf_object, "n_estimators"):
            clf_name = clf_name + "(" + get_single_classifier_name(clf_object.base_estimator) + ";" \
                       + str(clf_object.n_estimators) + ")"
        if hasattr(clf_object, "estimators"):
            if len(clf_object.estimators) < 5:
                clf_name = clf_name + "(" + "@".join([get_single_classifier_name(clf) for clf in clf_object.estimators]) + ";" \
                       + str(len(clf_object.estimators)) + ")"
            else:
                clf_name = clf_name + "(" + str(len(clf_object.estimators)) + ")"
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
            for x in list(clf_object.named_steps.keys()):
                if is_classifier(clf_object[x]):
                    clf_name = get_single_classifier_name(clf_object[x])
    elif isinstance(clf_object, tuple):
        clf_name = str(clf_object[0])
        for x in clf_object:
            if is_classifier(x):
                clf_name = get_single_classifier_name(x)
    else:
        clf_name = clf_object.__class__.__name__
        if clf_name == 'Pipeline':
            for x in list(clf_object.named_steps.keys()):
                if is_classifier(clf_object[x]):
                    clf_name = get_single_classifier_name(clf_object[x])
    return clf_name


def predict_confidence(clf, X):
    """
    Method to compute the confidence in predictions of a classifier
    :param clf: the classifier
    :param X: the test set
    :return: array of confidence scores
    """
    c_conf = None
    if is_pyod(clf):
        y_proba = predict_proba(clf, X)
        c_conf = numpy.max(y_proba, axis=1)
    if is_classifier(clf):
        if hasattr(clf, 'predict_confidence') and callable(clf.predict_confidence):
            c_conf = clf.predict_confidence(X)
        else:
            y_proba = predict_proba(clf, X)
            c_conf = numpy.max(y_proba, axis=1)
    return c_conf

def expected_calibration_error(y_true: numpy.ndarray, y_proba: numpy.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (Guo et al., 2017; lower is better).

    Buckets samples by predicted confidence (max class probability) into
    ``n_bins`` equal-width bins on ``[0, 1]``, and reports the weighted
    average gap between confidence and accuracy inside each bin.
    """
    y_true = numpy.asarray(y_true).astype(int).ravel()

    confidences = numpy.max(y_proba, axis=1)
    predictions = numpy.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = numpy.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Closed upper edge on the last bin so confidence==1.0 is counted.
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        n_in = int(in_bin.sum())
        if n_in > 0:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(accuracies[in_bin].mean())
            ece += (n_in / n) * abs(avg_conf - avg_acc)

    return float(ece)