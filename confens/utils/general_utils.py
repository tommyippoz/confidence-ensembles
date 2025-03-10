import configparser
import os
import shutil
import time
from collections.abc import Iterable

import numpy
import numpy as np
from pyod.models.base import BaseDetector
from sklearn.base import is_classifier


def load_config(file_config):
    """
    Method to load configuration parameters from input file
    :param file_config: name of the config file
    :return: array with 4 items: [dataset files,
                                    classifiers,
                                    label name,
                                    max number of rows (if correctly specified, NaN otherwise)]
    """
    config = configparser.RawConfigParser()
    if os.path.isfile(file_config):
        config.read(file_config)
        config_file = dict(config.items('CONFIGURATION'))
    
        # Processing classifiers
        s_classifiers = config_file['supervised_classifiers']
        if ',' in s_classifiers:
            s_classifiers = [x.strip() for x in s_classifiers.split(',')]
        else:
            s_classifiers = [s_classifiers]
        s_classifiers = [x for x in s_classifiers if x]
        u_classifiers = config_file['unsupervised_classifiers']
        if ',' in u_classifiers:
            u_classifiers = [x.strip() for x in u_classifiers.split(',')]
        else:
            u_classifiers = [u_classifiers]
        u_classifiers = [x for x in u_classifiers if x]

        # Folders
        d_folder = config_file['datasets_folder']
        if not d_folder.endswith("/"):
            d_folder = d_folder + "/"
        s_folder = config_file['sprout_scores_folder']
        if not s_folder.endswith("/"):
            s_folder = s_folder + "/"
    
        # Processing paths
        path_string = config_file['datasets']
        if ',' in path_string:
            path_string = [x.strip() for x in path_string.split(',')]
        else:
            path_string = [path_string]
        datasets_path = []
        for file_string in path_string:
            if file_string in ["MNIST", "DIGITS", "FASHION-MNIST"]:
                datasets_path.append(file_string)
            else:
                file_path = os.path.join(d_folder, file_string)
                if os.path.isdir(file_path):
                    datasets_path.extend([os.path.join(file_path, f) for f in os.listdir(file_path) if
                                          os.path.isfile(os.path.join(file_path, f))])
                else:
                    datasets_path.append(file_path)
    
        # Processing limit to rows
        lim_rows = config_file['limit_rows']
        if not lim_rows.isdigit():
            lim_rows = np.nan
        else:
            lim_rows = int(lim_rows)

        return datasets_path, d_folder, s_folder, s_classifiers, u_classifiers, config_file['label_tabular'], lim_rows
    
    else:
        # Config File does not exist
        return ["DIGITS"], ["RF"], "", "no"


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def clean_name(file, prequel):
    """
    Method to get clean name of a file
    :param file: the original file path
    :return: the filename with no path and extension
    """
    if prequel in file:
        file = file.replace(prequel, "")
    if '.' in file:
        file = file.split('.')[0]
    if file.startswith("/"):
        file = file[1:]
    return file


def get_full_class_name(class_obj):
    return class_obj.__module__ + "." + class_obj.__qualname__


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


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
            y_proba = clf.predict_proba(X)
            c_conf = numpy.max(y_proba, axis=1)
    return c_conf
