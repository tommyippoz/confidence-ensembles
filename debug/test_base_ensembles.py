# Support libs
import copy
import os
import random
import time

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
# Used to save a classifier and measure its size in KB
from joblib import dump
from logitboost import LogitBoost
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Scikit-Learn algorithms
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Name of the folder in which look for tabular (CSV) datasets
from xgboost import XGBClassifier

from confens.classifiers.Classifier import XGB, UnsupervisedClassifier
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
# The PYOD library contains implementations of unsupervised classifiers.
# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------
from confens.metrics.EnsembleMetric import SharedFaultMetric, DisagreementMetric

CSV_FOLDER = "input_folder/test"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "test_base_ensembles.csv"
# Percantage of test data wrt train data
TT_SPLIT = 0.5
# True if debug information needs to be shown
VERBOSE = True
# True if we want to conduct anomaly detection.
# This transforms multi-class labels into binary labels (rule: normal class vs others)
BINARIZE = True

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# --------- SUPPORT FUNCTIONS ---------------


def current_milli_time():
    """
    gets the current time in ms
    :return: a long int
    """
    return round(time.time() * 1000)


def get_clf_name(classifier):
    clf_name = classifier.classifier_name() if hasattr(classifier,
                                                       'classifier_name') else classifier.__class__.__name__
    if clf_name == 'Pipeline':
        keys = list(classifier.named_steps.keys())
        clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()

    return clf_name

def get_ensemble_learners(cont_perc):
    """
    Function to get a learner to use, given its string tag
    :param base_clf: the base estimator of the ensemble
    :param cont_perc: percentage of anomalies in the training set, required for unsupervised classifiers from PYOD
    :return: the list of classifiers to be trained
    """
    e_l = []
    # Others
    contamination = cont_perc if cont_perc < 0.5 else 0.5
    for n_base in [5, 10, 20, 50, 100]:
        e_l.append(XGBClassifier(n_estimators=n_base))
        e_l.append(RandomForestClassifier(n_estimators=n_base))
        e_l.append(LogitBoost(n_estimators=n_base))
        e_l.append(ExtraTreesClassifier(n_estimators=n_base))
        e_l.append(IForest(n_estimators=n_base, contamination=contamination))
        e_l.append(INNE(n_estimators=n_base, contamination=contamination))

    return e_l


# ----------------------- MAIN ROUTINE ---------------------


if __name__ == '__main__':

    existing_exps = None
    if os.path.exists(SCORES_FILE):
        existing_exps = pandas.read_csv(SCORES_FILE)
        existing_exps = existing_exps.loc[:, ['dataset_tag', 'clf_name', 'n_est']]
    else:
        with open(SCORES_FILE, 'w') as f:
            f.write("dataset_tag,clf_name,n_est,binary,tt_split,acc,misc,mcc,ok_conf,misc_conf,time,model_size,disagreement,sharedfault\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):

            # if file is a CSV, it is assumed to be a dataset to be processed
            df = pandas.read_csv(full_name, sep=",")
            df = df.sample(frac=1.0)
            if len(df.index) > 80000:
                df = df.iloc[:80000, :]
            if VERBOSE:
                print("\n------------ DATASET INFO -----------------")
                print("Data Points in Dataset '%s': %d" % (dataset_file, len(df.index)))
                print("Features in Dataset: " + str(len(df.columns)))

            # Filling NaN and Handling (Removing) constant features
            df = df.fillna(0)
            df = df.loc[:, df.nunique() > 1]
            if VERBOSE:
                print("Features in Dataframe after removing constant ones: " + str(len(df.columns)))

            features_no_cat = df.select_dtypes(exclude=['object']).columns
            if VERBOSE:
                print("Features in Dataframe after non-numeric ones (including label): " + str(len(features_no_cat)))

            # Binarize if needed (for anomaly detection you need a 2-class problem, requires one of the classes to have NORMAL_TAG)
            normal_perc = None
            y = df[LABEL_NAME].to_numpy()
            if BINARIZE:
                y = numpy.where(df[LABEL_NAME] == "normal", 0, 1)
                if VERBOSE:
                    normal_frame = df.loc[df[LABEL_NAME] == NORMAL_TAG]
                    normal_perc = len(normal_frame.index) / len(df.index)
                    print("Normal data: " + str(len(normal_frame.index)) + " items (" +
                          "{:.3f}".format(100.0 * normal_perc) + "%)")
            elif VERBOSE:
                print("Dataset contains %d Classes" % len(numpy.unique(y)))

            # Set up train test split excluding categorical values that some algorithms cannot handle
            # 1-Hot-Encoding or other approaches may be used instead of removing
            x_no_cat = df.select_dtypes(exclude=['object']).to_numpy()
            x_train, x_test, y_train, y_test = ms.train_test_split(x_no_cat, y, test_size=TT_SPLIT, shuffle=True)

            if VERBOSE:
                print('-------------------- CLASSIFIERS -----------------------')

            # Loop for training and testing each learner specified by LEARNER_TAGS
            i = 1
            contamination = 1 - normal_perc if normal_perc is not None else None
            base_learners = get_ensemble_learners(contamination)
            n_clfs = len(base_learners)
            for ens_clf in base_learners:

                classifier = copy.deepcopy(ens_clf)
                # Getting classifier Name
                clf_name = get_clf_name(classifier)
                n_bl = 1
                if hasattr(classifier, "n_estimators"):
                    n_bl = classifier.n_estimators
                elif isinstance(classifier, ConfidenceBagging) or isinstance(classifier, ConfidenceBoosting):
                    n_bl = classifier.n_base

                if existing_exps is not None and (((existing_exps['dataset_tag'] == full_name) &
                                                   (existing_exps['clf_name'] == clf_name) &
                                                    (existing_exps['n_est'] == n_bl)).any()):
                    print('%d/%d Skipping classifier %s, already in the results' % (i, n_clfs, clf_name))

                else:
                    # Training the algorithm to get a model
                    start_time = current_milli_time()
                    classifier.fit(x_train, y_train)

                    # Quantifying size of the model
                    dump(classifier, "ens_dump.bin", compress=9)
                    size = os.stat("ens_dump.bin").st_size
                    os.remove("ens_dump.bin")

                    # Computing metrics
                    y_pred = classifier.predict(x_test)
                    if hasattr(classifier, 'predict_confidence') and callable(classifier.predict_confidence):
                        y_conf = classifier.predict_confidence(x_test)
                    else:
                        y_proba = classifier.predict_proba(x_test)
                        y_conf = numpy.max(y_proba, axis=1)
                    conf_ok = y_conf[numpy.where(y_pred == y_test)[0]]
                    conf_ok = [0.5] if len(conf_ok) == 0 else conf_ok
                    conf_ok_metrics = [numpy.min(conf_ok), numpy.median(conf_ok), numpy.average(conf_ok),
                                       numpy.max(conf_ok)]
                    conf_misc = y_conf[numpy.where(y_pred != y_test)[0]]
                    conf_misc = [0.5] if len(conf_misc) == 0 else conf_misc
                    conf_misc_metrics = [numpy.min(conf_misc), numpy.median(conf_misc), numpy.average(conf_misc),
                                         numpy.max(conf_misc)]

                    acc = metrics.accuracy_score(y_test, y_pred)
                    misc = int((1 - acc) * len(y_test))
                    mcc = abs(metrics.matthews_corrcoef(y_test, y_pred))

                    if BINARIZE:
                        # Prints metrics for binary classification + train time and model size
                        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
                        print('%d/%d %s\t-> TP: %d, TN: %d, FP: %d, FN: %d, Accuracy: %.3f, MCC: %.3f, Conf Diff: %.3f - train time: %d ms'
                              ' - model size: %.3f KB' %
                              (i, n_clfs, clf_name, tp, tn, fp, fn, acc, mcc,
                               (conf_ok_metrics[2] - conf_misc_metrics[2]),
                               current_milli_time() - start_time, size / 1000.0))
                    else:
                        # Prints just accuracy for multi-class classification problems, no confusion matrix
                        print('%d/%d %s\t-> Accuracy: %.3f, Conf Diff: %.3f - train time: %d ms - model size: %.3f KB'
                              % (i, n_clfs, clf_name, acc, (conf_ok_metrics[2] - conf_misc_metrics[2]),
                                 current_milli_time() - start_time, size / 1000.0))

                    # Updates CSV file form metrics of experiment
                    with open(SCORES_FILE, "a") as myfile:
                        # Prints result of experiment in CSV file
                        myfile.write(full_name + "," + clf_name + "," + str(n_bl) + "," + str(BINARIZE) + "," +
                                     str(TT_SPLIT) + ',' + str(acc) + "," + str(misc) + "," + str(mcc) + "," +
                                     ";".join(["{:.4f}".format(met) for met in conf_ok_metrics]) + "," +
                                     ";".join(["{:.4f}".format(met) for met in conf_misc_metrics]) + "," +
                                     str(current_milli_time() - start_time) + "," + str(size) + "\n")


                classifier = None
                i += 1
