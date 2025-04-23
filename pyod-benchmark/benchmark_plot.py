# -*- coding: utf-8 -*-
"""Benchmark of all implemented algorithms
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import math
import os
import sys
from time import time

import matplotlib
import numpy
import pandas
from matplotlib import pyplot as plt

from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM

from pyod.models.sampling import Sampling

from pyod.models.inne import INNE

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# CSV file to save results in
RESULT_CSV = '20250423_pyod_bench.csv'

# Define data file and read X and y
# Same as pyod benchmark
mat_file_list = ['arrhythmia.mat',
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                 'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'shuttle.mat',
                 'vertebral.mat',
                 'vowels.mat',
                 'wbc.mat'
                 ]

# Same as PYOD benchmark
base_classifiers_indices = {
    'Minimum Covariance Determinant (MCD)': 7,
    'COPOD': 12,
    'Angle-based Outlier Detector (ABOD)': 0,
    'Cluster-based Local Outlier Factor': 1,
    'Feature Bagging': 2,
    'Histogram-base Outlier Detection (HBOS)': 3,
    'Isolation Forest': 4,
    'K Nearest Neighbors (KNN)': 5,
    'Local Outlier Factor (LOF)': 6,
    'One-class SVM (OCSVM)': 8,
    'Principal Component Analysis (PCA)': 9,
    'AutoEncoder': 10,
    'CD': 11,
    'DIF': 13,
    'ECOD': 14,
    'GMM': 15,
    'KDE': 16,
    'LODA': 17,
    'QMCD': 18,
    'Sampling': 19,
    'SOS': 20,
    'ALAD': 21,
    'AnoGAN': 22,
    'INNE': 23,
    'KPCA': 24,
    'LMDD': 25,
    'LOCI': 26,
    'LUNAR': 27,
    'MO_GAAL': 28,
    'RGraph': 29,
    'SO_GAAL': 30,
    'SOD': 31,
}

# ------------------------------------------------------------------------
# SUPPORT METHODS

def get_base_classifiers(outliers_fraction: float = 0.1) -> dict:
    """
    Returns the base classifiers to be used for each dataset
    """
    return {
        'CBLOF': CBLOF(
            n_clusters=10,
            contamination=outliers_fraction,
            check_estimator=False,
            random_state=random_state),
        'IForest': IForest(
            contamination=outliers_fraction,
            random_state=random_state),
        'Sampling': Sampling(
            contamination=outliers_fraction),
        'INNE': INNE(contamination=outliers_fraction),
        'Ens1': [
            CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            INNE(contamination=outliers_fraction),
            IForest(
                contamination=outliers_fraction,
                random_state=random_state)
        ],
        'Ens2': [
            CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            INNE(contamination=outliers_fraction),
            HBOS(contamination=outliers_fraction)
        ],
        'Ens3': [
            CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            INNE(contamination=outliers_fraction),
            OCSVM(contamination=outliers_fraction)
        ],
        'Ens4': [
            CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            INNE(contamination=outliers_fraction),
            OCSVM(contamination=outliers_fraction),
            HBOS(contamination=outliers_fraction),
            IForest(
                contamination=outliers_fraction,
                random_state=random_state)
        ]

    }


def roc_to_rank(my_dict: dict, clf_list: list = None) -> dict:
    """
    Returns the (average, std) rank of each classifier over datasets
    """
    for dataset_name in my_dict.keys():
        if clf_list is None:
            clf_list = list(my_dict[dataset_name].keys())
            break

    # Computing Ranks
    clf_rank = {}
    for c_name in clf_list:
        clf_rank[c_name] = []
    for dataset_name in my_dict.keys():
        data_list = [my_dict[dataset_name][c_name].item() for c_name in my_dict[dataset_name].keys()]
        ranks = [sorted(data_list, reverse=True).index(x) for x in data_list]
        for i in range(0, len(ranks)):
            clf_rank[clf_list[i]].append(ranks[i] + 1)

    # Averages Rank
    avg_rank = {}
    for clf_name in clf_list:
        avg_rank[clf_name] = {}
        avg_rank[clf_name]['avg'] = numpy.average(clf_rank[clf_name])
        avg_rank[clf_name]['std'] = numpy.std(clf_rank[clf_name])
    return avg_rank


def avg_duration(my_dict, clf_list: list = None) -> dict:
    """
    Returns the average duration of each classifier over datasets
    """
    for dataset_name in my_dict.keys():
        if clf_list is None:
            clf_list = my_dict[dataset_name].keys()
            break

    # Averages Rank
    avg_duration = {}
    for c_name in clf_list:
        avg_duration[c_name] = numpy.average([my_dict[d_name][c_name] for d_name in my_dict.keys()])
    return avg_duration


if __name__ == '__main__':

    existing_exps = None
    # Loads the results CSV file (if existing) to allow checking if some experiments were already run.
    # Otherwire, writes the header of the CSV and creates the CSV
    if os.path.exists(RESULT_CSV):
        existing_exps = pandas.read_csv(RESULT_CSV)
        existing_exps = existing_exps.loc[:, ['dataset', 'clf', 'roc', 'duration']]
    else:
        with open(RESULT_CSV, 'w') as f:
            f.write('dataset,clf,duration,roc,mcc,pr@n\n')

    # This is a dict of dicts that will contain ROC scores for each classifier and each dataset
    roc_dict = {}
    # This is a dict of dicts that will contain train+test time for each classifier and each dataset
    time_dict = {}

    # ------------------------------------------------------------------------
    # Iterates over datasets
    for j in range(len(mat_file_list)):

        mat_file = mat_file_list[j]
        mat = loadmat(os.path.join('data', mat_file))
        roc_dict[mat_file] = {}
        time_dict[mat_file] = {}
        print("\n\n\t Processing %s\n\n" % mat_file)

        X = mat['X']
        y = mat['y'].ravel()
        outliers_fraction = np.count_nonzero(y) / len(y)
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)

        random_state = np.random.RandomState(0)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        classifiers = {}
        base_classifiers = get_base_classifiers(outliers_fraction)
        for clf_name in base_classifiers.keys():
            clf = base_classifiers[clf_name]
            if not isinstance(clf, list):
                classifiers[clf_name] = clf
            classifiers['ConfBag(' + clf_name + ')'] = ConfidenceBagging(clf=clf, n_base=10)
            classifiers['ConfBoost(' + clf_name + ')'] = ConfidenceBoosting(clf=clf, n_base=20,
                                                                            relative_boost_thr=0.8)

        for clf_name, clf in classifiers.items():

            if existing_exps is not None and (((existing_exps['dataset'] == mat_file) &
                                               (existing_exps['clf'] == clf_name)).any()):
                print('Skipping classifier %s, already in the results' % clf_name)
                roc_dict[mat_file][clf_name] = existing_exps[(existing_exps['dataset'] == mat_file) &
                                                             (existing_exps['clf'] == clf_name)]['roc']
                time_dict[mat_file][clf_name] = existing_exps[(existing_exps['dataset'] == mat_file) &
                                                              (existing_exps['clf'] == clf_name)]['duration']
            elif not isinstance(clf, list):
                # Otherwise it runs the classifier (same as in PYOD benchmark)
                t0 = time()
                clf.fit(X_train_norm)
                test_scores = clf.decision_function(X_test_norm)

                # Handle NaN values in test_scores
                test_scores = np.nan_to_num(test_scores,
                                            nan=0.0,
                                            posinf=np.nanmax(test_scores),
                                            neginf=np.nanmin(test_scores))
                # Handle NaN values in y_test
                y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
                y_pred = clf.predict(X_test_norm)

                t1 = time()
                duration = round(t1 - t0, ndigits=4)
                roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
                prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
                mcc = round(matthews_corrcoef(y_test, y_pred), ndigits=4)

                print('{clf_name}\tMCC:{mcc} ROC:{roc}, precision @ rank n:{prn}, '
                      'execution time: {duration}s'.format(mcc=mcc, clf_name=clf_name,
                                                           roc=roc, prn=prn, duration=duration))
                roc_dict[mat_file][clf_name] = roc
                time_dict[mat_file][clf_name] = duration

                # Saves results in the CSV file
                with open(RESULT_CSV, 'a') as myhandle:
                    myhandle.write(mat_file + "," + clf_name + "," + str(duration) + "," + str(roc) + "," +
                                   str(mcc) + "," + str(prn) + "\n")

    # ------------------------------------------------------------------------
    # At the end, it plots contents of the roc_dict and duration_dict matrix
    clf_list = list(classifiers.keys())
    avg_dur = avg_duration(time_dict, clf_list)
    avg_roc_rank = roc_to_rank(roc_dict, clf_list)

    fig, ax = plt.subplots()
    x = [avg_roc_rank[a]['avg'] for a in clf_list]
    y = list(avg_dur.values())
    # Add Base
    plt.scatter([avg_roc_rank[a]['avg'] for a in clf_list if "ConfB" not in a],
                [avg_dur[a] for a in clf_list if "ConfB" not in a],
                marker='x', label='base')
    # Add ConfBag
    plt.scatter([avg_roc_rank[a]['avg'] for a in clf_list if "ConfBag" in a],
                [avg_dur[a] for a in clf_list if "ConfBag" in a],
                marker='^', label='ConfBag')
    # Add ConfBoost
    plt.scatter([avg_roc_rank[a]['avg'] for a in clf_list if "ConfBoost" in a],
                [avg_dur[a] for a in clf_list if "ConfBoost" in a],
                marker='o', label="ConfBoost")
    plt.xlabel("Average Rank (lower is better)")
    plt.ylabel("Average Time (lower is better)")
    plt.legend(loc='upper left')
    for i, txt in enumerate(clf_list):
        ax.annotate(txt, (x[i], y[i]), xytext=(x[i] - 0.2, y[i] + 0.6))

    plt.show()

