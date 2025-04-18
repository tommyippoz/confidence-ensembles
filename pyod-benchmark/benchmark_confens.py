# -*- coding: utf-8 -*-
"""Benchmark of all implemented algorithms
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
from time import time

import numpy
import pandas

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from confens.classifiers.ConfidenceEnsemble import ConfidenceEnsemble

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cd import CD
from pyod.models.copod import COPOD
from pyod.models.dif import DIF
from pyod.models.ecod import ECOD
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI  # 19S
from pyod.models.loda import LODA
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.sos import SOS

from pyod.models.alad import ALAD  # 40s
from pyod.models.anogan import AnoGAN  # 151s
from pyod.models.inne import INNE
from pyod.models.kpca import KPCA
from pyod.models.lscp import LSCP
from pyod.models.lunar import LUNAR
from pyod.models.mad import MAD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.rgraph import RGraph  # 271S
from pyod.models.rod import ROD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sod import SOD
from pyod.models.vae import VAE

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# TODO: add neural networks, LOCI, SOS, COF, SOD

RESULT_CSV = 'pyod_benc_result_withBase.csv'

# Define data file and read X and y
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

# define the number of iterations
n_ite = 1

df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc']
for alg in ['ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest', 'KNN', 'LOF',
            'MCD', 'OCSVM', 'PCA', 'AutoEncoder', 'CD', 'COPOD', 'DIF', 'ECOD',
            'GMM', 'KDE', 'LODA', 'QMCD', 'Sampling', 'SOS', 'ALAD', 'AnoGAN ',
            'INNE', 'KPCA', 'LMDD', 'LOCI', 'LSCP', 'LUNAR', 'MO_GAAL', 'RGraph', 'SO_GAAL', 'SOD', 'VAE']:
    df_columns.append('ConfBag10(' + alg + ')')
    df_columns.append('ConfBag20(' + alg + ')')
    df_columns.append(alg)
    df_columns.append('ConfBoost10(' + alg + ')')
    df_columns.append('ConfBoost10W(' + alg + ')')
    df_columns.append('ConfBoost5(' + alg + ')')
    df_columns.append('ConfBoost5W(' + alg + ')')

n_classifiers = len(df_columns) - 4

# initialize the container for saving the results
mcc_df = pd.DataFrame(columns=df_columns)
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

existing_exps = None
if os.path.exists(RESULT_CSV):
    existing_exps = pandas.read_csv(RESULT_CSV)
    existing_exps = existing_exps.loc[:, ['dataset', 'clf', 'iter']]
else:
    with open(RESULT_CSV, 'w') as f:
        f.write('dataset,clf,iter,duration,roc,mcc,pr@n,best_baselearner,best_baselearner_mcc\n')

for j in range(len(mat_file_list)):

    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join('data', mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    mcc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    mcc_mat = np.zeros([n_ite, n_classifiers])
    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        base_classifiers = {
            # 'Minimum Covariance Determinant (MCD)': MCD(
            #    contamination=outliers_fraction,
            #    random_state=random_state),
            'COPOD': COPOD(
                contamination=outliers_fraction),
            'Angle-based Outlier Detector (ABOD)': ABOD(
                contamination=outliers_fraction),
            'Cluster-based Local Outlier Factor': CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            'Feature Bagging': FeatureBagging(
                contamination=outliers_fraction,
                random_state=random_state),
            'Histogram-base Outlier Detection (HBOS)': HBOS(
                contamination=outliers_fraction),
            'Isolation Forest': IForest(
                contamination=outliers_fraction,
                random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(
                contamination=outliers_fraction),
            'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction),
            'One-class SVM (OCSVM)': OCSVM(
                contamination=outliers_fraction),
            'Principal Component Analysis (PCA)': PCA(
                contamination=outliers_fraction,
                random_state=random_state),
            'AutoEncoder': AutoEncoder(
                contamination=outliers_fraction),
            # 'CD': CD(
            #    contamination=outliers_fraction),
            # 'DIF': DIF(
            #    contamination=outliers_fraction),
            'ECOD': ECOD(
                contamination=outliers_fraction),
            'GMM': GMM(
                contamination=outliers_fraction),
            'KDE': KDE(
                contamination=outliers_fraction),

            'LODA': LODA(
                contamination=outliers_fraction),
            'QMCD': QMCD(
                contamination=outliers_fraction),
            'Sampling': Sampling(
                contamination=outliers_fraction),
            # 'SOS': SOS(
            #    contamination=outliers_fraction, ),
            # 'ALAD': ALAD(
            #     contamination=outliers_fraction),
            # 'AnoGAN':AnoGAN(
            #     contamination=outliers_fraction),
            'INNE': INNE(contamination=outliers_fraction),
            'KPCA': KPCA(contamination=outliers_fraction),
            # 'LMDD': LMDD(contamination=outliers_fraction),
            # 'LOCI': LOCI(contamination=outliers_fraction),
            'LUNAR': LUNAR(contamination=outliers_fraction),
            'MO_GAAL': MO_GAAL(contamination=outliers_fraction),
            # 'RGraph': RGraph(contamination=outliers_fraction),
            # 'SO_GAAL': SO_GAAL(contamination=outliers_fraction),
            # 'SOD': SOD(contamination=outliers_fraction),

        }

        classifiers = {}
        for clf_name in base_classifiers.keys():
            clf = base_classifiers[clf_name]
            classifiers['ConfBag10(' + clf_name + ')'] = ConfidenceBagging(clf=clf)
            classifiers['ConfBag20(' + clf_name + ')'] = ConfidenceBagging(clf=clf, n_base=20)
            classifiers[clf_name] = clf
            classifiers['ConfBoost10(' + clf_name + ')'] = ConfidenceBoosting(clf=clf, n_base=10)
            classifiers['ConfBoost10W(' + clf_name + ')'] = ConfidenceBoosting(clf=clf, n_base=10, weighted=True)
            classifiers['ConfBoost5(' + clf_name + ')'] = ConfidenceBoosting(clf=clf, n_base=5)
            classifiers['ConfBoost5W(' + clf_name + ')'] = ConfidenceBoosting(clf=clf, n_base=5, weighted=True)

        classifiers_indices = {}
        index = 0
        for clf_name in base_classifiers_indices.keys():
            classifiers_indices['ConfBag10(' + clf_name + ')'] = index + 1
            classifiers_indices['ConfBag20(' + clf_name + ')'] = index + 2
            classifiers_indices[clf_name] = index
            classifiers_indices['ConfBoost10(' + clf_name + ')'] = index + 3
            classifiers_indices['ConfBoost10W(' + clf_name + ')'] = index + 4
            classifiers_indices['ConfBoost5(' + clf_name + ')'] = index + 5
            classifiers_indices['ConfBoost5W(' + clf_name + ')'] = index + 6
            index = index + 7

        for clf_name, clf in classifiers.items():

            if existing_exps is not None and (((existing_exps['dataset'] == mat_file) &
                                               (existing_exps['iter'] == i) &
                                               (existing_exps['clf'] == clf_name)).any()):
                print('Skipping classifier %s, already in the results' % clf_name)
            else:
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
                      'execution time: {duration}s'.format(
                    mcc=mcc, clf_name=clf_name, roc=roc, prn=prn, duration=duration))

                if isinstance(clf, ConfidenceEnsemble):
                    y_pred, base_pred = clf.predict_proba(X_test_norm, get_base=True)
                    base_mcc = {}
                    for (base_k, base_p) in base_pred.items():
                        base_y = numpy.argmax(base_p, axis=1)
                        base_mcc[base_k] = matthews_corrcoef(y_test, base_y)
                    best_base = max(base_mcc, key=base_mcc.get)
                    best_base_mcc = base_mcc[best_base]
                else:
                    best_base = clf_name
                    best_base_mcc = mcc

                time_mat[i, classifiers_indices[clf_name]] = duration
                mcc_mat[i, classifiers_indices[clf_name]] = mcc
                roc_mat[i, classifiers_indices[clf_name]] = roc
                prn_mat[i, classifiers_indices[clf_name]] = prn

                with open(RESULT_CSV, 'a') as myhandle:
                    myhandle.write(mat_file + "," + clf_name + "," + str(i) + "," +
                                   str(duration) + "," + str(mcc) + "," + str(roc) + "," +
                                   str(prn) + "," + str(best_base) + "," + str(best_base_mcc) + "\n")

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    mcc_list = mcc_list + np.mean(mcc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(mcc_list).transpose()
    temp_df.columns = df_columns
    mcc_df = pd.concat([mcc_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    # Save the results for each run
    time_df.to_csv('time.csv', index=False, float_format='%.3f')
    mcc_df.to_csv('mcc.csv', index=False, float_format='%.3f')
    roc_df.to_csv('roc.csv', index=False, float_format='%.3f')
    prn_df.to_csv('prc.csv', index=False, float_format='%.3f')
