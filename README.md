# Confidence Ensembles

Python Frameworkthat implements confidence ensembles: ConfBag and ConfBoost

## Aim/Concept of the Project

One of the biggest ICT challenges of the current era is to solve classification problems by designing and training Machine Learning (ML) algorithms that assign a discrete label to input data with perfect accuracy i.e., no misclassifications. These classifiers can be used for solving a wide variety of tasks processing either structured tabular data or unstructured inputs (e.g., images) for binary or multi-class classification. In real-world applications, the most common data type is tabular data, comprising samples (rows) with the same set of features (columns). Tabular data is used in practical applications in many fields, and especially when stakeholders aim at monitoring the behavior of ICT systems for an early detection of anomalies, errors, or intrusions. Noticeably, monitoring activities generate a massive amount of (tabular) data that could be potentially used for classification but, unfortunately, is hardly labeled. In this case, the classification process can only be unsupervised, leading to well-known detrimental effects on the overall accuracy of the machinery, making it practically unusable in practice aside from specific corner cases.

To deal with these problems, ML experts and researchers keep designing new algorithms and providing trained models that have the potential to outperform their baseline on specific scenarios, can or cannot deal with unsupervised learning, and perform well under specific assumptions (e.g., the closed-world assumption of Independent and Identically Distributed (IID) train, validation, and test data). Constraining experiments behind such assumptions may be acceptable and suitable for research purposes but does not fit the deployment of classifiers for critical ICT systems, which are complex systems whose failure could directly impact the health of people, infrastructures, or the environment. Adopting classifiers to operate critical functions requires very low misclassification probability and to prepare countermeasures that avoid or mitigate the potential cascading effects of misclassifications to the encompassing critical system.

We tackle these problems by designing, implementing and benchmarking classifiers that are more accurate and robust than baselines, potentially becoming reliable enough to be deployed in the wild in critical real-world systems. Confidence ensembles, either Confidence Bagging (ConfBag) or Confidence Boosting (ConfBoost), can use any existing classifier as a base estimator, and build a meta-classifier through a training process that does not require any additional information with respect to those that are already used to train the base estimator. ConfBag reworks the traditional bagging by using only the most confident base-learners to assign the label to input data, whereas ConfBoost creates base-learners that are more and more specialized to classify train data items for which other base-learners cannot make a confident prediction. This is radically different from the typical boosting strategies, which always require a labeled training set as base-learners are typically trained using train data that gets misclassified by other base-learners. 
As a result, ConfBag and ConfBoost:
- can work with any existing classifier as base estimator, which is used to create base-learners;
-	have a generic and quite straightforward architecture that applies to all classification tasks and scenarios: (semi-)supervised or unsupervised, binary or multi-class, and can deal with tabular, image, or even textual input data; 
-	do not require additional information with respect to those already needed by the base estimator;
-	require little to no parameter tuning, are easy to use even from non-experts, and are implemented in a public Python framework that exposes interfaces that comply with de-facto standard libraries such as scikit-learn. 

## Dependencies

confidence-ensembles needs the following libraries:
- <a href="https://numpy.org/">NumPy</a>
- <a href="https://scipy.org/">SciPy</a>
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://scikit-learn.org/stable/">Scikit-Learn</a>
- <a href="https://github.com/yzhao062/pyod">PYOD</a>

## Usage

confidence-ensembles can wrap any classifier you may want to use, provided that the classifier implements scikit-learn like interfaces, namely
- classifier.fit(train_features, train_labels (optional)): trains the classifier; labels are optional (applies to unsupervised learning)
- classifier.predict_proba(test_set): takes a 2D ndarray and returns a 2D ndarray where each line contains probabilities for a given data point in the test_set

Assuming the classifier has such a structure, a confidence ensemble can be set up as it can be seen in the `test` folder

Snippet for supervised learning

https://github.com/tommyippoz/confidence-ensembles/blob/main/tests/example_supervised.py#L50-64

## Credits

Developed @ University of Trento, Povo, Italy, and University of Florence, Florence, Italy

Contributors
- Tommaso Zoppi
