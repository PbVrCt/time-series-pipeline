import os
import pathlib
import numpy as np

import keras_tuner as kt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn.metrics import accuracy_score

# Load the data
PATH = pathlib.Path(__file__).resolve().parent / "2_final data"
features = np.load(os.path.join(PATH, "train_val_features.npy"))
labels = np.load(os.path.join(PATH, "train_val_labels.npy"))
test_features = np.load(os.path.join(PATH, "test_features.npy"))
test_labels = np.load(os.path.join(PATH, "test_labels.npy"))
# Define the model: Support Vector Classifier
class SVM(kt.HyperModel):
    def build(self, hp):
        model_type = hp.Choice("model_type", ["SVC", "LinearSVC"])
        C = hp.Float("C", 1.0, 20.0, step=5)
        if model_type == "SVC":
            with hp.conditional_scope("model_type", "SVC"):
                kernel = hp.Choice("kernel", ["linear", "poly", "rbf", "sigmoid"])
                gamma = gamma = hp.Float("gamma", 0.01, 1, step=0.01)
                model = SVC(C=C, gamma=gamma, kernel=kernel)
        else:
            with hp.conditional_scope("model_type", "LinearSVC"):
                loss = hp.Choice("loss", ["hinge", "squared_hinge"])
                model = LinearSVC(C=C, loss=loss, max_iter=2000)
        return model


hypermodel = SVM()
# Find optimal hyperparameters
tuner = kt.tuners.SklearnTuner(
    oracle=kt.oracles.BayesianOptimizationOracle(
        # Keras docs: "Note that for this Tuner, the objective for the Oracle should always be set to Objective('score', direction='max')"
        objective=kt.Objective("score", "max"),
        max_trials=5,
    ),
    hypermodel=hypermodel,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    # Would be more appropiate to prune the sets based on label overlap, and possibly also to do the kind of CV described in "Advances in Financial ML"
    # but for now using .TimeSeriesSplit() with the gap parameter does the job
    cv=model_selection.TimeSeriesSplit(5, gap=100),
    project_name="Keras_tuner_metadata/SVM",
    overwrite=True,
)
tuner.search(features, labels)
# Show the results
tuner.results_summary(num_trials=3)
# Build the model with the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# # Do the final test on the test set
# best_model.fit(features, labels)
# predictions = best_model.predict(test_features)
# print('\n','Score on the test set: ',accuracy_score(test_labels, predictions))
