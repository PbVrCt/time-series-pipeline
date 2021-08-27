import os
import pathlib
import numpy as np

import keras_tuner as kt
from sklearn.preprocessing import OneHotEncoder
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn.metrics import accuracy_score

path = pathlib.Path(__file__).resolve().parent / '2_final data'
features  = np.load( os.path.join( path, 'train_val_features.npy') )
labels    = np.load( os.path.join( path, 'train_val_labels.npy') )
test_features  = np.load( os.path.join( path, 'test_features.npy') )
test_labels    = np.load( os.path.join( path, 'test_labels.npy') )

# One hot encode labels
labels      = OneHotEncoder().fit_transform(X=labels.reshape(-1,1)).toarray()
test_labels = OneHotEncoder().fit_transform(X=test_labels.reshape(-1,1)).toarray()
# Define the model: Random Forest
class RF(kt.HyperModel):
    def build(self, hp):
        model = ensemble.RandomForestClassifier(
            n_estimators=hp.Int('n_estimators', 30, 80, step=10),
            max_depth=hp.Int('max_depth', 3, 10))
        return model
hypermodel = RF()
#Find optimal hyperparameters
tuner = kt.tuners.SklearnTuner(
    oracle=kt.oracles.BayesianOptimizationOracle(
        # Keras docs: "Note that for this Tuner, the objective for the Oracle should always be set to Objective('score', direction='max')"
        objective=kt.Objective('score', 'max'), 
        max_trials=20),
    hypermodel=hypermodel,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    # Would be more appropiate to prune the sets based on label overlap, and possibly also to do the kind of CV described in "Advances in Financial ML"
    # but for now using .TimeSeriesSplit() with the gap parameter does the job
    cv=model_selection.TimeSeriesSplit(5, gap=100), 
    project_name='Keras_tuner_metadata/RF',
    overwrite=True,
    )
tuner.search(features, labels)
#Show the results
tuner.results_summary(num_trials=3)
# Build the model with the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0] 
best_model = tuner.hypermodel.build(best_hps)

# # Do the final test on the test set
# best_model.fit(features, labels)
# predictions = best_model.predict(test_features)
# print('\n','Score on the test set: ',accuracy_score(test_labels, predictions))