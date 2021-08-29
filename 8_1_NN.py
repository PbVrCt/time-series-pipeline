import os
import pathlib
import json
import datetime

import numpy as np
import tensorflow as tf
import keras_tuner as kt

from utils.NNspecifications import NNmodel, plotHistory

# Load the data
PATH = pathlib.Path(__file__).resolve().parent / "2_final data"
train_features = np.load(os.path.join(PATH, "train_features.npy"))
train_labels = np.load(os.path.join(PATH, "train_labels.npy"))
val_features = np.load(os.path.join(PATH, "val_features.npy"))
val_labels = np.load(os.path.join(PATH, "val_labels.npy"))
test_features = np.load(os.path.join(PATH, "test_features.npy"))
test_labels = np.load(os.path.join(PATH, "test_labels.npy"))
# One hot encode labels
train_labels = tf.one_hot(train_labels, depth=2, on_value=1, off_value=0)
val_labels = tf.one_hot(val_labels, depth=2, on_value=1, off_value=0)
test_labels = tf.one_hot(test_labels, depth=2, on_value=1, off_value=0)
# Define the model: NN
hypermodel = NNmodel(input_shape=train_features.shape[1], num_classes=2)
# Find optimal hyperparameters
# # Define a class to implement tunable callbacks
class MyTuner(kt.BayesianOptimization):
    def run_trial(self, *args, **kwargs):
        patience = hp.Int("patience", 0, 10, default=5)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
        log_dir = "TensorBoard_logs/data/" + datetime.datetime.now().strftime(
            "%H%M%S-%Y%m%d"
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.8,
            patience=patience,
            min_lr=0.000001,
            verbose=0,
            monitor="val_acurracy",
        )
        callbacks = [
            reduce_lr,
            early_stopping,
            tensorboard_callback,
        ]
        super(MyTuner, self).run_trial(*args, **kwargs, callbacks=callbacks)


# # Initialize the tuner
hp = kt.HyperParameters()
hp.Fixed("clipnorm", value=1)
hp.Fixed("clipvalue", value=0.5)
# hp.Fixed('learning_rate',value=5e-6)
hp.Fixed("patience", value=10)
tuner = MyTuner(
    hypermodel=hypermodel,
    hyperparameters=hp,
    objective=kt.Objective("val_loss", direction="min"),
    max_trials=20,
    seed=50,
    project_name="Keras_tuner_metadata/NN",
    overwrite=True,
)
# # Try hyperparameter combinations
tuner.search(
    train_features,
    train_labels,
    epochs=30,
    validation_data=(val_features, val_labels),
    batch_size=32,
    verbose=0,
    use_multiprocessing=True,
)
# Build the model with the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
# Fit the final model to the train and validation sets
# # Callbacks
patience = best_hps.get("patience")
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.7, patience=patience, min_lr=0.000001, verbose=0
)
# # Fit
history = best_model.fit(
    train_features,
    train_labels,
    epochs=50,
    validation_data=(val_features, val_labels),
    callbacks=[early_stopping, reduce_lr],
    batch_size=32,
    verbose=0,
)
# Show the results on the train and validation sets
print("\n")
tuner.results_summary(num_trials=1)
best_model.summary()
plotHistory(history)
print("\n")
# Confusion matrix
# TP FN
# FP TN
predictions = tf.math.argmax(best_model.predict(train_features), 1)
train_labels = tf.math.argmax(train_labels, 1)
conf_matrix = tf.math.confusion_matrix(labels=train_labels, predictions=predictions)
print("Train set confusion matrix ************", "\n", conf_matrix)
predictions = tf.math.argmax(best_model.predict(val_features), 1)
val_labels = tf.math.argmax(val_labels, 1)
conf_matrix = tf.math.confusion_matrix(labels=val_labels, predictions=predictions)
print("Val set confusion matrix **************", "\n", conf_matrix)

# # Do the final test on the test set
# result = best_model.evaluate(test_features, test_labels, verbose = 0)
# print("[test loss, test metrics]:", result)

# # Save the model and the hyperparameter configuration
# best_model.save('finalNN.tf')
# config = best_model.get_config()
# with open('model_config.json', 'w') as fp:
#     json.dump(config, fp)
