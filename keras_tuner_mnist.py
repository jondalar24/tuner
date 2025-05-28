"""
Hyperparameter Tuning with Keras Tuner
--------------------------------------
Automated tuning of a simple neural network on the MNIST dataset.
Author: Ángel Calvar Pastoriza
"""

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

# 1. Load and preprocess MNIST
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

# 2. Define model-building function
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    
    # Tune number of units
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation='relu'))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Tune learning rate
    hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=Adam(learning_rate=hp_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Create tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='kt_mnist_logs',
    project_name='mnist_tuning'
)

# 4. Perform search
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# 5. Retrieve best model and hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"Best number of units: {best_hps.get('units')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# 6. Build and train final model
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# 7. Evaluate on validation set
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Final validation accuracy: {val_acc}")
