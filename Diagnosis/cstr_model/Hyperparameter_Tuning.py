from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Define a function to create the model
def create_model(learning_rate=0.001, num_layers=3, activation='relu', dropout_rate=0.2, units=64):
    model = Sequential()
    model.add(Dense(units, activation=activation, input_shape=(n_features,)))
    model.add(Dropout(dropout_rate))
    for _ in range(num_layers - 1):
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(n_features, activation='linear'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.001, 0.0001],  # Learning rates to try
    'num_layers': [3, 4],                     # Number of layers to try
    'activation': ['relu'],           # Activation functions to try
    'dropout_rate': [0.1, 0.2],             # Dropout rates to try
    'units': [64, 128]                           # Units configuration to try
}

# Create a KerasRegressor based on the defined model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=32, verbose=0)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid_search.fit(train_data, train_data)

# Print best parameters and best score
print("Best parameters found: ", grid_result.best_params_)
print("Best score found: ", grid_result.best_score_)



# %pip uninstall tensorflow
# %pip install tensorflow==2.12.0