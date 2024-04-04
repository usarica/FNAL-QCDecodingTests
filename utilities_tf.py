from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate
from qkeras import QDense, QActivation
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np


def build_sequential_dense_model(
    n_features, output_n_pred,
    dense_layers,
    loss_fcn = "binary_crossentropy",
    output_activation = "sigmoid"
  ):
  """
  Build a sequential model with dense layers.
  - Arguments:
    n_features: Number of features in the input
    output_n_pred: Number of output predictions
    dense_layers: List of the number of neurons in each dense layer
    loss_fcn: Loss function to use, defaulted to 'binary_crossentropy'
    output_activation: Activation function for the output layer, defaulted to 'sigmoid'
  - Return type:
    Sequential model
  """
  nnlayers = [ Input(shape=(n_features,)) ]

  for n in dense_layers:
    nnlayers.append(Dense(n, activation='relu'))
  nnlayers.append(Dense(output_n_pred, activation=output_activation))

  model = Sequential(nnlayers)
  model.summary()
  model.compile(optimizer='adam', loss=loss_fcn, metrics=['accuracy'])

  return model


def build_sequential_qdense_model(
    n_features, output_n_pred,
    dense_layers,
    loss_fcn = "binary_crossentropy",
    output_activation = "quantized_sigmoid(4)"
  ):
  """
  Build a sequential model with QDense layers.
  - Arguments:
    n_features: Number of features in the input
    output_n_pred: Number of output predictions
    dense_layers: List of tuples of the number of neurons in each dense layer and the number of bits for the activation function
    loss_fcn: Loss function to use, defaulted to 'binary_crossentropy'
    output_activation: Activation function for the output layer, defaulted to 'quantized_sigmoid(4)'
  - Return type:
    Sequential model
  """
  nnlayers = [ Input(shape=(n_features,)) ]

  for n, b in dense_layers:
    nnlayers.append(QDense(n, activation=f'quantized_relu({b})'))
  nnlayers.append(QDense(output_n_pred, activation=output_activation))

  model = Sequential(nnlayers)
  model.summary()
  model.compile(optimizer='adam', loss=loss_fcn, metrics=['accuracy'])

  return model


def test_model(model, data_train, pred_train, data_test, pred_test, verbosity=0):
  """
  Print the test statistics of a TF model from the testing and training data.
  - Arguments:
    model: Model
    data_train: Training data
    pred_train: Training labels
    data_test: Testing data
    pred_test: Testing labels
    verbosity: Verbosity level
  Return type:
    None
  """
  loss, accuracy = model.evaluate(data_train, pred_train, verbose=verbosity)
  print(f"Training data loss: {loss:.6f}, accuracy: {accuracy:.6f}")
  loss, accuracy = model.evaluate(data_test, pred_test, verbose=verbosity)
  print(f"Test data loss: {loss:.6f}, accuracy: {accuracy:.6f}")


def predict_model(model, features, labels):
  """
  Get the predictions of a model on a given dataset.
  - Arguments:
    model: Model
    data: Data to predict
  - Return type:
    Numpy array
  """
  n_data = labels.shape[0]
  n_flips = np.sum(labels.reshape(-1,) != 0)
  n_unflips = np.sum(labels.reshape(-1,) == 0)
  prediction = model.predict(features, batch_size=n_data//10)
  prediction = (prediction>0)
  matches = (prediction != labels)
  matches_flipped = matches*(labels != 0)
  matches_unflipped = matches*(labels == 0)
  n_matches = np.sum(matches)
  n_matches_flipped = np.sum(matches_flipped)
  n_matches_unflipped = np.sum(matches_unflipped)
  print(f"Prediction accuracy: {1.0 - n_matches/n_data:.6f}")
  print(f"- Flipped/unflipped accuracies: {1.0 - n_matches_flipped/n_flips:.6f} / {1.0 - n_matches_unflipped/n_unflips:.6f}")

  return prediction


def split_data(*arrays, test_size=0.2, seed=12345, shuffle=False):
  """
  Split the data into training and testing sets.
  - Arguments:
    features: Features and labels
    test_size: Fraction of the data to use for testing
    seed: Random seed
    shuffle: Shuffle the data
  - Return type:
    Tuple of features_train, features_test, etc.
  """
  return train_test_split(*arrays, test_size=test_size, random_state=seed, shuffle=shuffle)


