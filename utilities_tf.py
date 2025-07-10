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
import uproot as uprt
import awkward as ak
import functools
import warnings
from copy import deepcopy


# Lazy Evaluation Utilities for TensorFlow Models
class LazyEvaluationCache:
    """Cache for expensive computations to support lazy evaluation."""
    
    def __init__(self, max_size=128):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key):
        """Get cached value if available."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Set cached value, evicting LRU if needed."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.access_count.clear()


# Global cache instance
_model_cache = LazyEvaluationCache()


def lazy_evaluation(cache_key=None, use_cache=True):
    """
    Decorator for lazy evaluation of TensorFlow model operations.
    
    Args:
        cache_key: Optional key for caching results
        use_cache: Whether to use caching
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key if not provided
            if cache_key is None:
                key = f"{func.__name__}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
            else:
                key = cache_key
            
            # Try to get from cache first
            if use_cache:
                cached_result = _model_cache.get(key)
                if cached_result is not None:
                    return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result if enabled
            if use_cache:
                _model_cache.set(key, result)
            
            return result
        return wrapper
    return decorator


def lazy_model_builder(builder_func):
    """
    Decorator that creates a lazy model builder.
    The actual model is only built when needed.
    """
    @functools.wraps(builder_func)
    def wrapper(*args, **kwargs):
        class LazyModel:
            def __init__(self, builder_func, args, kwargs):
                self._builder_func = builder_func
                self._args = args
                self._kwargs = kwargs
                self._model = None
                self._built = False
            
            def build(self):
                """Build the model if not already built."""
                if not self._built:
                    self._model = self._builder_func(*self._args, **self._kwargs)
                    self._built = True
                return self._model
            
            def __getattr__(self, name):
                """Forward attribute access to the underlying model."""
                if self._model is None:
                    self.build()
                return getattr(self._model, name)
            
            def __call__(self, *args, **kwargs):
                """Make the lazy model callable."""
                if self._model is None:
                    self.build()
                return self._model(*args, **kwargs)
        
        return LazyModel(builder_func, args, kwargs)
    return wrapper


def enable_lazy_evaluation():
    """Enable lazy evaluation optimizations globally."""
    try:
        import tensorflow as tf
        # Enable memory growth to support lazy evaluation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Enable XLA compilation for lazy evaluation
        tf.config.optimizer.set_jit(True)
        
    except ImportError:
        warnings.warn("TensorFlow not available, skipping lazy evaluation setup")
    except Exception as e:
        warnings.warn(f"Could not enable lazy evaluation optimizations: {e}")


@lazy_evaluation()
def build_sequential_dense_model(
    n_features, output_n_pred,
    dense_layers,
    loss_fcn = "binary_crossentropy",
    output_activation = "sigmoid",
    lazy_build = False
  ):
  """
  Build a sequential model with dense layers.
  - Arguments:
    n_features: Number of features in the input
    output_n_pred: Number of output predictions
    dense_layers: List of the number of neurons in each dense layer
    loss_fcn: Loss function to use, defaulted to 'binary_crossentropy'
    output_activation: Activation function for the output layer, defaulted to 'sigmoid'
    lazy_build: If True, return a lazy model that builds only when needed
  - Return type:
    Sequential model or LazyModel wrapper
  """
  def _build_model():
    nnlayers = [ Input(shape=(n_features,)) ]

    for n in dense_layers:
      nnlayers.append(Dense(n, activation='relu'))
    nnlayers.append(Dense(output_n_pred, activation=output_activation))

    model = Sequential(nnlayers)
    model.summary()
    model.compile(optimizer='adam', loss=loss_fcn, metrics=['accuracy'])
    return model
  
  if lazy_build:
    return lazy_model_builder(_build_model)()
  else:
    return _build_model()


@lazy_evaluation()
def build_sequential_qdense_model(
    n_features, output_n_pred,
    dense_layers,
    loss_fcn = "binary_crossentropy",
    output_activation = "quantized_sigmoid(4)",
    lazy_build = False
  ):
  """
  Build a sequential model with QDense layers.
  - Arguments:
    n_features: Number of features in the input
    output_n_pred: Number of output predictions
    dense_layers: List of tuples of the number of neurons in each dense layer and the number of bits for the activation function
    loss_fcn: Loss function to use, defaulted to 'binary_crossentropy'
    output_activation: Activation function for the output layer, defaulted to 'quantized_sigmoid(4)'
    lazy_build: If True, return a lazy model that builds only when needed
  - Return type:
    Sequential model or LazyModel wrapper
  """
  def _build_model():
    nnlayers = [ Input(shape=(n_features,)) ]

    for n, b in dense_layers:
      nnlayers.append(QDense(n, activation=f'quantized_relu({b})'))
    nnlayers.append(QDense(output_n_pred, activation=output_activation))

    model = Sequential(nnlayers)
    model.summary()
    model.compile(optimizer='adam', loss=loss_fcn, metrics=['accuracy'])
    return model
  
  if lazy_build:
    return lazy_model_builder(_build_model)()
  else:
    return _build_model()


@lazy_evaluation()
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


@lazy_evaluation()
def predict_model(model, features, labels, batch_size=None, use_lazy_prediction=False):
  """
  Get the predictions of a model on a given dataset.
  - Arguments:
    model: Model
    features: Input features
    labels: True labels
    batch_size: Batch size for prediction (auto-computed if None)
    use_lazy_prediction: Whether to use lazy prediction with smaller batches
  - Return type:
    Numpy array
  """
  n_data = labels.shape[0]
  n_flips = np.sum(labels.reshape(-1,) != 0)
  n_unflips = np.sum(labels.reshape(-1,) == 0)
  
  # Auto-compute batch size for lazy evaluation
  if batch_size is None:
    if use_lazy_prediction:
      # Use smaller batches for memory efficiency
      batch_size = min(n_data//20, 1000)
    else:
      batch_size = n_data//10
  
  # Lazy prediction with batch processing
  if use_lazy_prediction and n_data > batch_size:
    predictions = []
    for i in range(0, n_data, batch_size):
      end_idx = min(i + batch_size, n_data)
      batch_features = features[i:end_idx]
      batch_pred = model.predict(batch_features, batch_size=batch_size, verbose=0)
      predictions.append(batch_pred)
    prediction = np.concatenate(predictions, axis=0)
  else:
    prediction = model.predict(features, batch_size=batch_size)
  
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


def save_history(history, path):
  history_raw_data = deepcopy(history.history)
  for key, val in history_raw_data.items():
    if isinstance(val, np.ndarray):
      history_raw_data[key] = val.astype(np.float32)
    else:
      history_raw_data[key] = np.float32(val)
      
  history_data = ak.Array(history_raw_data)
  with uprt.recreate(f"{path}") as fout:
    fout["history"] = history_data


# Additional lazy evaluation utilities
def lazy_data_generator(data_arrays, batch_size=32, shuffle=False):
  """
  Create a lazy data generator that yields batches on demand.
  
  Args:
    data_arrays: List of numpy arrays (features, labels, etc.)
    batch_size: Batch size for yielding
    shuffle: Whether to shuffle data between epochs
  
  Yields:
    Batches of data
  """
  n_samples = len(data_arrays[0])
  indices = np.arange(n_samples)
  
  while True:
    if shuffle:
      np.random.shuffle(indices)
    
    for i in range(0, n_samples, batch_size):
      batch_indices = indices[i:i+batch_size]
      yield [arr[batch_indices] for arr in data_arrays]


def memory_efficient_model_training(model, train_data, val_data, 
                                   epochs=10, batch_size=32, 
                                   use_lazy_loading=True):
  """
  Train model with memory-efficient lazy evaluation.
  
  Args:
    model: TensorFlow model
    train_data: Training data (features, labels)
    val_data: Validation data (features, labels)
    epochs: Number of training epochs
    batch_size: Batch size
    use_lazy_loading: Whether to use lazy data loading
  
  Returns:
    Training history
  """
  if use_lazy_loading:
    # Use lazy data generator for memory efficiency
    train_gen = lazy_data_generator(train_data, batch_size=batch_size, shuffle=True)
    val_gen = lazy_data_generator(val_data, batch_size=batch_size, shuffle=False)
    
    steps_per_epoch = len(train_data[0]) // batch_size
    validation_steps = len(val_data[0]) // batch_size
    
    history = model.fit(
      train_gen,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data=val_gen,
      validation_steps=validation_steps,
      verbose=1
    )
  else:
    # Standard training
    history = model.fit(
      train_data[0], train_data[1],
      validation_data=(val_data[0], val_data[1]),
      epochs=epochs,
      batch_size=batch_size,
      verbose=1
    )
  
  return history


def create_lazy_model_ensemble(model_builders, build_on_demand=True):
  """
  Create an ensemble of models with lazy evaluation.
  
  Args:
    model_builders: List of model builder functions
    build_on_demand: Whether to build models only when needed
  
  Returns:
    Ensemble wrapper
  """
  class LazyEnsemble:
    def __init__(self, builders, lazy=True):
      self.builders = builders
      self.models = [None] * len(builders)
      self.lazy = lazy
      
      if not lazy:
        # Build all models immediately
        for i, builder in enumerate(builders):
          self.models[i] = builder()
    
    def get_model(self, index):
      """Get model at index, building if necessary."""
      if self.models[index] is None:
        self.models[index] = self.builders[index]()
      return self.models[index]
    
    def predict(self, x, method='average'):
      """Make predictions using ensemble."""
      predictions = []
      for i in range(len(self.builders)):
        model = self.get_model(i)
        pred = model.predict(x)
        predictions.append(pred)
      
      predictions = np.array(predictions)
      
      if method == 'average':
        return np.mean(predictions, axis=0)
      elif method == 'voting':
        return np.round(np.mean(predictions, axis=0))
      else:
        return predictions
    
    def __len__(self):
      return len(self.builders)
  
  return LazyEnsemble(model_builders, lazy=build_on_demand)


def get_model_cache_info():
  """Get information about the current model cache."""
  return {
    'cache_size': len(_model_cache.cache),
    'max_size': _model_cache.max_size,
    'cached_keys': list(_model_cache.cache.keys())
  }


def clear_model_cache():
  """Clear the global model cache."""
  _model_cache.clear()
  print("Model cache cleared.")