# Lazy Evaluation Improvements for TensorFlow Utilities

This document describes the lazy evaluation improvements added to the FNAL-QCDecodingTests repository.

## Overview

The lazy evaluation improvements focus on optimizing memory usage and computational efficiency by deferring expensive operations until they are actually needed. This is particularly important for quantum error correction neural networks that can have large memory footprints.

## Key Features

### 1. Lazy Model Building

**Problem**: Traditional model building creates all layers and compiles the model immediately, consuming memory even if the model isn't used right away.

**Solution**: Added `lazy_build` parameter to model building functions and `@lazy_model_builder` decorator.

```python
# Traditional (eager) model building
model = build_sequential_dense_model(100, 1, [64, 32])

# Lazy model building - model is built only when first used
lazy_model = build_sequential_dense_model(100, 1, [64, 32], lazy_build=True)
# or
@lazy_model_builder
def create_model():
    return build_sequential_dense_model(100, 1, [64, 32])
```

### 2. Computation Caching

**Problem**: Expensive computations (like model predictions) are recalculated even with identical inputs.

**Solution**: Added `@lazy_evaluation()` decorator with LRU cache.

```python
@lazy_evaluation()
def expensive_prediction(model, data):
    return model.predict(data)

# First call computes and caches
result1 = expensive_prediction(model, data)
# Second call with same args uses cache
result2 = expensive_prediction(model, data)  # No recomputation
```

### 3. Memory-Efficient Data Processing

**Problem**: Large datasets can cause memory issues when loaded all at once.

**Solution**: Added lazy data generators and batch processing.

```python
# Memory-efficient prediction with lazy batching
predictions = predict_model(model, features, labels, 
                           use_lazy_prediction=True, 
                           batch_size=100)

# Lazy data generator for training
gen = lazy_data_generator([features, labels], batch_size=32, shuffle=True)
```

### 4. Lazy Model Ensembles

**Problem**: Model ensembles build all models upfront, consuming significant memory.

**Solution**: Added lazy ensemble that builds models on-demand.

```python
# Create ensemble that builds models only when needed
ensemble = create_lazy_model_ensemble(
    [lambda: build_model1(), lambda: build_model2()],
    build_on_demand=True
)

# Only builds model 0 when accessed
prediction = ensemble.get_model(0).predict(data)
```

### 5. Global Optimization Settings

**Problem**: TensorFlow uses eager execution by default, which can be memory-intensive.

**Solution**: Added `enable_lazy_evaluation()` function to configure TensorFlow for lazy operation.

```python
# Enable lazy evaluation optimizations
enable_lazy_evaluation()
```

This function:
- Enables GPU memory growth to prevent pre-allocation
- Sets up mixed precision for better performance
- Enables XLA compilation for lazy graph execution

## API Reference

### Core Classes

#### `LazyEvaluationCache`
LRU cache for storing computation results.

- `get(key)`: Retrieve cached value
- `set(key, value)`: Store value with LRU eviction
- `clear()`: Clear all cached values

### Decorators

#### `@lazy_evaluation(cache_key=None, use_cache=True)`
Caches function results to avoid recomputation.

#### `@lazy_model_builder`
Creates a lazy wrapper that builds models only when accessed.

### Functions

#### `build_sequential_dense_model(..., lazy_build=False)`
Enhanced version with lazy building support.

#### `predict_model(..., use_lazy_prediction=False, batch_size=None)`
Enhanced prediction with memory-efficient batching.

#### `lazy_data_generator(data_arrays, batch_size=32, shuffle=False)`
Generator for memory-efficient data loading.

#### `memory_efficient_model_training(model, train_data, val_data, ...)`
Training function with lazy data loading.

#### `create_lazy_model_ensemble(model_builders, build_on_demand=True)`
Create ensemble with lazy model building.

### Utilities

#### `enable_lazy_evaluation()`
Configure TensorFlow for lazy evaluation.

#### `get_model_cache_info()`
Get information about current cache usage.

#### `clear_model_cache()`
Clear the global model cache.

## Performance Benefits

1. **Memory Efficiency**: Models and data are loaded only when needed
2. **Computation Caching**: Avoid redundant expensive operations
3. **Batch Processing**: Process large datasets in memory-efficient chunks
4. **Lazy Compilation**: Defer TensorFlow graph compilation until execution

## Usage Examples

### Basic Lazy Model Usage

```python
from utilities_tf import build_sequential_dense_model, enable_lazy_evaluation

# Enable lazy evaluation optimizations
enable_lazy_evaluation()

# Create lazy model
model = build_sequential_dense_model(
    n_features=100,
    output_n_pred=1,
    dense_layers=[64, 32],
    lazy_build=True
)

# Model is built only when first used
predictions = model.predict(data)
```

### Memory-Efficient Training

```python
from utilities_tf import memory_efficient_model_training, lazy_data_generator

# Train with lazy data loading
history = memory_efficient_model_training(
    model=model,
    train_data=[X_train, y_train],
    val_data=[X_val, y_val],
    epochs=10,
    batch_size=32,
    use_lazy_loading=True
)
```

### Ensemble with Lazy Building

```python
from utilities_tf import create_lazy_model_ensemble

# Create ensemble builders
builders = [
    lambda: build_sequential_dense_model(100, 1, [64, 32]),
    lambda: build_sequential_dense_model(100, 1, [128, 64]),
    lambda: build_sequential_dense_model(100, 1, [256, 128])
]

# Create lazy ensemble
ensemble = create_lazy_model_ensemble(builders, build_on_demand=True)

# Only builds models when predictions are needed
predictions = ensemble.predict(data, method='average')
```

## Backward Compatibility

All improvements are backward compatible. Existing code will continue to work without changes. Lazy evaluation features are opt-in through new parameters and functions.

## Testing

Run the test suite to verify functionality:

```bash
python test_lazy_simple.py
```

This tests the core lazy evaluation functionality without requiring TensorFlow installation.

## Future Enhancements

1. **Adaptive Caching**: Automatically adjust cache size based on memory usage
2. **Distributed Lazy Evaluation**: Support for distributed computing environments
3. **Lazy Layer Compilation**: Defer individual layer compilation in complex models
4. **Memory Profiling**: Built-in memory usage monitoring and optimization suggestions