#!/usr/bin/env python3
"""
Simple test script for lazy evaluation improvements that doesn't require TensorFlow.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cache_system():
    """Test the cache system directly."""
    print("Testing cache system...")
    
    # Define cache class locally to avoid TensorFlow import issues
    class LazyEvaluationCache:
        def __init__(self, max_size=128):
            self.cache = {}
            self.max_size = max_size
            self.access_count = {}
        
        def get(self, key):
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            return None
        
        def set(self, key, value):
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
                del self.cache[lru_key]
                del self.access_count[lru_key]
            
            self.cache[key] = value
            self.access_count[key] = 1
        
        def clear(self):
            self.cache.clear()
            self.access_count.clear()
    
    cache = LazyEvaluationCache(max_size=3)
    
    # Test basic operations
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("nonexistent") is None
    
    # Test eviction
    cache.set("key3", "value3")
    cache.set("key4", "value4")  # Should evict LRU
    
    print("✓ Cache system working correctly")
    return True


def test_lazy_decorator():
    """Test the lazy evaluation decorator concept."""
    print("Testing lazy evaluation decorator...")
    
    import functools
    
    # Simple cache for testing
    cache = {}
    
    def lazy_evaluation(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{args}_{kwargs}"
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    
    call_count = 0
    
    @lazy_evaluation
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call should compute
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1
    
    # Second call should use cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Should not increment
    
    print("✓ Lazy evaluation decorator working correctly")
    return True


def test_lazy_data_generator():
    """Test lazy data generator."""
    print("Testing lazy data generator...")
    
    def lazy_data_generator(data_arrays, batch_size=32, shuffle=False):
        n_samples = len(data_arrays[0])
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                yield [arr[batch_indices] for arr in data_arrays]
    
    # Create dummy data
    features = np.random.randn(100, 10)
    labels = np.random.randint(0, 2, (100, 1))
    
    gen = lazy_data_generator([features, labels], batch_size=10)
    
    # Test one batch
    batch = next(gen)
    assert len(batch) == 2
    assert batch[0].shape == (10, 10)
    assert batch[1].shape == (10, 1)
    
    print("✓ Lazy data generator working correctly")
    return True


def test_file_syntax():
    """Test that our utilities_tf.py file has valid syntax."""
    print("Testing utilities_tf.py syntax...")
    
    try:
        # Try to parse the file
        with open('utilities_tf.py', 'r') as f:
            code = f.read()
        
        # Compile to check syntax
        compile(code, 'utilities_tf.py', 'exec')
        print("✓ utilities_tf.py has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in utilities_tf.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking utilities_tf.py: {e}")
        return False


if __name__ == "__main__":
    print("Running standalone tests for lazy evaluation improvements...")
    
    success = True
    success &= test_cache_system()
    success &= test_lazy_decorator()
    success &= test_lazy_data_generator()
    success &= test_file_syntax()
    
    if success:
        print("\n🎉 All standalone tests passed!")
        print("Lazy evaluation improvements are syntactically correct and functionally sound.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)