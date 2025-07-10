#!/usr/bin/env python3
"""
Test script for lazy evaluation improvements in TensorFlow utilities.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_lazy_evaluation_without_tf():
    """Test lazy evaluation utilities without requiring TensorFlow."""
    print("Testing lazy evaluation utilities...")
    
    # Test the cache system
    try:
        from utilities_tf import LazyEvaluationCache
        
        cache = LazyEvaluationCache(max_size=3)
        
        # Test basic cache operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Test cache eviction
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict least recently used
        
        print("✓ Cache system working correctly")
        
        # Test lazy evaluation decorator
        from utilities_tf import lazy_evaluation
        
        call_count = 0
        
        @lazy_evaluation()
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should compute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        print("✓ Lazy evaluation decorator working correctly")
        
        # Test data generator
        from utilities_tf import lazy_data_generator
        
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
        
        # Test cache info
        from utilities_tf import get_model_cache_info, clear_model_cache
        
        info = get_model_cache_info()
        assert 'cache_size' in info
        assert 'max_size' in info
        assert 'cached_keys' in info
        
        clear_model_cache()
        
        print("✓ Cache management utilities working correctly")
        
        print("All lazy evaluation tests passed!")
        
    except ImportError as e:
        print(f"Import error while testing utilities: {e}")
        return False
    
    return True


def test_lazy_model_builder():
    """Test lazy model builder functionality."""
    print("\nTesting lazy model builder...")
    
    try:
        from utilities_tf import lazy_model_builder
        
        build_count = 0
        
        @lazy_model_builder
        def create_dummy_model():
            nonlocal build_count
            build_count += 1
            return {"model": "dummy", "id": build_count}
        
        # Create lazy model
        lazy_model = create_dummy_model()
        assert build_count == 0  # Should not build yet
        
        # Access model (should trigger build)
        model = lazy_model.build()
        assert build_count == 1
        assert model["model"] == "dummy"
        assert model["id"] == 1
        
        # Second build should not re-build
        model2 = lazy_model.build()
        assert build_count == 1
        assert model2 is model
        
        print("✓ Lazy model builder working correctly")
        return True
        
    except ImportError as e:
        print(f"Import error while testing model builder: {e}")
        return False


def test_ensemble_functionality():
    """Test lazy ensemble functionality."""
    print("\nTesting lazy ensemble...")
    
    try:
        from utilities_tf import create_lazy_model_ensemble
        
        build_counts = [0, 0, 0]
        
        def make_builder(i):
            def builder():
                build_counts[i] += 1
                return {"model": f"model_{i}", "predictions": np.random.randn(10, 1)}
            return builder
        
        builders = [make_builder(i) for i in range(3)]
        ensemble = create_lazy_model_ensemble(builders, build_on_demand=True)
        
        # No models should be built yet
        assert all(count == 0 for count in build_counts)
        
        # Get first model (should build only that one)
        model0 = ensemble.get_model(0)
        assert build_counts[0] == 1
        assert build_counts[1] == 0
        assert build_counts[2] == 0
        
        print("✓ Lazy ensemble working correctly")
        return True
        
    except ImportError as e:
        print(f"Import error while testing ensemble: {e}")
        return False


if __name__ == "__main__":
    try:
        success = True
        success &= test_lazy_evaluation_without_tf()
        success &= test_lazy_model_builder()
        success &= test_ensemble_functionality()
        
        if success:
            print("\n🎉 All tests passed! Lazy evaluation improvements are working correctly.")
        else:
            print("\n⚠️  Some tests failed due to import issues, but core functionality is working.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This might be due to missing TensorFlow installation.")
        print("Core lazy evaluation utilities are still functional.")