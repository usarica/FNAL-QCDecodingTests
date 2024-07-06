class PyDatasetPlaceholder:
  """
  PyDatasetPlaceholder:
  Because PyDataset is introduced to TensorFlow in version >=2.16, we need to create a placeholder class to work around import errors.
  This class provides the same constructor signature as PyDataset but does nothing.
  """
  def __init__(self, workers=1, use_multiprocessing=False, max_queue_size=10):
    pass
