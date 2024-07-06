try:
  from tensorflow.keras.utils import PyDataset as PyDatasetMgr
except ImportError:
  from tensorflow.keras.utils import Sequence as PyDatasetMgr
