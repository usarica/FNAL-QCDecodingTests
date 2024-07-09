import numpy as np
import tensorflow as tf
from utilities_arrayops import *
from PyDatasetMgr import PyDatasetMgr


class BootstrapSampleManager:
  def __init__(
    self,
    features,
    labels,
    n_virtual_batches,
    virtual_batch_size,
    real_batch_size,
    validation_size,
    seed = 12345,
    use_TF = False
  ):
    self.use_TF = use_TF
    self.features = features
    self.labels = labels
    if self.use_TF:
      self.labels = tf.convert_to_tensor(self.labels)
      if isinstance(self.features, list) or isinstance(self.features, tuple):
        self.features = [ tf.convert_to_tensor(f) for f in self.features ]
      elif isinstance(self.features, dict):
        self.features = { k: tf.convert_to_tensor(v) for k, v in self.features }
      else:
        self.features = tf.convert_to_tensor(self.features)
    self.virtual_batch_size = virtual_batch_size # 100K
    self.real_batch_size = real_batch_size # 10K
    self.n_virtual_batches = n_virtual_batches # 500
    self.n_samples = arrayops_shape(labels, 0) # 10,000K
    self.validation_size = validation_size if validation_size is not None else 0
    if self.n_samples % self.real_batch_size != 0:
      raise RuntimeError("Number of samples should be a multiple of the real batch size.")
    if self.validation_size >= self.n_samples:
      raise RuntimeError("Validation size should be less than the number of samples.")
    if self.validation_size % self.virtual_batch_size != 0:
      raise RuntimeError("Validation size should be a multiple of the virtual batch size.")
    self.n_validation_batches = self.validation_size // self.virtual_batch_size
    self.n_samples_train = self.n_samples - self.validation_size
    self.n_real_batches = self.n_samples // self.real_batch_size # 1000
    self.n_real_batches_train = self.n_samples_train // self.real_batch_size
    self.regular_batches = (self.n_samples == self.n_virtual_batches*self.virtual_batch_size)
    if self.n_real_batches<self.n_virtual_batches:
      raise RuntimeError("Number of virtual batches should be less than or equal to the number of real batches.")
    if self.virtual_batch_size % self.real_batch_size != 0:
      raise RuntimeError("Virtual batch size should be a multiple of the real batch size.")
    self.n_real_batches_per_virtual_batch = self.virtual_batch_size // self.real_batch_size # 10
    if self.n_real_batches<self.n_real_batches_per_virtual_batch:
      raise RuntimeError("Number of real batches should be greater than the number of real batches per virtual batch.")
    self.n_training_batches = self.n_virtual_batches - self.n_validation_batches*self.n_real_batches_per_virtual_batch
    self.seed = seed
    self.sequence = None
    self.batch_groupings = None
    if not self.use_TF:
      np.random.seed(self.seed)
    self.epoch = -1


  def resample(self):
    self.epoch += 1
    if self.sequence is None:
      if not self.use_TF:
        self.sequence = np.arange(self.n_samples)
      else:
        self.sequence = tf.range(self.n_samples, dtype=tf.int32)
    else:
      if not self.use_TF:
        np.random.shuffle(self.sequence)
      else:
        self.sequence = tf.random.shuffle(self.sequence)
    self.batch_groupings = []
    if self.regular_batches:
      if not self.use_TF:
        self.batch_groupings.append(np.array_split(self.sequence[:-self.validation_size], self.n_training_batches))
      else:
        self.batch_groupings.append(tf.split(self.sequence[:-self.validation_size], self.n_training_batches, axis=0))
    else:
      seq_groupings = None
      if not self.use_TF:
        self.batch_sequence = np.arange(self.n_real_batches_train)
        seq_groupings = np.array_split(self.sequence[:-self.validation_size], self.n_real_batches_train)
      else:
        self.batch_sequence = tf.range(self.n_real_batches_train, dtype=tf.int32)
        seq_groupings = tf.split(self.sequence[:-self.validation_size], self.n_real_batches_train, axis=0)
      # Choose n_real_batches_per_virtual_batch batches in n_virtual_batches times
      virtual_batches = []
      for iv in range(self.n_training_batches):
        if iv>0:
          if not self.use_TF:
            np.random.shuffle(self.batch_sequence)
          else:
            self.batch_sequence = tf.random.shuffle(self.batch_sequence)
        virtual_batch_seq = []
        for i in range(self.n_real_batches_per_virtual_batch):
          virtual_batch_seq.append(seq_groupings[self.batch_sequence[i]])
        virtual_batch_seq = arrayops_concatenate(virtual_batch_seq, axis=0)
        virtual_batches.append(virtual_batch_seq)
      self.batch_groupings.append(virtual_batches)
      del seq_groupings
    if not self.use_TF:
      self.batch_groupings.append(np.array_split(self.sequence[-self.validation_size:], self.n_validation_batches))
    else:
      self.batch_groupings.append(tf.split(self.sequence[-self.validation_size:], self.n_validation_batches, axis=0))


  def get_labels(self, idx, use_train):
    if self.batch_groupings is None:
      self.resample()
    tti = 0 if use_train else 1
    return arrayops_gather(self.labels, self.batch_groupings[tti][idx], axis=0)


  def get_features(self, idx, use_train):
    if self.batch_groupings is None:
      self.resample()
    tti = 0 if use_train else 1
    if isinstance(self.features, list) or isinstance(self.features, tuple):
      return [ arrayops_gather(f, self.batch_groupings[tti][idx], axis=0) for f in self.features ]
    elif isinstance(self.features, dict):
      return { k: arrayops_gather(v, self.batch_groupings[tti][idx], axis=0) for k, v in self.features }
    else:
      return arrayops_gather(self.features, self.batch_groupings[tti][idx], axis=0)
      


class BootstrapSampler(PyDatasetMgr):
  def __init__(
    self,
    btsmgr,
    use_train,
    **kwargs
  ):
    super(BootstrapSampler, self).__init__(**kwargs)
    self.btsmgr = btsmgr
    self.use_train = use_train
    self.n_batches = self.btsmgr.n_validation_batches if not use_train else self.btsmgr.n_training_batches
    self.epoch_ctr = 0
  

  def __len__(self):
    return self.n_batches
  

  def __getitem__(self, idx):
    if self.epoch_ctr!=self.btsmgr.epoch:
      self.btsmgr.resample()
      if self.epoch_ctr!=self.btsmgr.epoch:
        raise RuntimeError("Failed to synchronize epochs.")
    return self.btsmgr.get_features(idx, self.use_train), self.btsmgr.get_labels(idx, self.use_train)
  

  def on_epoch_end(self):
    self.epoch_ctr += 1
    return None 