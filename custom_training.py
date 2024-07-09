import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from utilities_arrayops import *


class RCNNRecurrenceLoss(Loss):
  def __init__(
    self,
    rounds,
    has_final_state,
    scale_init=1.,
    scale_leadin=1.,
    scale_recurrent=1.,
    scale_final=1.,
    **kwargs
  ):
    super().__init__(**kwargs)
    self.rounds = rounds
    self.has_final_state = has_final_state

    self.initial_round = 0
    self.leadin_round = self.initial_round+1 if self.rounds>2 else None
    self.final_round = -1 if has_final_state else None
    self.has_recurrent_rounds = self.rounds>3
    self.recurrent_round_bound = self.final_round

    self.scale_init = scale_init
    self.scale_leadin = scale_leadin
    self.scale_recurrent = scale_recurrent
    self.scale_final = scale_final


  def call(self, y_true, y_pred):
    init = tf.keras.losses.binary_crossentropy(y_true[:,self.initial_round], y_pred[:,self.initial_round])*self.scale_init
    leadin = tf.keras.losses.binary_crossentropy(y_true[:,self.leadin_round], y_pred[:,self.leadin_round])*self.scale_leadin if self.leadin_round is not None else None
    recurrent = None
    if self.has_recurrent_rounds:
      if self.recurrent_round_bound is not None:
        recurrent = tf.keras.losses.binary_crossentropy(y_true[:,self.leadin_round+1:self.recurrent_round_bound], y_pred[:,self.leadin_round+1:self.recurrent_round_bound])
      else:
        recurrent = tf.keras.losses.binary_crossentropy(y_true[:,self.leadin_round+1:], y_pred[:,self.leadin_round+1:])
      recurrent *= self.scale_recurrent
    final = tf.keras.losses.binary_crossentropy(y_true[:,self.final_round], y_pred[:,self.final_round])*self.scale_final if self.final_round is not None else None

    res = tf.math.reduce_mean(init)
    if leadin is not None:
      res += tf.math.reduce_mean(leadin)
    if recurrent is not None:
      res += tf.math.reduce_mean(recurrent)
    if final is not None:
      res += tf.math.reduce_mean(final)
    return res
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'rounds': self.rounds,
        'has_final_state': self.has_final_state,
        'scale_init': self.scale_init,
        'scale_leadin': self.scale_leadin,
        'scale_recurrent': self.scale_recurrent,
        'scale_final': self.scale_final,
      }
    )
    return config
    


class RCNNRecurrenceAccuracy(Metric):
  def __init__(
    self,
    rounds,
    has_final_state,
    **kwargs
  ):
    super().__init__(**kwargs)
    self.rounds = rounds
    self.has_final_state = has_final_state

    self.initial_round = 0
    self.leadin_round = self.initial_round+1 if self.rounds>2 else None
    self.final_round = -1 if has_final_state else None
    self.has_recurrent_rounds = self.rounds>3
    self.recurrent_round_bound = self.final_round

    self.counter = self.add_weight(name='counter', initializer='zeros')

    self.init_accuracy = self.add_weight(name='init_accuracy', initializer='zeros')
    self.leadin_accuracy = self.add_weight(name='leadin_accuracy', initializer='zeros') if self.leadin_round is not None else None
    self.recurrent_accuracy = self.add_weight(name='recurrent_accuracy', initializer='zeros') if self.has_recurrent_rounds else None
    self.final_accuracy = self.add_weight(name='final_accuracy', initializer='zeros') if self.final_round is not None else None


  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'rounds': self.rounds,
        'has_final_state': self.has_final_state,
      }
    )
    return config


  @staticmethod
  def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    binary_accuracy: This is a method taken from https://github.com/keras-team/keras/blob/v3.3.3/keras/src/metrics/accuracy_metrics.py#L65-L75.
    In some versions of TensorFlow, there is a type casting bug for y_pred, i.e.,
    casting the type of y_pred>threshold to y_pred.dtype, then comparing to y_true does not make much sense and causes errors that do not need to be there.
    The difference with the function at the link is that we assume the ranks of the tensors are the same, i.e.,
    we are not calling squeeze_or_expand_to_same_rank.
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_true.dtype)
    return tf.math.reduce_mean(tf.cast(tf.math.equal(y_true, y_pred), dtype=tf.float32))


  def update_state(self, y_true, y_pred, sample_weight=None):
    n = arrayops_cast(arrayops_shape(y_true, 0), dtype=y_pred.dtype)

    init = RCNNRecurrenceAccuracy.binary_accuracy(y_true[:,self.initial_round], y_pred[:,self.initial_round])
    self.init_accuracy.assign_add(init*n)
    if self.leadin_round is not None:
      leadin = RCNNRecurrenceAccuracy.binary_accuracy(y_true[:,self.leadin_round], y_pred[:,self.leadin_round])
      self.leadin_accuracy.assign_add(leadin*n)
    if self.has_recurrent_rounds:
      if self.recurrent_round_bound is not None:
        recurrent = RCNNRecurrenceAccuracy.binary_accuracy(y_true[:,self.leadin_round+1:self.recurrent_round_bound], y_pred[:,self.leadin_round+1:self.recurrent_round_bound])
      else:
        recurrent = RCNNRecurrenceAccuracy.binary_accuracy(y_true[:,self.leadin_round+1:], y_pred[:,self.leadin_round+1:])
      self.recurrent_accuracy.assign_add(recurrent*n)
    if self.final_round is not None:
      final = RCNNRecurrenceAccuracy.binary_accuracy(y_true[:,self.final_round], y_pred[:,self.final_round])
      self.final_accuracy.assign_add(final*n)

    self.counter.assign_add(n)
    
  
  def result(self):
    res = [ self.init_accuracy ]
    if self.leadin_round is not None:
      res.append(self.leadin_accuracy)
    if self.has_recurrent_rounds:
      res.append(self.recurrent_accuracy)
    if self.final_round is not None:
      res.append(self.final_accuracy)
    res = tf.stack(res, axis=0)/self.counter
    return res
  

  def reset_state(self):
    self.counter.assign(0.)
    self.init_accuracy.assign(0.)
    if self.leadin_round is not None:
      self.leadin_accuracy.assign(0.)
    if self.has_recurrent_rounds:
      self.recurrent_accuracy.assign(0.)
    if self.final_round is not None:
      self.final_accuracy.assign(0.)
