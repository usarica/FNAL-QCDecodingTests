import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric


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

    res = init
    if leadin is not None:
      res += leadin
    if recurrent is not None:
      res += recurrent
    if final is not None:
      res += final
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


  def update_state(self, y_true, y_pred, sample_weight=None):
    init = tf.keras.metrics.binary_accuracy(y_true[:,self.initial_round], y_pred[:,self.initial_round])
    self.init_accuracy.assign_add(init)
    if self.leadin_round is not None:
      leadin = tf.keras.metrics.binary_accuracy(y_true[:,self.leadin_round], y_pred[:,self.leadin_round])
      self.leadin_accuracy.assign_add(leadin)
    if self.has_recurrent_rounds:
      if self.recurrent_round_bound is not None:
        recurrent = tf.keras.metrics.binary_accuracy(y_true[:,self.leadin_round+1:self.recurrent_round_bound], y_pred[:,self.leadin_round+1:self.recurrent_round_bound])
      else:
        recurrent = tf.keras.metrics.binary_accuracy(y_true[:,self.leadin_round+1:], y_pred[:,self.leadin_round+1:])
      self.recurrent_accuracy.assign_add(recurrent)
    if self.final_round is not None:
      final = tf.keras.metrics.binary_accuracy(y_true[:,self.final_round], y_pred[:,self.final_round])
      self.final_accuracy.assign_add(final)
    
  
  def result(self):
    res = [ self.init_accuracy ]
    if self.leadin_round is not None:
      res.append(self.leadin_accuracy)
    if self.has_recurrent_rounds:
      res.append(self.recurrent_accuracy)
    if self.final_round is not None:
      res.append(self.final_accuracy)
    res = { wgt.name: wgt for wgt in res }
    return res
  

  def reset_states(self):
    self.init_accuracy.assign(0.)
    if self.leadin_round is not None:
      self.leadin_accuracy.assign(0.)
    if self.has_recurrent_rounds:
      self.recurrent_accuracy.assign(0.)
    if self.final_round is not None:
      self.final_accuracy.assign(0.)
