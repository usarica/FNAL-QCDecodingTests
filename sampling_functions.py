"""
These are master functions to collect the MC samples using sinter and stim.
"""

import numpy as np
from matplotlib import pyplot as plt

import sinter # MC sampling
from typing import List # Organizing input to sinter

from circuit_generators import get_circuit_with_inserted_measurements


def create_task_sinter(fcn_circuit_generator, args_circuit_generator, **sinter_args):
  """
  Create a sinter task given a circuit generator function, its arguments, and any extra arguments to sinter.Task.
  - Arguments:
    fcn_circuit_generator: Function that generates the circuit. Could be any function that returns a stim.Circuit.
    args_circuit_generator: Arguments to the circuit generator.
    sinter_args: Arguments to the sinter task. A common example argument is json_metadata. Check sinter.Task help for more details.
  - Return type:
    sinter.Task
  - Example call:
    create_task_sinter(
      get_builtin_circuit,
      {"builtin_name": "repetition_code:memory", "distance": 5, "rounds": 3, "before_round_data_depolarization": 0.1},
      json_metadata = {"distance": 5, "error_rate": 0.1}
    )
  """
  return sinter.Task(
    circuit = fcn_circuit_generator(**args_circuit_generator),
    **sinter_args
  )


def generate_samples_sinter(num_workers, tasks, num_samples, decoders, **sinter_collect_args):
  """
  Run a sinter collect task given a number of workers, tasks, decoders, number of samples, and any extra arguments to sinter.collect.
  - Arguments:
    num_workers: Number of worker nodes.
    tasks: List of sinter.Task objects.
    decoders: List of decoders. Either use this argument, or set this to None with custom_decoders argument specified.
    num_samples: Number of samples to use.
    sinter_collect_args: Extra arguments to sinter.collect.
  - Return type:
    List[sinter.TaskStats]t
  - Example call:
    generate_samples_sinter(
      num_workers = 4,
      tasks = [task1, task2],
      decoders = ['pymatching'], # or set the custom_decoders argument
      num_samples = 1000000
    )
  """
  res: List[sinter.TaskStats] = sinter.collect(
    num_workers = num_workers,
    tasks = tasks,
    decoders = decoders,
    max_shots = num_samples,
    **sinter_collect_args
  )
  return res


def get_variable_from_sample_sinter(result, varname, varlabels=None):
  """
  Get a label from the result of the sinter.collect call.
  - Arguments:
    result: A sinter.TaskStats object.
    varname: Name of the label to get from the json_metadata, or some other object in the result data.
    varlabels: Label dictionary for the variable.
      If not None (default), the function attempts to map replace the entries with strings specified in this dictionary.
  - Return type:
    List
  - Example call:
    get_variable_from_sample_sinter(result, "distance")
  """
  res = None
  if hasattr(result, varname):
    res = getattr(result, varname)
  elif varname in result.json_metadata:
    res = result.json_metadata[varname]
  if res is None:
    raise RuntimeError(f"Label {varname} not found in the result.")
  if varlabels is not None:
    if res in varlabels:
      res = varlabels[res]
  return res


def plot_error_rate_sinter(
    ax, results,
    x_title, x_label=None, y_label=None,
    plot_label=None,
    group_title=None,
    group_labels=None,
    x_range=None, y_range=None,
    use_log_scale = True,
    **sinter_plotter_args
    ):
  """
  Plot the results from sinter.
  - Arguments:
    ax: Axes object to plot on.
    results: sinter.CollectResult object.
    x_title: Name of the json_metadata variable to use in the x-axis, or could be another entry in the results argument.
    x_label: Label for the x-axis. The default is to use x_title if set to None.
    y_label: Label for the y-axis. The default is to use y_title if set to None.
    plot_label: Title for the plot at the top. Defaulted to None for no plot title.
    group_title: Label for grouping the results. Defaulted to None for no grouping.
    group_labels: Labels dictionary of the grouping in the legend. Defaulted to None for no labeling.
      This argument is ignored if group_title is None.
    
    Note:
      The group_title and group_labels arguments are supposed to be passed as
        group_func = lambda st, tt=group_title, lbl=group_labels: get_variable_from_sample_sinter(st, tt, lbl)
      into the group_func argument of sinter.plot_error_rate.

    x_range: Range for the x-axis. Defaulted to None for no custom range.
    y_range: Range for the y-axis. Defaulted to None for no custom range.
    use_log_scale: Whether to use log scale for the axes. Defaulted to True.
    sinter_plotter_args: Extra arguments to the sinter plotter.
  - Return type:
    None
  - Example call:
    plot_error_rate_sinter(
      ax = ax,
      results = result,
      x_title = "noise",
      x_label = "Physical noise",
      y_label = "Logical errors per sample",
      plot_label = "Physical and logical error rates",
      group_title = "decoder"
      group_labels = {'pymatching': "pyMatching"}
    )
  """
  fcn_x = lambda st, tt=x_title: get_variable_from_sample_sinter(st, tt)
  fcn_group = lambda st, tt=group_title, lbl=group_labels: get_variable_from_sample_sinter(st, tt, lbl) if group_title is not None else None
  sinter.plot_error_rate(
    ax = ax,
    stats = results,
    x_func = fcn_x,
    group_func = fcn_group,
    **sinter_plotter_args
  )
  if x_range is not None:
    ax.set_xlim(x_range)
  if y_range is not None:
    ax.set_ylim(y_range)
  if plot_label is not None:
    ax.set_title(plot_label)
  if x_label is None:
    x_label = x_title
  ax.set_xlabel(x_label)
  if y_label is not None:
    ax.set_ylabel(y_label)
  if use_log_scale:
    ax.loglog()
  ax.grid(which='major')
  #ax.grid(which='minor')
  ax.legend()


class CircuitWithProjectiveErrors:
  """
  CircuitWithProjectiveErrors: Container class that holds a stim circuit and a modified version with projective errors.
  Projective errors are defined to be errors on the defined logical observables at each round, projected on the measurement basis.
  They are distinct from errors on the superposition of states and are probabilistic in nature.
  Hopefully, one is training a NN to model the statistical behavior of these errors, so there is no need for per-shot correctness.
  """

  def __init__(self, d, r, reference_circuit, seed=12345):
    """
    CircuitWithProjectiveErrors: Initialize the class for specific code distance and number of rounds, and with an unmodified circuit.
    - Arguments:
      d: Distance of the code.
      r: Number of rounds.
      reference_circuit: The unmodified stim.Circuit object.
    """
    self.d = d
    self.r = r
    self.reference_circuit = reference_circuit
    self.modified_circuit, self.obs_idxs = get_circuit_with_inserted_measurements(reference_circuit)
    self.converter = self.reference_circuit.compile_m2d_converter()
    self.seed = seed
    self.r_sampler = self.reference_circuit.compile_sampler(seed=seed)
    self.m_sampler = self.modified_circuit.compile_sampler(seed=seed)


  def sample(self, n_samples, binary_t = np.int8, newseed = None):
    if newseed is not None and newseed!=self.seed:
      self.seed = newseed
      self.m_sampler = self.modified_circuit.compile_sampler(seed=newseed)

    measurements_mod = self.m_sampler.sample(n_samples, bit_packed=False)

    idx_veto = [ i for i in range(2*self.d**2-1) ]
    for _ in range(self.r):
      idx_veto.extend([ i+idx_veto[-1]+self.d**2 for i in range(self.d**2) ])
    idx_filter = [ i for i in range(measurements_mod.shape[1]) if i not in idx_veto ]

    measurements = measurements_mod[:, idx_filter]
    det_evts, flips = self.converter.convert(measurements=measurements, separate_observables=True, bit_packed=False)
    det_evts = det_evts.astype(binary_t)

    measurements_mod_data = measurements_mod[:, idx_veto[self.d**2-1:]].reshape((-1, len(idx_veto[self.d**2-1:])//self.d**2, self.d**2)).astype(binary_t)
    measurements_mod_data = measurements_mod_data[:,:,self.obs_idxs]
    for ir in range(1, measurements_mod_data.shape[1]):
      measurements_mod_data[:,ir,:] = np.bitwise_xor(measurements_mod_data[:,ir,:],measurements_mod_data[:,0,:]).astype(binary_t)
    measurements_mod_data = measurements_mod_data[:,1:,:]

    flips_rest = np.sum(measurements_mod_data, axis=2) % 2
    flips = np.concatenate((flips_rest, flips), axis=1).astype(binary_t)

    return measurements, det_evts, flips
  

  def sample_reference(self, n_samples, binary_t = np.int8, newseed = None):
    if newseed is not None and newseed!=self.seed:
      self.seed = newseed
      self.r_sampler = self.reference_circuit.compile_sampler(seed=newseed)

    measurements = self.r_sampler.sample(n_samples, bit_packed=False)
    det_evts, flips = self.converter.convert(measurements=measurements, separate_observables=True, bit_packed=False)
    det_evts = det_evts.astype(binary_t)
    flips = flips.astype(binary_t)
    return measurements, det_evts, flips
