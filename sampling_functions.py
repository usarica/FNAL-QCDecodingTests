"""
These are master functions to collect the MC samples using sinter.
"""


import matplotlib
matplotlib.use('pgf')

from matplotlib import pyplot as plt

import sinter # MC sampling
from typing import List # Organizing input to sinter


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
    sinter.CollectResult
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
  plt.rcParams.update(
    {
      "font.family": "serif",
      "text.usetex": True,
      "pgf.rcfonts": False,
      "pgf.texsystem": 'pdflatex', # default is xetex
      "pgf.preamble": "\n".join(
        [
          "\\usepackage[T1]{fontenc}",
          "\\usepackage{mathpazo}",
          "\\usepackage{amsmath}",
          "\\usepackage{amssymb}"
        ]
      )
    }
  )

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
