"""
These are a bunch of example functions to generate circuits.
Add more as needed.
"""

import stim # Quantum circuit simulator


def get_builtin_circuit(builtin_name, distance, rounds, **noise_args):
  """
  Wrapper around the stim.Circuit.generated function to generate a circuit.
  - Arguments:
    builtin_name: Name of the builtin circuit to generate.
    distance: Distance of the code.
    rounds: Number of rounds of the circuit.
    noise_args: Noise parameters to use.
  - Return type:
    stim.Circuit
  """
  return stim.Circuit.generated(
    code_task = builtin_name, distance = distance, rounds = rounds,
    **noise_args
  )

