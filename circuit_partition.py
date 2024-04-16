import numpy as np
from copy import deepcopy


def coord_to_idx(coords, d):
  return np.int8(coords[0] + (coords[1] - coords[0]%2)*(d+0.5))


def get_measure_qubits_ord(d):
  res = []
  for x in range(d+1):
    for y in range(d+1):
      on_boundary_1 = x == 0 or x == d
      on_boundary_2 = y == 0 or y == d
      parity = x % 2 != y % 2
      if (parity and on_boundary_1) or (not parity and on_boundary_2): continue
      coords = [2*x, 2*y]
      lbl = 'M:Z' if not parity else 'M:X'
      res.append([coord_to_idx(coords, d), lbl, coords])
  return res


def get_data_qubits_ord(d):
  res = []
  for x in range(d):
    for y in range(d):
      coords = [2*x+1, 2*y+1]
      is_ZL = y==0
      is_XL = x==0
      lbl = 'D'
      if is_ZL:
        lbl = lbl + ":ZL"
      if is_XL:
        lbl = lbl + ":XL"
      res.append([coord_to_idx(coords, d), lbl, coords])
  return res


def get_kernel_parity_flips(n_shifts, shift_x, shift_y):
  """
  get_kernel_parity_flips: Find out which edges are at the boundary,
                    flip the kernel along x or y if necessary,
                    and extract the standard boundary configuration of the resulting kernel.
  Arguments:
  - n_shifts: The number of shifts in the kernel
  - shift_x: The shift along the x-axis
  - shift_y: The shift along the y-axis
  Returns:
  - Whether the flipped configuration has a boundary on the left (x=+1) or right (x=-1), or top (y=+1) or bottom (y=-1)
  - Whether the kernel is flipped along x or y
  """
  flip_x = ((shift_x + shift_y)%2)==1
  flip_y = False
  parity_x = (1 if shift_x==0 else (-1 if shift_x==n_shifts-1 else 0))
  if flip_x:
    parity_x = -parity_x
  parity_y = (1 if shift_y==0 else (-1 if shift_y==n_shifts-1 else 0))
  if (parity_x==0 and parity_y==-1) \
    or (parity_x==1 and parity_y==-1) \
    or (parity_x==-1 and parity_y!=1):
      flip_x = not flip_x
      flip_y = not flip_y
      parity_x = -parity_x
      parity_y = -parity_y
  return parity_x, parity_y, flip_x, flip_y


def get_unique_kernel_types(k, d):
  res = []
  n_shifts = d-k+1
  for shift_y in range(n_shifts):
    for shift_x in range(n_shifts):
      parity_x, parity_y, _, _ = get_kernel_parity_flips(n_shifts, shift_x, shift_y)
      has_parity = False
      for pp in res:
        if pp[0][0]==parity_x and pp[0][1]==parity_y:
          has_parity = True
          pp[1].append(shift_x + shift_y*n_shifts)
          break
      if not has_parity:
        res.append([[parity_x, parity_y], [shift_x + shift_y*n_shifts]])
  return res


def shift_frame(data_qubits_kxk, measure_qubits_kxk, k, d, shift_x, shift_y):
  _, _, flip_x, flip_y = get_kernel_parity_flips(d-k+1, shift_x, shift_y)
  res_data_qubits_kxk = deepcopy(data_qubits_kxk)
  res_measure_qubits_kxk = deepcopy(measure_qubits_kxk)
  for q in res_data_qubits_kxk:
    q[2][0] = (q[2][0] if not flip_x else 2*k - q[2][0]) + 2*shift_x
    q[2][1] = (q[2][1] if not flip_y else 2*k - q[2][1]) + 2*shift_y
  for q in res_measure_qubits_kxk:
    q[2][0] = (q[2][0] if not flip_x else 2*k - q[2][0]) + 2*shift_x
    q[2][1] = (q[2][1] if not flip_y else 2*k - q[2][1]) + 2*shift_y
  return res_data_qubits_kxk, res_measure_qubits_kxk


def group_det_bits_kxk(det_bits_dxd, d, r, k, use_rotated_z, data_bits_dxd=None, binary_t=np.int8, idx_t=np.int8):
  """
  group_det_bits_kxk: Group the (d^2-1) detector bits into groups of size (k^2-1) for a kxk subset of the dxd surfsce code.
  Arguments:
  - det_bits_dxd: The measure bits for the dxd surface code as ordered by stim
  - d: The distance of the surface code
  - r: The number of rounds
  - k: The size of the subset of the surface code
  - use_rotated_z: Whether to use the ZL or XL boundaries in the rotated surface code
  - data_bits_dxd: The data bits for the dxd surface code as ordered by stim (optional)
  - binary_t: The data type for the bits
  - idx_t: The data type for the indices
  Returns:
  In order of appearance:
  - A 3D array of shape=[(d-k+1)^2, number of samples, (k^2-1)] for the detector bits
    that would be used in each kernel with each 2D [,:,:] subset ordered in a way consistent with stim
  - A 3D array of shape=[(d-k+1)^2, number of samples, k^2] for the data bits
  - A 3D array of shape=[(d-k+1)^2, number of samples, k] for the logical observables of each kernel
  - A 3D array of shape=[(d-k+1)^2, number of samples, r] for the partial translation map
    of the kernel flip predictions to the original logical observable in the dxd surface code
  """
  # Number of kernels along each dimension
  n_shifts = d-k+1

  det_bits_kxk_all = []
  data_bits_kxk_all = []
  obs_bits_kxk_all = []
  kernel_result_translation_map = np.zeros(shape=(n_shifts**2, det_bits_dxd.shape[0], r), dtype=binary_t)

  # Measure qubits
  measure_qubits_ord_dxd = get_measure_qubits_ord(d)
  measure_qubits_ord_kxk = get_measure_qubits_ord(k)
  n_measure_dxd = len(measure_qubits_ord_dxd)
  n_measure_kxk = len(measure_qubits_ord_kxk)

  # Data qubits
  data_qubits_ord_dxd = get_data_qubits_ord(d)
  data_qubits_ord_kxk = get_data_qubits_ord(k)
  n_data_dxd = len(data_qubits_ord_dxd)
  n_data_kxk = len(data_qubits_ord_kxk)

  # Ordering of qubits by measurement sequence:
  # Measurement sequence is the same as the ordered sequence of qubit indices in stim
  # Measure qubits are measured r times, and then data qubits are measured once.
  measurement_sequence_data_dxd = sorted(data_qubits_ord_dxd)
  measurement_sequence_measure_dxd = sorted(measure_qubits_ord_dxd)

  for shift_y in range(n_shifts):
    for shift_x in range(n_shifts):
      idx_kernel = shift_x + shift_y*n_shifts
      # Get the filtering indices to translate kxk kernel results to the dxd ZL or XL boundaries
      parity_x, parity_y, flip_x, flip_y = get_kernel_parity_flips(n_shifts, shift_x, shift_y)
      is_top = (parity_y!=-1 and not flip_y)
      is_left = (parity_x!=-1 and not flip_x)
      x_boundary = 1 + 2*shift_x + (0 if is_left else 2*k)
      y_boundary = 1 + 2*shift_y + (0 if is_top else 2*k)
      measure_boundary_filter = []
      for m in range(n_measure_dxd):
        if (use_rotated_z and measurement_sequence_measure_dxd[m][2][1]<y_boundary and measurement_sequence_measure_dxd[m][1]=='M:Z') \
          or (not use_rotated_z and measurement_sequence_measure_dxd[m][2][0]<x_boundary and measurement_sequence_measure_dxd[m][1]=='M:X'):
          measure_boundary_filter.append(m)
      measure_boundary_filter.extend([ idx_m + rr*n_measure_dxd for idx_m in measure_boundary_filter for rr in range(1, r)])
      measure_boundary_filter = np.array(measure_boundary_filter, dtype=idx_t)
      measure_boundary_filter = measure_boundary_filter.reshape(r, len(measure_boundary_filter)//r)
      for rr in range(r):
        kernel_result_translation_map[idx_kernel, :, rr] = np.sum(det_bits_dxd[:, measure_boundary_filter[rr]], axis=1)

      data_qubits_kxk, measure_qubits_kxk = shift_frame(data_qubits_ord_kxk, measure_qubits_ord_kxk, k, d, shift_x, shift_y)

      measurement_sequence_measure_qubits_kxk = sorted(measure_qubits_kxk)
      measurement_sequence_data_qubits_kxk = sorted(data_qubits_kxk)

      # Find the index of the measure qubit in the dxd surface code through the position map
      det_bits_kxk_idx_filter = np.zeros(r*n_measure_kxk, dtype=idx_t)
      for i in range(n_measure_kxk):
        pos_q_kxk = measurement_sequence_measure_qubits_kxk[i][2]
        found_match = False
        for j in range(n_measure_dxd):
          if np.array_equal(measurement_sequence_measure_dxd[j][2], pos_q_kxk):
            found_match = True
            for iround in range(r):
              det_bits_kxk_idx_filter[i + iround*n_measure_kxk] = j + iround*n_measure_dxd
            break
        if not found_match:
          raise RuntimeError(f"Measure qubit with position ({pos_q_kxk[0], pos_q_kxk[1]}) not found in the dxd surface code.")
      # Filter det_bits
      det_bits_kxk_all.append(det_bits_dxd[:, det_bits_kxk_idx_filter])
      
      # Find the index of the data qubit in the dxd surface code through the position map
      if data_bits_dxd is not None:
        obs_bits_kxk_idx_filter = np.zeros(k, dtype=idx_t)
        data_bits_kxk_idx_filter = np.zeros(k**2, dtype=idx_t)
        iobs = 0
        for i in range(n_data_kxk):
          pos_q_kxk = measurement_sequence_data_qubits_kxk[i][2]
          lbl_kxk = measurement_sequence_data_qubits_kxk[i][1]
          found_match = False
          for j in range(n_data_dxd):
            if np.array_equal(measurement_sequence_data_dxd[j][2], pos_q_kxk):
              found_match = True
              data_bits_kxk_idx_filter[i] = j
              if ('XL' in lbl_kxk and not use_rotated_z) or ('ZL' in lbl_kxk and use_rotated_z):
                obs_bits_kxk_idx_filter[iobs] = j
                iobs = iobs + 1
              break
          if not found_match:
            raise RuntimeError(f"Data qubit with position ({pos_q_kxk[0], pos_q_kxk[1]}) not found in the dxd surface code.")
        # Filter data_measurements
        data_bits_kxk_all.append(data_bits_dxd[:, data_bits_kxk_idx_filter])
        # Reverse the order of the obs_bits filter to match stim conventions
        obs_bits_kxk_idx_filter = np.flip(obs_bits_kxk_idx_filter, axis=0)
        obs_bits_kxk_all.append(data_bits_dxd[:, obs_bits_kxk_idx_filter])
  kernel_result_translation_map = kernel_result_translation_map % 2
  return np.array(det_bits_kxk_all, dtype=binary_t), np.array(data_bits_kxk_all, dtype=binary_t), np.array(obs_bits_kxk_all, dtype=binary_t), kernel_result_translation_map
