from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Initializer
from circuit_partition import *
from types_cfg import *
from utilities_arrayops import *


def get_layer_input_maps(n_ancillas, npol, is_symmetric):
  diag_param_map = None
  nondiag_param_map = None
  triangular_polmap = None
  triangular_backpolmap = None

  if npol>1:
    diag = []
    diag_wmap = []
    nondiag = []
    nondiag_wmap = []
    if not is_symmetric:
      for iy in range(n_ancillas):
        for ix in range(iy, n_ancillas):
          if ix!=iy:
            nondiag.append(ix+iy*n_ancillas)
          else:
            diag.append(ix+iy*n_ancillas)
      diag_param_map = [ ii for ii in range(len(diag)) ]
      nondiag_param_map = [ ii for ii in range(len(nondiag)) ]
    else:
      for iy in range(n_ancillas):
        oy = iy if iy<n_ancillas//2 else n_ancillas-1-iy
        flipped_y = (iy!=oy)
        for ix in range(iy, n_ancillas):
          ox = ix if ix<n_ancillas//2 else n_ancillas-1-ix
          flipped_x = (ix!=ox)
          if ix!=iy:
            nondiag.append(ix+iy*n_ancillas)
            jx = ix
            jy = iy
            if flipped_y == flipped_x:
              jx = ox
              jy = oy
            elif ox<iy:
              jy = ox
              jx = n_ancillas-1-iy
            if jy>jx:
              jx, jy = jy, jx
            nondiag_wmap.append(jx+jy*n_ancillas)
          else:
            diag.append(ix+iy*n_ancillas)
            diag_wmap.append(ox+oy*n_ancillas)
      tmp_wmap = []
      idx = 0
      jj = 0
      for ii in diag_wmap:
        if ii not in tmp_wmap:
          tmp_wmap.append(ii)
          diag_wmap[jj] = idx
          idx += 1
        else:
          diag_wmap[jj] = tmp_wmap.index(ii)
        jj += 1
      diag_param_map = diag_wmap
      tmp_wmap = []
      idx = 0
      jj = 0
      for ii in nondiag_wmap:
        if ii not in tmp_wmap:
          tmp_wmap.append(ii)
          nondiag_wmap[jj] = idx
          idx += 1
        else:
          nondiag_wmap[jj] = tmp_wmap.index(ii)
        jj += 1
      nondiag_param_map = nondiag_wmap
      del tmp_wmap

    triangular_polmap = diag
    triangular_polmap.extend(nondiag)
    triangular_polmap_sorted = deepcopy(triangular_polmap)
    triangular_polmap_sorted.sort()
    triangular_backpolmap = [ triangular_polmap.index(ii) for ii in triangular_polmap_sorted ]
  else:
    if not is_symmetric:
      diag_param_map = [ ii for ii in range(n_ancillas) ]
    else:
      diag_param_map = [ ii for ii in range(n_ancillas//2) ]
      diag_param_map.extend([ n_ancillas//2-1-ii for ii in range(n_ancillas//2) ])
  return diag_param_map, nondiag_param_map, triangular_polmap, triangular_backpolmap


def get_layer_output_map(d, is_symmetric):
  if not is_symmetric:
    return [ ii for ii in range(d**2) ]
  else:
    res = []
    for iy in range(d):
      for ix in range(d):
        ox = ix
        oy = iy
        if not (ix>iy or (ix==iy and ix<(d+1)//2)):
          ox = (d-ix-1)
          oy = (d-iy-1)
        idx_kqubit = ox+oy*d - (oy*(oy+1))//2
        if oy>=(d+1)//2:
          idx_kqubit = idx_kqubit - (oy-(d+1)//2+1)
        res.append(idx_kqubit)
    return res


def get_detector_bits_perround_map(d, r, npol, is_symmetric, ignore_diagonal=False):
  """
  get_detector_bits_perround_map: Get the mapping of the detector bits to the output qubits.
  If we have a linear map
  Detector bits (i) [i<Ndb] -> W_ij -> Data qubit (j) [j<Ndq]
  with npol=1,
  the result will be the list
  [[ j, [{0, ... , Ndb-1} x rounds] ]]
  for each j<Ndq, except if is_symmetric is True and j is in the second, symmetrized half of the data qubits.
  In that case, the list would contain
  [[ j_sym, [{Ndb-1, ... , 0} x rounds]]]
  for j_sym being the index of the symmetrized data qubit.
  If npol>1, we assume a quadratic form between detector bits i1 and i2, but the rules to reverse or not are the same as above.
  """
  if ignore_diagonal and npol==1:
    raise ValueError("Ignoring the diagonal terms is only possible for npol>1.")

  n_ancillas = d**2-1
  outmap = get_layer_output_map(d, is_symmetric)
  res = []
  ioffset = 0 if not ignore_diagonal else 1

  det_bits_unswap_map = None
  if npol>1:
    det_bits_unswap_map = []
    for iy in range(n_ancillas*r):
      for ix in range(iy + ioffset, n_ancillas*r):
        det_bits_unswap_map.append(ix+iy*n_ancillas*r - ((iy+ioffset)*(iy+1+ioffset))//2)
  else:
    det_bits_unswap_map = [ q for q in range(n_ancillas*r) ]

  if is_symmetric:
    det_bits_swap_map = []
    if npol>1:
      for iiy in range(n_ancillas*r):
        rry = iiy//n_ancillas
        iy = iiy%n_ancillas
        for iix in range(iiy + ioffset, n_ancillas*r):
          rrx = iix//n_ancillas
          ix = iix%n_ancillas
          oy = (n_ancillas-1-iy) + rry*n_ancillas
          ox = (n_ancillas-1-ix) + rrx*n_ancillas
          if ox<oy:
            ox, oy = oy, ox
          det_bits_swap_map.append(ox+oy*n_ancillas*r - ((oy+ioffset)*(oy+1+ioffset))//2)
    else:
      for rr in range(r):
        det_bits_swap_map.extend([ n_ancillas-1-q + rr*n_ancillas for q in range(n_ancillas) ])
    for iy in range(d):
      for ix in range(d):
        is_swapped = not (ix>iy or (ix==iy and ix<(d+1)//2))
        jout = outmap[iy*d+ix]
        ilist = det_bits_unswap_map
        if is_swapped:
          ilist = det_bits_swap_map
        res.append([ jout, ilist ])
  else:
    for iy in range(d):
      for ix in range(d):
        jout = outmap[iy*d+ix]
        res.append([ jout, det_bits_unswap_map ])
  return res


def get_states_perround_map(d, r, npol, is_symmetric):
  """
  get_states_perround_map: Get the mapping of the states to the output qubits.
  If we have a linear map
  States (i) [i<Ndq] -> W_ij -> Data qubit (j) [j<Ndq]
  with npol=1,
  the result will be the list
  [[ j, [{0, ... , Ndq-1} x rounds] ]]
  for each j<Ndq, except if is_symmetric is True and j is in the second, symmetrized half of the data qubits.
  In that case, the list would contain
  [[ j_sym, [{Ndq-1, ... , 0} x rounds]]]
  for j_sym being the index of the symmetrized data qubit.
  If npol>1, we assume a quadratic form between states i1 and i2, but the rules to reverse or not are the same as above.
  """
  n_states = d**2
  outmap = get_layer_output_map(d, is_symmetric)
  res = []

  states_unswap_map = None
  if npol>1:
    states_unswap_map = []
    for iy in range(n_states*r):
      for ix in range(iy, n_states*r):
        states_unswap_map.append(ix+iy*n_states*r - (iy*(iy+1))//2)
  else:
    states_unswap_map = [ q for q in range(n_states*r) ]

  if is_symmetric:
    states_swap_map = []
    if npol>1:
      for iiy in range(n_states*r):
        rry = iiy//n_states
        iy = iiy%n_states
        for iix in range(iiy, n_states*r):
          rrx = iix//n_states
          ix = iix%n_states
          oy = (n_states-1-iy) + rry*n_states
          ox = (n_states-1-ix) + rrx*n_states
          if ox<oy:
            ox, oy = oy, ox
          states_swap_map.append(ox+oy*n_states*r - (oy*(oy+1))//2)
    else:
      for rr in range(r):
        states_swap_map.extend([ n_states-1-q + rr*n_states for q in range(n_states) ])
    for iy in range(d):
      for ix in range(d):
        is_swapped = not (ix>iy or (ix==iy and ix<(d+1)//2))
        jout = outmap[iy*d+ix]
        ilist = states_unswap_map
        if is_swapped:
          ilist = states_swap_map
        res.append([ jout, ilist ])
  else:
    for iy in range(d):
      for ix in range(d):
        jout = outmap[iy*d+ix]
        res.append([ jout, states_unswap_map ])
  return res


def get_triplet_states_perround_map(d, r, npol, is_symmetric, ignore_diagonal=False):
  """
  get_triplet_states_perround_map: Get the mapping of the input triplet states to the output qubits.
  If we have a linear map
  Input (i) [i<Ndb] -> W_ij -> Data qubit (j) [j<Ndq]
  with npol=1,
  the result will be the list
  [[ j, [{0, ... , Ndb-1} x rounds] ]]
  for each j<Ndq, except if is_symmetric is True and j is in the second, symmetrized half of the data qubits.
  In that case, the list would contain
  [[ j_sym, [{Ndb-1, ... , 0} x rounds]]]
  for j_sym being the index of the symmetrized data qubit.
  If npol>1, we assume a quadratic form between inputs i1 and i2, but the rules to reverse or not are the same as above.

  Note that the parameter r is the number of triplet state rounds, i.e., the actual number of rounds is r+2.
  """
  map_single_round = get_detector_bits_perround_map(d, 1, npol, is_symmetric, ignore_diagonal)
  res = []
  for jout, ilist in map_single_round:
    nlist = deepcopy(ilist)
    for rr in range(1, r):
      nlist.expand([ q + rr*len(ilist) for q in ilist ])
    res.append([ jout, nlist ])
  return res


def get_detector_evts_perround_map(d, r, n_remove_last_dets, npol, is_symmetric, ignore_first_round=False, ignore_diagonal=False):
  """
  get_detector_evts_perround_map: Get the mapping of the detector events to the output qubits.
  If we have a linear map
  Detector event (i) [i<Nde] -> W_ij -> Data qubit (j) [j<Ndq]
  with npol=1,
  the result will be the list
  [[ j, [{0, ... , Nde-1} x rounds] ]]
  for each j<Ndq, except if is_symmetric is True and j is in the second, symmetrized half of the data qubits.
  In that case, the list would contain
  [[ j_sym, [{0, ... , Nde-1}_reverse x rounds]]]
  for the list {0, ... , Nde-1}_reverse that corresponds to a reverse-evaluation of the detector events in each round,
  and j_sym is the index of the symmetrized data qubit.
  Note that the first round contains (d^2-1)/2 detector events, so that has special treatment.
  If npol>1, we assume a quadratic form between detector events i1 and i2, but the rules to reverse or not are the same as above.
  """
  if ignore_diagonal and npol==1:
    raise ValueError("Ignoring the diagonal terms is only possible for npol>1.")

  n_ancillas = d**2-1
  outmap = get_layer_output_map(d, is_symmetric)
  res = []
  ioffset = 0 if not ignore_diagonal else 1

  ndim = n_ancillas*r - n_remove_last_dets
  if ignore_first_round:
    ndim -= n_ancillas//2

  det_evts_unswap_map = None
  if npol>1:
    det_evts_unswap_map = []
    for iy in range(ndim):
      for ix in range(iy+ioffset, ndim):
        det_evts_unswap_map.append(ix+iy*ndim - ((iy+ioffset)*(iy+1+ioffset))//2)
  else:
    det_evts_unswap_map = [ q for q in range(ndim) ]

  if is_symmetric:
    det_evts_swap_map = []

    qorder = [] if ignore_first_round else [ [q, 0] for q in range(n_ancillas//2) ]
    for rr in range(1,r):
      qorder.extend([ [q, rr] for q in range(n_ancillas) ])
    if n_remove_last_dets<n_ancillas//2:
      qorder.extend([ [q, r] for q in range(n_ancillas//2 - n_remove_last_dets) ])
    for iyord in range(len(qorder)):
      ry = qorder[iyord][1]
      iy = qorder[iyord][0]
      oy = n_ancillas-1-iy if (ry>0 and ry<r) else n_ancillas//2-1-iy
      for ixord in range(iyord+ioffset, len(qorder)):
        rx = qorder[ixord][1]
        ix = qorder[ixord][0]
        ox = n_ancillas-1-ix if (rx>0 and rx<r) else n_ancillas//2-1-ix

        roy = None
        rox = None
        for iqord, qord in enumerate(qorder):
          if qord == [oy, ry]:
            roy = iqord
          if qord == [ox, rx]:
            rox = iqord
          if roy is not None and rox is not None:
            break
        if roy is None or rox is None:
          raise RuntimeError(f"Could not find the indices for qubits {oy, ry} and {ox, rx} in the order list ({qorder}).")
        if rox<roy:
          rox, roy = roy, rox
        det_evts_swap_map.append(rox+roy*ndim - ((roy+ioffset)*(roy+1+ioffset))//2)

    for iy in range(d):
      for ix in range(d):
        is_swapped = not (ix>iy or (ix==iy and ix<(d+1)//2))
        jout = outmap[ix+iy*d]
        ilist = det_evts_unswap_map
        if is_swapped:
          ilist = det_evts_swap_map
        res.append([ jout, ilist ])

  else:
    for iy in range(d):
      for ix in range(d):
        jout = outmap[iy*d+ix]
        res.append([ jout, det_evts_unswap_map ])

  return res


def get_detector_bits_perround_embedding_map(d, r, npol, is_symmetric):
  """
  get_detector_bits_perround_embedding_map: Get the embedding mapping of the detector bits.
  If npol>1, we assume a quadratic form between detector bits i1 and i2, but the rules to reverse or not are the same as above.
  """
  if npol==1:
    return None

  n_ancillas = d**2-1
  ndims = n_ancillas*r
  res = []
  if not is_symmetric:
    ipar = 0
    dpar = -1
    for yr in range(r):
      for iy in range(n_ancillas):
        oy = iy + yr*n_ancillas
        for xr in range(r):
          for ix in range(n_ancillas):
            ox = ix + xr*n_ancillas
            if ox<oy:
              continue
            if ox==oy:
              res.append(dpar)
              dpar -= 1
            else:
              res.append(ipar)
              ipar += 1
  else:
    idx_list = []
    idx_dlist = []
    for yr in range(r):
      for iy in range(n_ancillas):
        oy = iy + yr*n_ancillas
        jy = iy if iy<n_ancillas//2 else n_ancillas-1-iy
        flipped_y = (iy!=jy)
        for xr in range(r):
          for ix in range(n_ancillas):
            ox = ix + xr*n_ancillas
            jx = ix if ix<n_ancillas//2 else n_ancillas-1-ix
            flipped_x = (ix!=jx)

            if ox<oy:
              continue

            kx = jx
            ky = jy
            if ix!=iy:
              kx = ix
              ky = iy
              if flipped_y == flipped_x:
                kx = jx
                ky = jy
              elif jx<iy:
                ky = jx
                kx = n_ancillas-1-iy
              if ky>kx:
                kx, ky = ky, kx
              
            kx = kx + xr*n_ancillas
            ky = ky + yr*n_ancillas

            idx_entry = kx + ky*ndims
            if ox==oy:
              if idx_entry not in idx_dlist:
                res.append(-1-len(idx_dlist))
                idx_dlist.append(idx_entry)
              else:
                res.append(-1-idx_dlist.index(idx_entry))
            else:
              if idx_entry not in idx_list:
                res.append(len(idx_list))
                idx_list.append(idx_entry)
              else:
                res.append(idx_list.index(idx_entry))
  return res


def get_detector_evts_perround_embedding_map(d, r, obs_type, npol, is_symmetric, ignore_first_round, ignore_last_round):
  """
  get_detector_evts_perround_embedding_map: Get the embedding mapping of the detector events.
  If npol>1, we assume a quadratic form between detector bits i1 and i2, but the rules to reverse or not are the same as above.
  """
  if ignore_first_round and ignore_last_round:
    return get_detector_bits_perround_embedding_map(d, r-1, npol, is_symmetric)

  if npol==1:
    return None

  measure_qubits_ord = get_measure_qubits_ord(d)
  measurement_sequence_measure = sorted(measure_qubits_ord)
  to_std_map = dict()
  imq = 0
  for mq in measure_qubits_ord:
    if not ((obs_type=="ZL" and mq[1]=="M:Z") or (obs_type=="XL" and mq[1]=="M:X")):
      continue
    for iomq, omq in enumerate(measurement_sequence_measure):
      if mq[0] == omq[0]:
        to_std_map[imq] = iomq
        break
    imq += 1

  n_ancillas = d**2-1
  rmin = 0
  rmax = r+1
  if ignore_first_round:
    rmin = 1
  if ignore_last_round:
    rmax -= 1
  ndims = n_ancillas*(rmax-rmin)
  res = []
  if not is_symmetric:
    dpar = -1
    ipar = 0
    inc_yrna = 0
    for yr in range(rmin, rmax):
      special_yr = (yr==0 or yr==r)
      nay = n_ancillas//2 if special_yr else n_ancillas
      for iiy in range(nay):
        oy = iiy + inc_yrna
        inc_xrna = 0
        for xr in range(rmin, rmax):
          special_xr = (xr==0 or xr==r)
          nax = n_ancillas//2 if special_xr else n_ancillas
          for iix in range(nax):
            ox = iix + inc_xrna
            if ox<oy:
              continue
            if ox==oy:
              res.append(dpar)
              dpar -= 1
            else:
              res.append(ipar)
              ipar += 1
          inc_xrna += nax
      inc_yrna += nay
  else:
    idx_dlist = []
    idx_list = []
    inc_yrna = 0
    for yr in range(rmin, rmax):
      special_yr = (yr==0 or yr==r)
      nay = n_ancillas//2 if special_yr else n_ancillas
      for iiy in range(nay):
        inc_xrna = 0
        for xr in range(rmin, rmax):
          special_xr = (xr==0 or xr==r)
          nax = n_ancillas//2 if special_xr else n_ancillas
          for iix in range(nax):
            oy = iiy + inc_yrna
            ox = iix + inc_xrna
            if ox<oy:
              continue

            iy = iiy if not special_yr else to_std_map[iiy]
            yrr = yr
            ix = iix if not special_xr else to_std_map[iix]
            xrr = xr
            if ix + inc_xrna < iy + inc_yrna:
              ix, iy = iy, ix
              xrr, yrr = yrr, xrr

            jy = iy if iy<n_ancillas//2 else n_ancillas-1-iy
            flipped_y = (iy!=jy)

            jx = ix if ix<n_ancillas//2 else n_ancillas-1-ix
            flipped_x = (ix!=jx)

            kx = jx
            ky = jy
            if ix!=iy:
              kx = ix
              ky = iy
              if flipped_y == flipped_x:
                kx = jx
                ky = jy
              elif jx<iy:
                ky = jx
                kx = n_ancillas-1-iy
              if ky>kx:
                kx, ky = ky, kx
              
            kx = kx + xrr*n_ancillas
            ky = ky + yrr*n_ancillas

            idx_entry = kx + ky*ndims
            if ox==oy:
              if idx_entry not in idx_dlist:
                res.append(-1-len(idx_dlist))
                idx_dlist.append(idx_entry)
              else:
                res.append(-1-idx_dlist.index(idx_entry))
            else:
              if idx_entry not in idx_list:
                res.append(len(idx_list))
                idx_list.append(idx_entry)
              else:
                res.append(idx_list.index(idx_entry))
          inc_xrna += nax
      inc_yrna += nay
  return res


def get_detector_bit_state_relation_map(d, r, npol, is_symmetric):
  nearest_data_qubit_idxmap = get_data_qubits_next_to_measure_qubits(d)
  n_ancillas = d**2-1
  ndims = n_ancillas*r
  res = []
  for iy in range(ndims):
    rry = iy//n_ancillas
    iiy = iy%n_ancillas
    dqs_y = nearest_data_qubit_idxmap[iiy]
    if npol==1:
      res.append(dqs_y)
      continue
    for ix in range(iy, ndims):
      rrx = ix//n_ancillas
      iix = ix%n_ancillas
      dqs_x = nearest_data_qubit_idxmap[iix]
      res.append(sorted(set(dqs_y).union(dqs_x)))
  return res



class DetectorBitStateEmbedder(Layer):
  """
  DetectorBitStateEmbedder: Convert binary detector bit data into a linear (npol=1) or quadratic (npol>1) form.
  If npol=1, the returned vector is just the original input.
  If npol>1, there are only two possible values for the state of diagonal terms: 0x0 -> -1, and 1x1-> 1. They are returned as is.
  For non-diagonal terms, there are 3 possible values: 0x0 -> -1x-1, 0x1/1x0 -> -1x1/1x-1, and 1x1 -> 1x1.
  The state -1x1/1x-1 is mapped to a parameter p, and the states -1x-1 and 1x1 are mapped to -1.
  """
  def __init__(
      self,
      distance,
      rounds,
      is_symmetric,
      npol,
      ignore_diagonal,
      **kwargs
    ):
    super(DetectorBitStateEmbedder, self).__init__(**kwargs)
    self.distance = distance
    self.rounds = rounds
    self.is_symmetric = is_symmetric
    self.npol = npol
    self.ignore_diagonal = ignore_diagonal
    self.ndims = (self.distance**2 - 1)*self.rounds

    self.embedder_label = f"DetectorBitStateEmbedder_npol{self.npol}"

    self.embedding_param_map = get_detector_bits_perround_embedding_map(self.distance, self.rounds, self.npol, self.is_symmetric)
    self.nondiag_param_map = None
    if self.embedding_param_map is not None:
      self.nondiag_param_map = [ i for i in self.embedding_param_map if i>=0 ]
    self.n_embedding_params = 0
    self.embedding_params_nondiag = None
    if self.nondiag_param_map is not None:
      for i in self.nondiag_param_map:
        if i>self.n_embedding_params:
          self.n_embedding_params = i
      self.n_embedding_params += 1
      self.embedding_params_nondiag = self.add_weight(
        name=f"{self.embedder_label}_params_nondiag",
        shape=[ 1, self.n_embedding_params ], # States -1x1/1x-1 need a probability assignment
        initializer='zeros',
        trainable=True
      )

    self.triangular_polmap = None
    self.triangular_backpolmap = None
    if self.npol>1:
      triangular_polmap_diag = []
      triangular_polmap_nondiag = []
      for iy in range(self.ndims):
        for ix in range(iy, self.ndims):
          if ix==iy:
            triangular_polmap_diag.append(ix+iy*self.ndims)
          else:
            triangular_polmap_nondiag.append(ix+iy*self.ndims)
      if not self.ignore_diagonal:
        self.triangular_polmap = triangular_polmap_diag
        self.triangular_polmap.extend(triangular_polmap_nondiag)
        triangular_polmap_sorted = sorted(self.triangular_polmap)
        self.triangular_backpolmap = [ self.triangular_polmap.index(ii) for ii in triangular_polmap_sorted ]
      else:
        self.triangular_polmap = triangular_polmap_nondiag
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "distance": self.distance,
        "rounds": self.rounds,
        "is_symmetric": self.is_symmetric,
        "npol": self.npol,
        "ignore_diagonal": self.ignore_diagonal
      }
    )
    return config


  def embed_pol_state(self, res_diag, res_nondiag, embedding_nondiag_tr):
    # At this point, all terms have values (1, 4 | 2).
    # Non-diagonal terms
    res_nondiag = res_nondiag - 2 # (1, 2, 4) -> (-1, 0, 2)
    res_nondiag_sq = res_nondiag*res_nondiag # (-1, 0, 2) -> (1, 0, 4), need 3 bits + 1 for the sign
    res_nondiag_ones = tf.ones_like(res_nondiag)
    res_nondiag = tf.stack([res_nondiag_sq, res_nondiag, res_nondiag_ones], axis=2) # { x^2, x, 1 }
    res_nondiag = tf.cast(res_nondiag, tf.float32)
    res_nondiag = tf.reduce_sum(res_nondiag*embedding_nondiag_tr, axis=2)
    # Diagonal terms
    if res_diag is not None:
      res_diag = (2*(res_diag-2)-1)/3 # (1, 4) -> (-1, 1).
      res_diag = tf.cast(res_diag, tf.float32)
      res = tf.concat([res_diag, res_nondiag], axis=1)
      res = tf.gather(res, self.triangular_backpolmap, axis=1)
    else:
      res = res_nondiag
    return res


  def get_transformed_state(self, x, embedding_nondiag_tr):
    res = None
    if self.npol==1:
      res = tf.cast(x, tf.float32)
    else:
      xx = x + 1
      xsh = arrayops_shape(x)
      res = tf.matmul(
        tf.cast(tf.reshape(xx, shape=(xsh[0], xsh[1], 1)), tf.int16),
        tf.cast(tf.reshape(xx, shape=(xsh[0], 1, xsh[1])), tf.int16)
      )
      res = tf.gather(tf.reshape(res, shape=(xsh[0], -1)), self.triangular_polmap, axis=1)

      res_diag = None
      res_nondiag = None
      if self.ignore_diagonal:
        res_nondiag = res
      else:
        res_diag = res[:,0:self.ndims]
        res_nondiag = res[:,self.ndims:]
      # res has values (1, 4 | 2) now
      res = self.embed_pol_state(res_diag, res_nondiag, embedding_nondiag_tr)
    return res


  def get_transformed_embedding_params(self, n):
    if self.npol>1:
      params_nondiag = tf.math.sigmoid(VariableBounds.clip_zlike(self.embedding_params_nondiag))
      #params_nondiag = tf.math.sigmoid(self.embedding_params_nondiag)
      params_nondiag = tf.gather(params_nondiag, self.nondiag_param_map, axis=1)
      params_nondiag_p1o2 = (params_nondiag + 1)/2
      params_nondiag = tf.stack(
        [
          -params_nondiag_p1o2,
          params_nondiag_p1o2,
          params_nondiag
        ],
        axis=2
      )
      return tf.repeat(params_nondiag, n, axis=0)
    return None


  def call(self, input):
    n = arrayops_shape(input, 0)
    embedding_nondiag_tr = self.get_transformed_embedding_params(n)
    return self.get_transformed_state(input, embedding_nondiag_tr)



class DetectorEventStateEmbedder(Layer):
  """
  DetectorEventStateEmbedder: Convert binary detector event data into a linear (npol=1) or quadratic (npol>1) form.
  If npol=1, the returned vector is just the original input.
  If npol>1, there are only two possible values for the state of diagonal terms: 0x0 -> -1, and 1x1-> 1. They are returned as is.
  For non-diagonal terms, there are 3 possible values: 0x0 -> -1x-1, 0x1/1x0 -> -1x1/1x-1, and 1x1 -> 1x1.
  The state -1x-1 is mapped to -1, and the state 1x1 is mapped to a parameter W in the range [0, inf).
  The state -1x1/1x-1 is mapped to (W+1)*p-1, where p is a parameter between 0 and 1.
  """
  def __init__(
      self,
      distance,
      rounds,
      ignore_first_round,
      ignore_last_round,
      obs_type,
      is_symmetric,
      npol,
      ignore_diagonal,
      **kwargs
    ):
    super(DetectorEventStateEmbedder, self).__init__(**kwargs)
    self.distance = distance
    self.rounds = rounds
    self.ignore_first_round = ignore_first_round
    self.ignore_last_round = ignore_last_round
    self.obs_type = obs_type
    self.is_symmetric = is_symmetric
    self.npol = npol
    self.ignore_diagonal = ignore_diagonal
    self.n_ancillas = (self.distance**2 - 1)
    self.ndims = self.n_ancillas*(self.rounds-1)
    if not self.ignore_first_round:
      self.ndims += self.n_ancillas//2
    if not self.ignore_last_round:
      self.ndims += self.n_ancillas//2

    self.embedder_label = f"DetectorEventStateEmbedder_npol{self.npol}"

    self.embedding_param_map = get_detector_evts_perround_embedding_map(self.distance, self.rounds, self.obs_type, self.npol, self.is_symmetric, self.ignore_first_round, self.ignore_last_round)
    self.nondiag_param_map = None
    if self.embedding_param_map is not None:
      self.nondiag_param_map = [ i for i in self.embedding_param_map if i>=0 ]
    self.n_embedding_params = 0
    self.embedding_params_nondiag_neutral = None
    self.embedding_params_nondiag_positive = None
    if self.nondiag_param_map is not None:
      for i in self.nondiag_param_map:
        if i>self.n_embedding_params:
          self.n_embedding_params = i
      self.n_embedding_params += 1
      self.embedding_params_nondiag_neutral = self.add_weight(
        name=f"{self.embedder_label}_params_nondiag_neutral",
        shape=[ 1, self.n_embedding_params ], # States -1x1/1x-1 need a probability assignment
        initializer='zeros',
        trainable=True
      )
      self.embedding_params_nondiag_positive = self.add_weight(
        name=f"{self.embedder_label}_params_nondiag_positive",
        shape=[ 1, self.n_embedding_params ], # State 1x1 also needs a weight assignment that distinguishes it from -1x-1 -> -1.
        initializer='zeros',
        trainable=True
      )

    self.triangular_polmap = None
    self.triangular_backpolmap = None
    if self.npol>1:
      triangular_polmap_diag = []
      triangular_polmap_nondiag = []
      for iy in range(self.ndims):
        for ix in range(iy, self.ndims):
          if ix==iy:
            triangular_polmap_diag.append(ix+iy*self.ndims)
          else:
            triangular_polmap_nondiag.append(ix+iy*self.ndims)
      if not self.ignore_diagonal:
        self.triangular_polmap = triangular_polmap_diag
        self.triangular_polmap.extend(triangular_polmap_nondiag)
        triangular_polmap_sorted = sorted(self.triangular_polmap)
        self.triangular_backpolmap = [ self.triangular_polmap.index(ii) for ii in triangular_polmap_sorted ]
      else:
        self.triangular_polmap = triangular_polmap_nondiag
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "distance": self.distance,
        "rounds": self.rounds,
        "ignore_first_round": self.ignore_first_round,
        "ignore_last_round": self.ignore_last_round,
        "obs_type": self.obs_type,
        "is_symmetric": self.is_symmetric,
        "npol": self.npol,
        "ignore_diagonal": self.ignore_diagonal
      }
    )
    return config


  def embed_pol_state(self, res_diag, res_nondiag, embedding_nondiag_tr):
    # At this point, all terms have values (1, 4 | 2).
    # Non-diagonal terms
    res_nondiag = res_nondiag - 2 # (1, 2, 4) -> (-1, 0, 2)
    res_nondiag_sq = res_nondiag*res_nondiag # (-1, 0, 2) -> (1, 0, 4), need 3 bits + 1 for the sign
    res_nondiag_ones = tf.ones_like(res_nondiag)
    res_nondiag = tf.stack([res_nondiag_sq, res_nondiag, res_nondiag_ones], axis=2) # { x^2, x, 1 }
    res_nondiag = tf.cast(res_nondiag, tf.float32)
    res_nondiag = tf.reduce_sum(res_nondiag*embedding_nondiag_tr, axis=2)
    # Diagonal terms
    if res_diag is not None:
      res_diag = (2*(res_diag-2)-1)/3 # (1, 4) -> (-1, 1).
      res_diag = tf.cast(res_diag, tf.float32)
      res = tf.concat([res_diag, res_nondiag], axis=1)
      res = tf.gather(res, self.triangular_backpolmap, axis=1)
    else:
      res = res_nondiag
    return res


  def get_transformed_state(self, x, embedding_nondiag_tr):
    res = None
    if self.npol==1:
      res = tf.cast(x, tf.float32)
    else:
      xx = x + 1
      xsh = arrayops_shape(x)
      res = tf.matmul(
        tf.cast(tf.reshape(xx, shape=(xsh[0], xsh[1], 1)), tf.int16),
        tf.cast(tf.reshape(xx, shape=(xsh[0], 1, xsh[1])), tf.int16)
      )
      res = tf.gather(tf.reshape(res, shape=(xsh[0], -1)), self.triangular_polmap, axis=1)

      res_diag = None
      res_nondiag = None
      if self.ignore_diagonal:
        res_nondiag = res
      else:
        res_diag = res[:,0:self.ndims]
        res_nondiag = res[:,self.ndims:]
      # res has values (1, 4 | 2) now
      res = self.embed_pol_state(res_diag, res_nondiag, embedding_nondiag_tr)
    return res


  def get_transformed_embedding_params(self, n):
    if self.npol>1:
      params_nondiag_positive = tf.math.exp(VariableBounds.clip_zlike(self.embedding_params_nondiag_positive))
      #params_nondiag_positive = tf.math.exp(self.embedding_params_nondiag_positive)
      params_nondiag_positive = tf.gather(params_nondiag_positive, self.nondiag_param_map, axis=1)

      params_nondiag_neutral = tf.math.sigmoid(VariableBounds.clip_zlike(self.embedding_params_nondiag_neutral))
      #params_nondiag_neutral = tf.math.sigmoid(self.embedding_params_nondiag_neutral)
      params_nondiag_neutral = tf.gather(params_nondiag_neutral, self.nondiag_param_map, axis=1)
      params_nondiag_neutral = ((params_nondiag_positive+1)*params_nondiag_neutral - 1)

      coefn = params_nondiag_neutral/2
      coefp = params_nondiag_positive/6
      embedding_nondiag_tr = tf.stack(
        [
          -coefn + coefp - 1./3.,
          coefn + coefp + 2./3.,
          params_nondiag_neutral
        ],
        axis=2
      )
      return tf.repeat(embedding_nondiag_tr, n, axis=0)
    return None


  def call(self, input):
    n = arrayops_shape(input, 0)
    embedding_nondiag_tr = self.get_transformed_embedding_params(n)
    return self.get_transformed_state(input, embedding_nondiag_tr)



class TripletStateProbEmbedder(Layer):
  """
  TripletStateProbEmbedder: Convert the input data into a quadratic form with trainable probability assignments.

  The input needs to have dimensions (batch size, rounds>2, number of ancillas).

  The output is a list of values between 0 and 1.
  - npol=1:
    The output has dimensions (batch size, rounds-2, n_ancillas), and
    the values are 0, p, and 1, which correspond to round triplet states -1, 0, and 1, respectively.
  - npol>1:
    The output has dimensions (batch size, rounds-2, n_ancillas*(n_ancillas+1)/2).
    
    Diagonal terms:
    The round triplet states are still the states themselves, -1, 0, 1,
    so the output values are 0, p, and 1, respectively,

    Nondiagonal terms:
    There are 6 possible values ways to combine round triplet states: -1x-1, -1x0, -1x1, 0x0, 0x1, and 1x1.
    The output parameter map is therefore -1x-1 -> 0, -1x0 -> p1*p2, -1x1 -> p1*p4 + p1*p2*(1-p4), 0x0 -> p1*p3 + p1*p2*(1-p3), 0x1 -> p1, and 1x1 -> 1.
  
  Notes:
  - Given that the rank of the second dimension is rounds-2, the output assumes the same probability map in each triplet window,
    and it does not provide window-by-window correlations.
    
    It is therefore assumed that the user will use the embedding in a recurrent architecture, which will take care of time correlations
    as a separated feature of the NN architecture.
  
  - Since one of the input arguments is 'is_symmetric', the parameter maps will respect the symmetries of the circuit if requested.
  """
  def __init__(
      self,
      distance,
      rounds,
      npol,
      is_symmetric,
      **kwargs
    ):
    super(TripletStateProbEmbedder, self).__init__(**kwargs)
    self.distance = distance
    self.rounds = rounds
    self.npol = npol
    self.is_symmetric = is_symmetric
    self.n_ancillas = (self.distance**2 - 1)
    self.embed_label = f"TripletStateProbEmbedder_d{self.distance}_r{self.rounds}_npol{self.npol}"

    self.state_tracker = None
    self.delta_tracker = None

    self.triangular_polmap = None
    self.triangular_backpolmap = None
    self.embedding_params_diag = None
    self.embedding_params_nondiag = None
    self.diag_param_map = None
    self.nondiag_param_map = None
    if self.npol>1:
      self.diag_param_map, self.nondiag_param_map, self.triangular_polmap, self.triangular_backpolmap = get_layer_input_maps(self.n_ancillas, self.npol, self.is_symmetric)
      max_diag = 0
      for ii in self.diag_param_map:
        if ii>max_diag:
          max_diag = ii
      max_nondiag = 0
      for ii in self.nondiag_param_map:
        if ii>max_nondiag:
          max_nondiag = ii

      if not self.is_symmetric:
        self.diag_param_map = None
        self.nondiag_param_map = None

      self.embedding_params_diag = self.add_weight(
        name=f"{self.embed_label}_params_diag",
        shape=[ 1, max_diag+1 ], # Only state = 0 needs a probability assignment; state = -1 -> 0, and state = 1 -> 1
        initializer='zeros',
        trainable=True
      )
      self.embedding_params_nondiag = self.add_weight(
        name=f"{self.embed_label}_params_nondiag",
        shape=[ 1, max_nondiag+1, 4 ], # States -1x0, -1x1, 0x0, 0x1 need probability assignment; state = -1x-1 -> 0, and state = 1x1 -> 1
        initializer='zeros',
        trainable=True
      )
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'distance': self.distance,
        'rounds': self.rounds,
        'npol': self.npol,
        'is_symmetric': self.is_symmetric
      }
    )
    return config


  def embed_pol_state(self, res_diag, res_nondiag, embedding_diag_tr, embedding_nondiag_tr):
    # Diagonal terms
    res_diag = tf.cast(res_diag, tf.float32)
    res_diag = res_diag[:,:,0] + res_diag[:,:,1]*embedding_diag_tr
    # Non-diagonal terms
    res_nondiag = tf.cast(res_nondiag, tf.float32)
    res_nondiag = res_nondiag[:,:,0] + tf.reduce_sum(res_nondiag[:,:,1:5]*embedding_nondiag_tr, axis=2)
    res = tf.concat([res_diag, res_nondiag], axis=1)
    res = tf.gather(res, self.triangular_backpolmap, axis=1)
    return res


  def transform_raw_state(self, x, embedding_diag_tr, embedding_nondiag_tr):
    res = None
    if self.npol==1:
      res = tf.cast(x, tf.float32)
    else:
      xx = x + 3 # (-1, 0, 1) -> (2, 3, 4)
      xsh = arrayops_shape(x)
      res = tf.matmul(
        tf.cast(tf.reshape(xx, shape=(xsh[0], xsh[1], 1)), tf.int16),
        tf.cast(tf.reshape(xx, shape=(xsh[0], 1, xsh[1])), tf.int16)
      )
      res = tf.gather(tf.reshape(res, shape=(xsh[0], -1)), self.triangular_polmap, axis=1)
      res_diag = res[:,0:self.n_ancillas]
      res_nondiag = res[:,self.n_ancillas:]
      # res has values (4, 8, 16 | 6, 9, 12) now
      is16_diag = tf.equal(res_diag, 16)
      is9_diag = tf.equal(res_diag, 9)
      is16_nondiag = tf.equal(res_nondiag, 16)
      is12_nondiag = tf.equal(res_nondiag, 12)
      is9_nondiag = tf.equal(res_nondiag, 9)
      is8_nondiag = tf.equal(res_nondiag, 8)
      is6_nondiag = tf.equal(res_nondiag, 6)
      res_diag = tf.stack([is16_diag, is9_diag], axis=2)
      res_nondiag = tf.stack([is16_nondiag, is12_nondiag, is9_nondiag, is8_nondiag, is6_nondiag], axis=2)
      res = self.embed_pol_state(res_diag, res_nondiag, embedding_diag_tr, embedding_nondiag_tr)
    return res


  def get_transformed_embedding_params(self, n):
    if self.npol>1:
      params_diag = tf.math.sigmoid(VariableBounds.clip_zlike(self.embedding_params_diag))
      params_nondiag = tf.math.sigmoid(VariableBounds.clip_zlike(self.embedding_params_nondiag))
      #params_diag = tf.math.sigmoid(self.embedding_params_diag)
      #params_nondiag = tf.math.sigmoid(self.embedding_params_nondiag)
      if self.diag_param_map is not None:
        params_diag = tf.gather(params_diag, self.diag_param_map, axis=1)
      if self.nondiag_param_map is not None:
        params_nondiag = tf.gather(params_nondiag, self.nondiag_param_map, axis=1)

      embedding_diag_tr = tf.repeat(params_diag, n, axis=0)
      f12 = params_nondiag[:,:,0] # States 1,0
      f6 = params_nondiag[:,:,3]*f12 # States -1,0
      f9 = (params_nondiag[:,:,1]*f12 + (1-params_nondiag[:,:,1])*f6) # States 0,0
      f8 = (params_nondiag[:,:,2]*f12 + (1-params_nondiag[:,:,2])*f6) # States -1,1
      embedding_nondiag_tr = tf.repeat(tf.stack([f12, f9, f8, f6], axis=2), n, axis=0)
      return embedding_diag_tr, embedding_nondiag_tr
    return None, None


  def get_raw_state(self, input):
    data_err = input[:,0,:] + input[:,2,:] - input[:,0,:]*input[:,2,:]*2 # 001/011/110/100-like sequences
    measure_err = input[:,1,:]*(1 - (input[:,0,:] + input[:,2,:])) + input[:,0,:]*input[:,2,:] # 010/101-like sequences

    self.delta_tracker = self.delta_tracker*(1-measure_err*2)
    self.state_tracker = tf.math.minimum(
      tf.math.maximum(
        self.state_tracker + self.delta_tracker*data_err,
        -tf.ones_like(self.state_tracker)
      ),
      tf.ones_like(self.state_tracker)
    )
    self.delta_tracker = self.delta_tracker*(1-self.state_tracker*self.state_tracker*(1-measure_err)) - self.state_tracker*(1-measure_err)

    return self.state_tracker
  

  def get_transformed_state(self, input, embedding_diag_tr, embedding_nondiag_tr):
    res = []
    for rr in range(self.rounds-2):
      x = input[:,rr:rr+3,:]
      x = self.get_raw_state(x)
      x = self.transform_raw_state(x, embedding_diag_tr, embedding_nondiag_tr)
      res.append(x)
    res = tf.stack(res, axis=1) # (n_samples, n_rounds-2, number of pol-1 or pol-2 ancilla coordinates) dimensions
    return res


  def reset_tracker_states(self, n):
    self.state_tracker = -arrayops_ones(shape=(n, self.n_ancillas), dtype=tf.int8)
    self.delta_tracker = arrayops_ones(shape=(n, self.n_ancillas), dtype=tf.int8)


  def call(self, input):
    n = arrayops_shape(input, 0)
    self.reset_tracker_states(n)
    embedding_diag_tr, embedding_nondiag_tr = self.get_transformed_embedding_params(n)
    return self.get_transformed_state(input, embedding_diag_tr, embedding_nondiag_tr)



class CNNKernel(Layer):
  def __init__(
      self,
      kernel_parity, obs_type, kernel_distance, rounds,
      npol=1,
      do_all_data_qubits = False,
      include_det_bits = True,
      include_det_evts = True,
      n_remove_last_det_evts = 0,
      discard_activation = False,
      discard_bias = False,
      **kwargs
    ):
    super(CNNKernel, self).__init__(**kwargs)
    self.kernel_parity = kernel_parity
    self.obs_type = obs_type
    self.kernel_distance = kernel_distance
    self.rounds = rounds
    self.npol = npol
    self.do_all_data_qubits = do_all_data_qubits
    self.include_det_bits = include_det_bits
    self.include_det_evts = include_det_evts
    if not self.include_det_bits and not self.include_det_evts:
      raise ValueError("At least one of the detector bits or detector events must be included.")
    self.discard_activation = discard_activation
    self.discard_bias = discard_bias
    self.n_remove_last_det_evts = n_remove_last_det_evts
    self.n_ancillas = (self.kernel_distance**2 - 1)
    self.is_symmetric = kernel_parity[0]==0 and kernel_parity[1]==0

    constraint_label = f"{kernel_parity[0]}_{kernel_parity[1]}"

    ndim1 = self.n_ancillas*self.rounds # Number of ancilla measurements
    ndim2 = 0 # Number of detector events
    if self.include_det_evts:
      ndim2 += self.n_ancillas//2 + self.n_ancillas*(self.rounds-1) # Number of detector event bits within each round
      ndim2 += self.n_ancillas//2 - self.n_remove_last_det_evts
    self.triangular_polmap_det_bits = None
    self.triangular_polmap_det_evts = None
    if self.npol>1:
      self.triangular_polmap_det_bits = []
      for iy in range(ndim1):
        for ix in range(iy, ndim1):
          self.triangular_polmap_det_bits.append(ix+iy*ndim1)
      self.triangular_polmap_det_evts = []
      for iy in range(ndim2):
        for ix in range(iy, ndim2):
          self.triangular_polmap_det_evts.append(ix+iy*ndim2)
      ndim1 = ndim1*(ndim1+1)//2
      ndim2 = ndim2*(ndim2+1)//2

    num_outputs = 1
    self.kernel_weights_det_bits_swap_map = None
    self.kernel_weights_det_evts_swap_map = None
    self.final_res_map = None
    if self.do_all_data_qubits:
      self.kernel_weights_det_bits_swap_map = get_detector_bits_perround_map(self.kernel_distance, self.rounds, self.npol, self.is_symmetric)
      self.kernel_weights_det_evts_swap_map = get_detector_evts_perround_map(self.kernel_distance, self.rounds, self.n_remove_last_det_evts, self.npol, self.is_symmetric)
      self.final_res_map = get_layer_output_map(self.kernel_distance, self.is_symmetric)
      if self.is_symmetric:
        num_outputs = (self.kernel_distance**2 + 1)//2
      else:
        num_outputs = self.kernel_distance**2

    self.ndims = [ [ ndim1, num_outputs ], [ ndim2, num_outputs ] ]
    self.kernel_weights_det_bits = None
    if self.include_det_bits:
      self.kernel_weights_det_bits = self.add_weight(
        name=f"CNNkernel{self.kernel_distance}_{constraint_label}_w_det_bits",
        shape=self.ndims[0],
        initializer='zeros',
        trainable=True
      )
    self.kernel_weights_det_evts = None
    if self.include_det_evts:
      self.kernel_weights_det_evts = self.add_weight(
        name=f"CNNkernel{self.kernel_distance}_{constraint_label}_w_det_evts",
        shape=self.ndims[1],
        initializer='zeros',
        trainable=True
      )
    self.kernel_bias = None
    self.kernel_activation = None
    if self.npol<=1 and not self.discard_bias:
      self.kernel_bias = self.add_weight(
        name=f"CNNkernel{self.kernel_distance}_{constraint_label}_b",
        shape=[ 1, num_outputs ],
        initializer='zeros',
        trainable=True
      )
    if not self.discard_activation:
      self.kernel_activation = tf.keras.activations.sigmoid
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'kernel_parity': self.kernel_parity,
        'obs_type': self.obs_type,
        'kernel_distance': self.kernel_distance,
        'rounds': self.rounds,
        'npol': self.npol,
        'do_all_data_qubits': self.do_all_data_qubits,
        'include_det_bits': self.include_det_bits,
        'include_det_evts': self.include_det_evts,
        'n_remove_last_det_evts': self.n_remove_last_det_evts,
        'discard_activation': self.discard_activation,
        'discard_bias': self.discard_bias
      }
    )
    return config


  def build(self, input_shape):
    pass


  def transform_inputs(self, x, tmap):
    res = None
    if self.npol==1:
      res = x
    else:
      xsh = arrayops_shape(x)
      res = tf.matmul(
        tf.cast(tf.reshape((x+1), shape=(xsh[0], xsh[1], 1)), tf.int16),
        tf.cast(tf.reshape((x+1), shape=(xsh[0], 1, xsh[1])), tf.int16)
      )
      res = tf.gather(tf.reshape(res, (xsh[0], -1)), tmap, axis=1)
      res = (-res*res+res*9-14)/6 # (1, 2, 4) -> (-1, 0, 1)
    return tf.cast(res, tf.float32)
  

  def get_mapped_weights(self, w, wmap):
    if not self.is_symmetric or wmap is None:
      return w
    wgts_mapped = []
    for mm in wmap:
      jout = mm[0]
      ilist = mm[1]
      if ilist is None:
        wgts_mapped.append(w[:,jout])
      else:
        wgts_mapped.append(tf.gather(w[:,jout], ilist))
    return tf.stack(wgts_mapped, axis=1)
  

  def get_mapped_bias(self, n):
    if self.kernel_bias is None or not self.is_symmetric or self.final_res_map is None:
      return tf.repeat(self.kernel_bias, n, axis=0) if self.kernel_bias is not None else None
    return tf.repeat(tf.gather(self.kernel_bias, self.final_res_map, axis=1), n, axis=0)


  def evaluate(self, bits, do_evts):
    w = None
    tmap = None
    wmap = None
    if not do_evts:
      w = self.kernel_weights_det_bits
      tmap = self.triangular_polmap_det_bits
      wmap = self.kernel_weights_det_bits_swap_map
    else:
      w = self.kernel_weights_det_evts
      tmap = self.triangular_polmap_det_evts
      wmap = self.kernel_weights_det_evts_swap_map
    return tf.matmul(self.transform_inputs(bits, tmap), self.get_mapped_weights(w, wmap))


  def call(self, inputs):
    n = None
    res = None
    det_bits = None
    det_evts = None
    if self.include_det_bits and self.include_det_evts:
      det_bits = inputs[0]
      det_evts = inputs[1]
      n = arrayops_shape(det_bits, 0)
    else:
      if self.include_det_bits:
        if isinstance(inputs, list):
          det_bits = inputs[0]
        else:
          det_bits = inputs
        n = arrayops_shape(det_bits, 0)
      else:
        if isinstance(inputs, list):
          det_evts = inputs[0]
        else:
          det_evts = inputs
        n = arrayops_shape(det_evts, 0)

    if self.include_det_bits:
      res = self.evaluate(det_bits, False)
    if self.include_det_evts:
      res2 = self.evaluate(det_evts, True)
      if res is not None:
        res = res + res2
      else:
        res = res2
    bias_term = self.get_mapped_bias(n)
    if bias_term is not None:
      res = res + bias_term
    # At this point, res is z-like, and we can clip it.
    res = VariableBounds.clip_zlike(res)
    if self.kernel_activation is not None:
      res = self.kernel_activation(res)
    return res



class CNNKernelWithEmbedding(Layer):
  def __init__(
      self,
      kernel_parity, obs_type, kernel_distance, rounds,
      npol=1,
      do_all_data_qubits = False,
      include_det_bits = True,
      include_det_evts = True,
      n_remove_last_det_evts = 0,
      discard_activation = False,
      discard_bias = False,
      ignore_first_det_evt_round = False,
      use_exp_act = False,
      **kwargs
    ):
    super(CNNKernelWithEmbedding, self).__init__(**kwargs)
    self.kernel_parity = kernel_parity
    self.obs_type = obs_type
    self.kernel_distance = kernel_distance
    self.rounds = rounds
    self.npol = npol
    self.do_all_data_qubits = do_all_data_qubits
    self.include_det_bits = include_det_bits
    self.include_det_evts = include_det_evts
    if not self.include_det_bits and not self.include_det_evts:
      raise ValueError("At least one of the detector bits or detector events must be included.")
    self.discard_activation = discard_activation
    self.discard_bias = discard_bias
    self.n_remove_last_det_evts = n_remove_last_det_evts
    self.n_ancillas = (self.kernel_distance**2 - 1)
    self.is_symmetric = kernel_parity[0]==0 and kernel_parity[1]==0
    self.ignore_diagonal_det_bits = self.npol > 1
    self.ignore_diagonal_det_evts = self.npol > 1 and self.include_det_bits
    self.ignore_first_det_evt_round = ignore_first_det_evt_round
    self.use_exp_act = use_exp_act
    self.rounds_det_evts = self.rounds
    if self.ignore_first_det_evt_round and self.n_remove_last_det_evts>0:
      self.rounds_det_evts += 1

    label = f"CNNKernelWithEmbedding_{kernel_parity[0]}_{kernel_parity[1]}"

    ndim1 = self.n_ancillas*self.rounds # Number of ancilla measurements
    ndim2 = 0 # Number of detector events
    if self.include_det_evts:
      if not self.ignore_first_det_evt_round:
        ndim2 += self.n_ancillas//2
      ndim2 += self.n_ancillas*(self.rounds_det_evts-1) # Number of detector event bits within each round
      ndim2 += self.n_ancillas//2 - self.n_remove_last_det_evts
    if self.npol>1:
      if self.ignore_diagonal_det_bits:
        ndim1 = ndim1 - 1
      if self.ignore_diagonal_det_evts:
        ndim2 = ndim2 - 1
      ndim1 = ndim1*(ndim1+1)//2
      ndim2 = ndim2*(ndim2+1)//2

    self.embedding_det_bits = None
    self.embedding_det_evts = None
    if self.include_det_bits:
      self.embedding_det_bits = DetectorBitStateEmbedder(
        distance = self.kernel_distance,
        rounds = self.rounds,
        is_symmetric = self.is_symmetric,
        npol = self.npol,
        ignore_diagonal = self.ignore_diagonal_det_bits
      )
    if self.include_det_evts:
      self.embedding_det_evts = DetectorEventStateEmbedder(
        distance = self.kernel_distance,
        rounds = self.rounds_det_evts,
        ignore_first_round = self.ignore_first_det_evt_round,
        ignore_last_round = (self.n_remove_last_det_evts>0),
        obs_type = self.obs_type,
        is_symmetric = self.is_symmetric,
        npol = self.npol,
        ignore_diagonal = self.ignore_diagonal_det_evts
      )

    num_outputs = 1
    self.kernel_weights_det_bits_swap_map = None
    self.kernel_weights_det_evts_swap_map = None
    self.final_res_map = None
    if self.do_all_data_qubits:
      self.kernel_weights_det_bits_swap_map = get_detector_bits_perround_map(
        self.kernel_distance, self.rounds, self.npol, self.is_symmetric, ignore_diagonal=self.ignore_diagonal_det_bits
      )
      self.kernel_weights_det_evts_swap_map = get_detector_evts_perround_map(
        self.kernel_distance, self.rounds_det_evts, self.n_remove_last_det_evts, self.npol, self.is_symmetric,
        ignore_first_round=self.ignore_first_det_evt_round, ignore_diagonal=self.ignore_diagonal_det_evts
      )
      self.final_res_map = get_layer_output_map(self.kernel_distance, self.is_symmetric)
      if self.is_symmetric:
        num_outputs = (self.kernel_distance**2 + 1)//2
      else:
        num_outputs = self.kernel_distance**2

    self.ndims = [ [ ndim1, num_outputs ], [ ndim2, num_outputs ] ]
    self.kernel_weights_det_bits = None
    if self.include_det_bits:
      self.kernel_weights_det_bits = self.add_weight(
        name=f"{label}_w_det_bits",
        shape=self.ndims[0],
        initializer='zeros',
        trainable=True
      )
    self.kernel_weights_det_evts = None
    if self.include_det_evts:
      self.kernel_weights_det_evts = self.add_weight(
        name=f"{label}_w_det_evts",
        shape=self.ndims[1],
        initializer='zeros',
        trainable=True
      )
    self.kernel_bias = None
    self.kernel_activation = None
    if self.npol<=1 and not self.discard_bias:
      self.kernel_bias = self.add_weight(
        name=f"{label}_b",
        shape=[ 1, num_outputs ],
        initializer='zeros',
        trainable=True
      )
    if not self.discard_activation:
      if self.use_exp_act:
        self.kernel_activation = tf.keras.activations.exponential
      else:
        self.kernel_activation = tf.keras.activations.sigmoid
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'kernel_parity': self.kernel_parity,
        'obs_type': self.obs_type,
        'kernel_distance': self.kernel_distance,
        'rounds': self.rounds,
        'npol': self.npol,
        'do_all_data_qubits': self.do_all_data_qubits,
        'include_det_bits': self.include_det_bits,
        'include_det_evts': self.include_det_evts,
        'n_remove_last_det_evts': self.n_remove_last_det_evts,
        'discard_activation': self.discard_activation,
        'discard_bias': self.discard_bias,
        'ignore_first_det_evt_round': self.ignore_first_det_evt_round,
        'use_exp_act': self.use_exp_act
      }
    )


  def build(self, input_shape):
    pass


  def transform_inputs(self, x, tmap):
    return tmap(x)
  

  def get_mapped_weights(self, w, wmap):
    if not self.is_symmetric or wmap is None:
      return w
    wgts_mapped = []
    for mm in wmap:
      jout = mm[0]
      ilist = mm[1]
      if ilist is None:
        wgts_mapped.append(w[:,jout])
      else:
        wgts_mapped.append(tf.gather(w[:,jout], ilist))
    return tf.stack(wgts_mapped, axis=1)
  

  def get_mapped_bias(self, n):
    if self.kernel_bias is None or not self.is_symmetric or self.final_res_map is None:
      return tf.repeat(self.kernel_bias, n, axis=0) if self.kernel_bias is not None else None
    return tf.repeat(tf.gather(self.kernel_bias, self.final_res_map, axis=1), n, axis=0)


  def evaluate(self, bits, do_evts):
    w = None
    tmap = None
    wmap = None
    if not do_evts:
      w = self.kernel_weights_det_bits
      tmap = self.embedding_det_bits
      wmap = self.kernel_weights_det_bits_swap_map
    else:
      w = self.kernel_weights_det_evts
      tmap = self.embedding_det_evts
      wmap = self.kernel_weights_det_evts_swap_map
    return tf.matmul(self.transform_inputs(bits, tmap), self.get_mapped_weights(w, wmap))


  def call(self, inputs):
    n = None
    res = None
    det_bits = None
    det_evts = None
    if self.include_det_bits and self.include_det_evts:
      det_bits = inputs[0]
      det_evts = inputs[1]
      n = arrayops_shape(det_bits, 0)
    else:
      if self.include_det_bits:
        if isinstance(inputs, list):
          det_bits = inputs[0]
        else:
          det_bits = inputs
        n = arrayops_shape(det_bits, 0)
      else:
        if isinstance(inputs, list):
          det_evts = inputs[0]
        else:
          det_evts = inputs
        n = arrayops_shape(det_evts, 0)

    if self.include_det_bits:
      res = self.evaluate(det_bits, False)
    if self.include_det_evts:
      res2 = self.evaluate(det_evts, True)
      if res is not None:
        res = res + res2
      else:
        res = res2

    bias_term = self.get_mapped_bias(n)
    if bias_term is not None:
      res = res + bias_term
    # At this point, res is z-like, and we can clip it.
    res = VariableBounds.clip_zlike(res)
    if self.kernel_activation is not None:
      res = self.kernel_activation(res)
    return res



class CNNStateCorrelatorWeightInitializer(Initializer):
  def __init__(
      self,
      distance,
      rounds,
      is_symmetric,
      npol,
      include_npol2_linear,
      disable_fractions,
      new_state_reversability,
      scale = 1.
    ):
    self.distance = distance
    self.rounds = rounds
    self.is_symmetric = is_symmetric
    self.npol = npol
    self.include_npol2_linear = include_npol2_linear
    self.disable_fractions = disable_fractions
    self.new_state_reversability = new_state_reversability
    self.scale = scale
    self.n_fracs = self.rounds-1
    self.n_phases = self.rounds*(self.rounds-1)//2

    ndim = self.distance**2 * self.rounds
    num_inputs = ndim if self.npol==1 else ndim*(ndim+(3 if self.include_npol2_linear else 1))//2
    num_outputs = self.distance**2 if not self.is_symmetric else (self.distance**2 + 1)//2
    n_evolve_params = self.n_fracs*(1 if self.disable_fractions else 2) + (1 if self.new_state_reversability else 0) + self.n_phases
    self.shape_exp = [ n_evolve_params, num_inputs, num_outputs ]


  def __call__(self, shape, dtype=None):
    # Check first if the shape is compatible with the expected shape
    if list(shape) != self.shape_exp:
      raise ValueError(f"Expected shape {self.shape_exp}, but received shape {shape}")
    # Initialize weights for state reversion for old states
    res = np.zeros(shape, dtype=convert_to_npdtype(dtype))
    #for iparam in range(self.n_fracs):
    #  for r in range(self.rounds):
    #    res[iparam,r,:,:] = (self.n_fracs if iparam+1==r else -1)
    # No need to initialize fraction and phase weights - they should already be 0.
    #idx_offset = self.n_fracs if self.disable_fractions else 2*self.n_fracs
    #if self.new_state_reversability:
    #  res[idx_offset,0,:,:] = self.n_fracs
    #  res[idx_offset,1:,:,:] = -1
    #res = res*self.scale
    return tf.convert_to_tensor(res, dtype=dtype)


  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'distance': self.distance,
        'rounds': self.rounds,
        'is_symmetric': self.is_symmetric,
        'npol': self.npol,
        'include_npol2_linear': self.include_npol2_linear,
        'disable_fractions': self.disable_fractions,
        'new_state_reversability': self.new_state_reversability,
        'scale': self.scale
      }
    )
    return config



class CNNStateCorrelatorBiasInitializer(Initializer):
  def __init__(
      self,
      distance,
      rounds,
      is_symmetric,
      npol,
      disable_fractions,
      new_state_reversability,
      scale = 1.
    ):
    self.distance = distance
    self.rounds = rounds
    self.is_symmetric = is_symmetric
    self.npol = npol
    self.disable_fractions = disable_fractions
    self.new_state_reversability = new_state_reversability
    self.scale = scale
    self.n_fracs = self.rounds-1
    self.n_phases = self.rounds*(self.rounds-1)//2

    num_outputs = self.distance**2 if not self.is_symmetric else (self.distance**2 + 1)//2
    n_evolve_params = self.n_fracs*(1 if self.disable_fractions else 2) + (1 if self.new_state_reversability else 0) + self.n_phases
    self.shape_exp = [ n_evolve_params, 1, num_outputs ]


  def __call__(self, shape, dtype=None):
    # Check first if the shape is compatible with the expected shape
    if list(shape) != self.shape_exp:
      raise ValueError(f"Expected shape {self.shape_exp}, but received shape {shape}")
    # Initialize weights for state reversion for old states
    res = np.zeros(shape, dtype=convert_to_npdtype(dtype))
    #res[:self.n_fracs,:,:] = 2
    res[:self.n_fracs,:,:] = 1
    idx_offset = self.n_fracs if self.disable_fractions else 2*self.n_fracs
    if self.new_state_reversability:
      #res[idx_offset,:,:] = 2
      res[idx_offset,:,:] = 1
    res = res*self.scale
    return tf.convert_to_tensor(res, dtype=dtype)


  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'distance': self.distance,
        'rounds': self.rounds,
        'is_symmetric': self.is_symmetric,
        'npol': self.npol,
        'disable_fractions': self.disable_fractions,
        'new_state_reversability': self.new_state_reversability,
        'scale': self.scale
      }
    )
    return config



class CNNStateCorrelator(Layer):
  disable_fractions = False
  new_state_reversability = True


  @staticmethod
  def set_disable_fractions(flag):
    CNNStateCorrelator.disable_fractions = flag

  
  @staticmethod
  def set_new_state_reversability(flag):
    CNNStateCorrelator.new_state_reversability = flag


  def __init__(
    self,
    distance,
    rounds,
    is_symmetric,
    npol = 1,
    use_exp_act = True,
    reset_disable_fractions = None,
    reset_new_state_reversability = None,
    **kwargs
  ):
    super(CNNStateCorrelator, self).__init__(**kwargs)
    self.distance = distance
    self.rounds = rounds
    #self.npol = npol
    self.npol = 1
    self.is_symmetric = is_symmetric
    self.use_exp_act = use_exp_act
    self.include_npol2_linear = True and self.npol>1

    # In case we are constructing through a call to from_config(), we need to have a reset functionality for static variables.
    if reset_disable_fractions is not None:
      CNNStateCorrelator.set_disable_fractions(reset_disable_fractions)
    if reset_new_state_reversability is not None:
      CNNStateCorrelator.set_new_state_reversability(reset_new_state_reversability)

    label = f"CNNStateCorrelator_d{self.distance}_r{self.rounds}"

    num_outputs = self.distance**2 if not self.is_symmetric else (self.distance**2 + 1)//2
    ndim = self.distance**2 * self.rounds
    num_inputs = ndim if self.npol==1 else ndim*(ndim+(3 if self.include_npol2_linear else 1))//2
    self.triangular_polmap_states = None
    if self.npol>1:
      self.triangular_polmap_states = []
      for iy in range(ndim):
        for ix in range(iy, ndim):
          self.triangular_polmap_states.append(ix+iy*ndim)
    self.output_map = None
    self.output_weight_map = None
    if self.is_symmetric:
      self.output_map = get_layer_output_map(self.distance, self.is_symmetric)
      self.output_weight_map = get_states_perround_map(self.distance, self.rounds, self.npol, self.is_symmetric)
      if self.include_npol2_linear:
        owm1 = get_states_perround_map(self.distance, self.rounds, 1, self.is_symmetric)
        for i in range(len(owm1)):
          tmp_map = deepcopy(owm1[i][1])
          for v in self.output_weight_map[i][1]:
            tmp_map.append(v + ndim)
          self.output_weight_map[i][1] = tmp_map

    self.n_fracs = self.rounds-1
    self.n_phases = self.rounds*(self.rounds-1)//2
    n_evolve_params = self.n_fracs*(1 if CNNStateCorrelator.disable_fractions else 2) + (1 if CNNStateCorrelator.new_state_reversability else 0) + self.n_phases
    self.params_state_evolutions = self.add_weight(
      name=f"{label}_w_state_evolutions",
      shape=[ n_evolve_params, num_inputs, num_outputs ],
      initializer=CNNStateCorrelatorWeightInitializer(
        distance=self.distance,
        rounds=self.rounds,
        is_symmetric=self.is_symmetric,
        npol=self.npol,
        include_npol2_linear=self.include_npol2_linear,
        disable_fractions=CNNStateCorrelator.disable_fractions,
        new_state_reversability=CNNStateCorrelator.new_state_reversability,
        scale=1.
      ),
      trainable=True
    )
    self.params_b = self.add_weight(
      name=f"{label}_b",
      shape=[ n_evolve_params, 1, num_outputs ],
      initializer=CNNStateCorrelatorBiasInitializer(
        distance=self.distance,
        rounds=self.rounds,
        is_symmetric=self.is_symmetric,
        npol=self.npol,
        disable_fractions=CNNStateCorrelator.disable_fractions,
        new_state_reversability=CNNStateCorrelator.new_state_reversability,
        scale=1.
      ),
      trainable=True
    )

    self.reverse_activation = tf.math.tanh
    self.cpwgt_activation = tf.math.tanh

  
  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'distance': self.distance,
        'rounds': self.rounds,
        'is_symmetric': self.is_symmetric,
        'npol': self.npol,
        'use_exp_act': self.use_exp_act,
        'reset_disable_fractions': CNNStateCorrelator.disable_fractions,
        'reset_new_state_reversability': CNNStateCorrelator.new_state_reversability
      }
    )
    return config

  
  def get_mapped_weights(self, w, wmap):
    if wmap is None:
      return w
    wgts_mapped = []
    for mm in wmap:
      jout = mm[0]
      ilist = mm[1]
      if ilist is None:
        wgts_mapped.append(w[:,jout])
      else:
        wgts_mapped.append(tf.gather(w[:,jout], ilist))
    return tf.stack(wgts_mapped, axis=1)


  def get_mapped_bias(self, bias, n):
    if bias is None or not self.is_symmetric or self.output_map is None:
      return tf.repeat(bias, n, axis=0) if bias is not None else None
    return tf.repeat(tf.gather(bias, self.output_map, axis=1), n, axis=0)


  def transform_states(self, states, n):
    if self.npol==1:
      return states
    else:
      mm = tf.reshape(states, shape=(n, -1, 1))
      mmt = tf.transpose(mm, perm=[0, 2, 1])
      res = tf.gather(tf.reshape(tf.matmul(mm, mmt), shape=(n, -1)), self.triangular_polmap_states, axis=1)
      if not self.include_npol2_linear:
        return res
      else:
        return tf.concat([ states, res ], axis=1)


  def call(self, inputs):
    n = arrayops_shape(inputs[0], 0)
    states = self.transform_states(tf.reshape(tf.stack(inputs, axis=1), shape=(n, -1)), n)

    # c_reverse is in [-1, 1]
    c_reverse = []
    # f is supposed to be in [0, 1], f_z in (-inf, inf), f_log_1pexpz in [0, inf)
    f_z = []
    for ifrac in range(self.n_fracs):
      reverse_arg_sum = tf.matmul(states, self.get_mapped_weights(self.params_state_evolutions[ifrac], self.output_weight_map))
      reverse_arg_sum += self.get_mapped_bias(self.params_b[ifrac], n)
      # No need for clipping. Boundaries at -1 and 1 are acceptable.
      c_reverse.append(self.reverse_activation(reverse_arg_sum))
      fwgt_z = None
      if not CNNStateCorrelator.disable_fractions:
        fwgt_z = tf.matmul(states, self.get_mapped_weights(self.params_state_evolutions[ifrac+self.n_fracs], self.output_weight_map))
        fwgt_z += self.get_mapped_bias(self.params_b[ifrac+self.n_fracs], n)
        fwgt_z = VariableBounds.clip_zlike(fwgt_z)
        f_z.append(fwgt_z)
    f_log_1pexpz = [ tf.math.log1p(tf.math.exp(fz)) for fz in f_z ]

    idx_offset = self.n_fracs*(1 if CNNStateCorrelator.disable_fractions else 2)

    c_reverse_new_state = None
    if CNNStateCorrelator.new_state_reversability:
      c_reverse_new_state = tf.matmul(states, self.get_mapped_weights(self.params_state_evolutions[idx_offset], self.output_weight_map))
      c_reverse_new_state += self.get_mapped_bias(self.params_b[idx_offset], n)
      c_reverse_new_state = self.reverse_activation(c_reverse_new_state)
      idx_offset += 1
    
    two_cos_phi = []
    # two_cos_phi is in [-2, 2]
    for iph in range(self.n_phases):
      jph = idx_offset + iph
      cpwgt_arg_sum = tf.matmul(states, self.get_mapped_weights(self.params_state_evolutions[jph], self.output_weight_map))
      cpwgt_arg_sum += self.get_mapped_bias(self.params_b[jph], n)
      two_cos_phi.append(self.cpwgt_activation(cpwgt_arg_sum)*2)

    f_denom = None
    state_args = []
    for ist, state in enumerate(inputs):
      fwgt = None
      if not CNNStateCorrelator.disable_fractions and self.n_fracs>0:
        if f_denom is None:
          f_denom = -f_log_1pexpz[ist]
        elif ist<self.n_fracs:
          f_denom -= f_log_1pexpz[ist]
        if ist<self.rounds-1:
          fwgt = f_z[ist] + f_denom
        else:
          fwgt = f_denom

      state_arg = None
      if ist==0:
        state_arg = state
        if CNNStateCorrelator.new_state_reversability:
          state_arg = state_arg*c_reverse_new_state
      else:
        state_arg = state*c_reverse[ist-1]
      if fwgt is not None:
        state_arg = state_arg + fwgt
      state_args.append(state_arg)

    res = None
    for ist in range(self.rounds):
      iarg = state_args[ist]
      state_part = tf.math.exp(iarg)
      if res is None:
        res = state_part
      else:
        res += state_part
      for jst in range(ist+1, self.rounds):
        res += tf.math.exp((iarg + state_args[jst])/2)*two_cos_phi[jst + ist*(self.rounds-1) - (ist+1)*ist//2 - 1]

    res = VariableBounds.clip_exp(res)
    if self.use_exp_act:
      return res # This is already an exponential-like variable in [0, inf).
    else:
      return res/(1+res)



class CNNStateDenseCorrelator(Layer):
  def __init__(
    self,
    distance,
    rounds,
    is_symmetric,
    hidden_specs,
    npol = 1,
    use_exp_act = True,
    **kwargs
  ):
    super(CNNStateDenseCorrelator, self).__init__(**kwargs)
    self.distance = distance
    self.rounds = rounds
    self.npol = npol
    self.is_symmetric = is_symmetric
    self.hidden_specs = hidden_specs
    self.use_exp_act = use_exp_act

    ndim = self.distance**2 * self.rounds
    self.triangular_polmap_states = None
    if self.npol>1:
      self.triangular_polmap_states = []
      for iy in range(ndim):
        for ix in range(iy, ndim):
          self.triangular_polmap_states.append(ix+iy*ndim)

    self.output_map = None
    self.output_weight_map = None
    if self.is_symmetric:
      self.output_map = get_layer_output_map(self.distance, self.is_symmetric)
      self.output_weight_map = get_states_perround_map(self.distance, self.rounds, self.npol, self.is_symmetric)

    self.hidden_layers = []
    if hidden_specs is not None:
      for hl in hidden_specs:
        if type(hl)==int:
          self.hidden_layers.append(Dense(hl, activation="relu"))
        elif type(hl)==dict:
          is_activation = hl["is_activation"] if "is_activation" in hl else False
          if not is_activation:
            n_nodes = hl["n_nodes"] # Mandatory
            has_activation = hl["has_activation"] if "has_activation" in hl else True
            activation = None
            if has_activation:
              activation = hl["activation"] if "activation" in hl else "relu"
            self.hidden_layers.append(Dense(n_nodes, activation=activation))
          else:
            self.hidden_layers.append(hl["activation"])
    num_outputs = self.distance**2 if not self.is_symmetric else (self.distance**2 + 1)//2
    self.hidden_layers.append(Dense(num_outputs, activation=None))
    self.output_act = tf.keras.activations.sigmoid if not self.use_exp_act else tf.keras.activations.exponential

  
  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'distance': self.distance,
        'rounds': self.rounds,
        'is_symmetric': self.is_symmetric,
        'hidden_specs': self.hidden_specs,
        'npol': self.npol,
        'use_exp_act': self.use_exp_act
      }
    )
    return config


  def transform_states(self, states, n):
    if self.npol==1:
      return states
    else:
      mm = tf.reshape(states, shape=(n, -1, 1))
      mmt = tf.transpose(mm, perm=[0, 2, 1])
      return tf.gather(tf.reshape(tf.matmul(mm, mmt), shape=(n, -1)), self.triangular_polmap_states, axis=1)
    
  
  def apply_final_activation(self, z):
    if self.output_act is not None:
      return self.output_act(z)
    else:
      return z


  def call(self, inputs):
    n = arrayops_shape(inputs[0], 0)
    states = self.transform_states(tf.reshape(tf.stack(inputs, axis=1), shape=(n, -1)), n)
    xf = states
    for hl in self.hidden_layers:
      xf = hl(xf)
    res = xf
    if self.is_symmetric:
      xr = tf.gather(states, self.output_weight_map[-1][1], axis=1)
      for hl in self.hidden_layers:
        xr = hl(xr)
      res = []
      for q in range(len(self.output_map)):
        qr = self.output_map[q]
        if q == qr:
          res.append(xf[:,qr])
        else:
          res.append(xr[:,qr])
      res = tf.stack(res, axis=1)
    res = VariableBounds.clip_zlike(res)
    return self.apply_final_activation(res)



class CNNKernelEnums:
  kFixed = 0
  kEmbedded = 1



class CNNEmbeddedKernelChooser:
  kernel_type = CNNKernelEnums.kFixed


  @classmethod
  def set_kernel_type(cls, kernel_type):
    cls.kernel_type = kernel_type
  

  @classmethod
  def get_kernel_class(cls):
    if cls.kernel_type==CNNKernelEnums.kFixed:
      return CNNKernel
    elif cls.kernel_type==CNNKernelEnums.kEmbedded:
      return CNNKernelWithEmbedding
    else:
      raise ValueError("Unknown kernel type.")


  @classmethod
  def get_kernel(cls, **kwargs):
    return cls.get_kernel_class()(**kwargs)



class RCNNEmbeddedKernelChooser:
  kernel_type = CNNKernelEnums.kEmbedded


  @classmethod
  def set_kernel_type(cls, kernel_type):
    cls.kernel_type = kernel_type
  

  @classmethod
  def get_kernel_class(cls):
    if cls.kernel_type==CNNKernelEnums.kFixed:
      return CNNKernel
    elif cls.kernel_type==CNNKernelEnums.kEmbedded:
      return CNNKernelWithEmbedding
    else:
      raise ValueError("Unknown kernel type.")


  @classmethod
  def get_kernel(cls, **kwargs):
    return cls.get_kernel_class()(**kwargs)



class StateCorrelatorEnums:
  kAnalytical = 0
  kDense = 1



class RCNNStateCorrelatorChooser:
  correlator_type = StateCorrelatorEnums.kAnalytical
  hidden_layer_specs = None


  @classmethod
  def set_correlator_type(cls, correlator_type, **kwargs):
    cls.correlator_type = correlator_type
    if correlator_type==StateCorrelatorEnums.kDense:
      cls.hidden_layer_specs = kwargs.get("hidden_layer_specs", None)


  @classmethod
  def get_correlator_class(cls):
    if cls.correlator_type==StateCorrelatorEnums.kAnalytical:
      return CNNStateCorrelator
    elif cls.correlator_type==StateCorrelatorEnums.kDense:
      return CNNStateDenseCorrelator
    else:
      raise ValueError("Unknown correlator type.")


  @classmethod
  def get_state_correlator_ctor_args(cls, distance, rounds, is_symmetric, npol, use_exp_act, **kwargs):
    res = {
      'distance': distance,
      'rounds': rounds,
      'is_symmetric': is_symmetric,
      'npol': npol,
      'use_exp_act': use_exp_act
    }
    if cls.correlator_type==StateCorrelatorEnums.kDense:
      res["hidden_specs"] = cls.hidden_layer_specs
    for key, val in kwargs.items():
      res[key] = val
    return res


  @classmethod
  def get_correlator(cls, distance, rounds, is_symmetric, npol, use_exp_act, **kwargs):
    args = cls.get_state_correlator_ctor_args(
      distance = distance,
      rounds = rounds,
      is_symmetric = is_symmetric,
      npol = npol,
      use_exp_act = use_exp_act,
      **kwargs
    )
    return cls.get_correlator_class()(**args)



class RCNNInitialStateKernel(Layer):
  rounds_first = 1


  @staticmethod
  def set_rounds_first(val):
    RCNNInitialStateKernel.rounds_first = val
  
  
  def __init__(
      self,
      kernel_parity,
      obs_type, kernel_distance,
      npol = 1,
      use_exp_act = True,
      rounds_first_override = None,
      **kwargs
    ):
    super(RCNNInitialStateKernel, self).__init__(**kwargs)
    self.embedded_kernel = RCNNEmbeddedKernelChooser.get_kernel(
      kernel_parity = kernel_parity,
      obs_type = obs_type,
      kernel_distance = kernel_distance,
      rounds = RCNNInitialStateKernel.rounds_first if rounds_first_override is None else rounds_first_override,
      npol = npol,
      do_all_data_qubits = True,
      include_det_bits = True,
      include_det_evts = (RCNNInitialStateKernel.rounds_first>1),
      n_remove_last_det_evts = (kernel_distance**2 - 1)//2,
      discard_activation = False,
      discard_bias = True,
      ignore_first_det_evt_round = False,
      use_exp_act = use_exp_act,
      **kwargs
    )

  
  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'kernel_parity': self.embedded_kernel.kernel_parity,
        'obs_type': self.embedded_kernel.obs_type,
        'kernel_distance': self.embedded_kernel.kernel_distance,
        'npol': self.embedded_kernel.npol,
        'use_exp_act': self.embedded_kernel.use_exp_act,
        'rounds_first_override': self.embedded_kernel.rounds
      }
    )
    return config


  def call(self, inputs):
    return self.embedded_kernel(inputs)



class RCNNInitialDoubletStateKernel(Layer):
  rounds_first = 2


  @staticmethod
  def set_rounds_first(val):
    RCNNInitialDoubletStateKernel.rounds_first = val
  
  
  def __init__(
      self,
      kernel_parity,
      obs_type, kernel_distance,
      npol = 1,
      use_exp_act = True,
      rounds_first_override = None,
      **kwargs
    ):
    super(RCNNInitialDoubletStateKernel, self).__init__(**kwargs)
    self.embedded_kernel = RCNNEmbeddedKernelChooser.get_kernel(
      kernel_parity = kernel_parity,
      obs_type = obs_type,
      kernel_distance = kernel_distance,
      rounds = RCNNInitialDoubletStateKernel.rounds_first if rounds_first_override is None else rounds_first_override,
      npol = npol,
      do_all_data_qubits = True,
      include_det_bits = True,
      include_det_evts = (RCNNInitialDoubletStateKernel.rounds_first>1),
      n_remove_last_det_evts = (kernel_distance**2 - 1)//2,
      discard_activation = True,
      discard_bias = True,
      ignore_first_det_evt_round = False,
      use_exp_act = use_exp_act,
      **kwargs
    )

    self.n_states = 2
    self.state_correlator = RCNNStateCorrelatorChooser.get_correlator(
      distance = self.embedded_kernel.kernel_distance,
      rounds = self.n_states,
      is_symmetric = self.embedded_kernel.is_symmetric,
      npol = self.embedded_kernel.npol,
      use_exp_act = self.embedded_kernel.use_exp_act
    )

  
  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'kernel_parity': self.embedded_kernel.kernel_parity,
        'obs_type': self.embedded_kernel.obs_type,
        'kernel_distance': self.embedded_kernel.kernel_distance,
        'npol': self.embedded_kernel.npol,
        'use_exp_act': self.embedded_kernel.use_exp_act,
        'rounds_first_override': self.embedded_kernel.rounds
      }
    )
    return config


  def call(self, inputs):
    # Input assumptions:
    # * inputs[0]: z of the previous state, described by exp(z)/(1+exp(z)) = exp(z/2)/(exp(-z/2) + exp(z/2))
    # * inputs[1]: det_bits of the first three rounds
    # * inputs[2]: det_evts of last two rounds
    old_state = inputs[0]
    current_state = self.embedded_kernel(inputs[1:])
    return self.state_correlator([ current_state, old_state ])



class RCNNRecurrenceBaseKernel(Layer):
  def __init__(
      self,
      kernel_parity,
      obs_type, kernel_distance,
      append_name,
      npol = 1,
      use_exp_act = True,
      include_initial_state = False,
      include_prev2_state = False,
      **kwargs
    ):
    super(RCNNRecurrenceBaseKernel, self).__init__(**kwargs)
    self.kernel_parity = kernel_parity
    self.obs_type = obs_type
    self.kernel_distance = kernel_distance
    self.append_name = append_name
    self.rounds = 2
    self.npol = npol
    self.n_ancillas = (self.kernel_distance**2 - 1)
    self.is_symmetric = kernel_parity[0]==0 and kernel_parity[1]==0
    self.use_exp_act = use_exp_act
    self.include_initial_state = include_initial_state
    self.include_prev2_state = include_prev2_state
    label = f"{append_name}_{kernel_parity[0]}_{kernel_parity[1]}"

    # For detector bits, we will embed them into a triplet state and then evaluate weigt products.
    self.embedder_det_bits = TripletStateProbEmbedder(
      self.kernel_distance,
      self.rounds+1,
      self.npol,
      self.is_symmetric
    )
    # For detector events, we can let the CNNKernelWithEmbedding layer take care of the embedding and weight evaluation.
    self.evaluator_det_evts = RCNNEmbeddedKernelChooser.get_kernel(
      kernel_parity = kernel_parity,
      obs_type = obs_type,
      kernel_distance = kernel_distance,
      rounds = self.rounds,
      npol = npol,
      do_all_data_qubits = True,
      include_det_bits = False,
      include_det_evts = True,
      n_remove_last_det_evts = self.n_ancillas//2,
      discard_activation = True, # We are going to add the results directly, so no need for an activation just yet.
      discard_bias = True,
      ignore_first_det_evt_round = True
    )

    num_outputs = None
    if self.is_symmetric:
      num_outputs = (self.kernel_distance**2 + 1)//2
    else:
      num_outputs = self.kernel_distance**2

    # Swap map for the triplet states
    self.kernel_weights_triplet_states_swap_map = get_triplet_states_perround_map(self.kernel_distance, self.rounds-1, self.npol, self.is_symmetric)
    # Weights for the triplet states
    ninput_triplet_states = len(self.kernel_weights_triplet_states_swap_map[0][1])
    self.kernel_weights_triplet_states = self.add_weight(
      name=f"{label}_w_triplet_states",
      shape=[ ninput_triplet_states, num_outputs ],
      initializer='zeros',
      trainable=True
    )

    self.n_states = 2
    if self.include_initial_state:
      self.n_states += 1
    if self.include_prev2_state:
      self.n_states += 1
    self.state_correlator = RCNNStateCorrelatorChooser.get_correlator(
      distance = self.kernel_distance,
      rounds = self.n_states,
      is_symmetric = self.is_symmetric,
      npol = self.npol,
      use_exp_act = self.use_exp_act
    )
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'kernel_parity': self.kernel_parity,
        'obs_type': self.obs_type,
        'kernel_distance': self.kernel_distance,
        'npol': self.npol,
        'use_exp_act': self.use_exp_act
        # No need for include_initial_state or include_prev2_state. These parameters are overridden through daughter classes, which is how the constructor of this class should be called.
      }
    )
    return config


  def get_mapped_weights(self, w, wmap):
    if wmap is None:
      return w
    wgts_mapped = []
    for mm in wmap:
      jout = mm[0]
      ilist = mm[1]
      if ilist is None:
        wgts_mapped.append(w[:,jout])
      else:
        wgts_mapped.append(tf.gather(w[:,jout], ilist))
    return tf.stack(wgts_mapped, axis=1)


  def call(self, inputs):
    # Input assumptions:
    # - If include_initial_state is False:
    # * inputs[0]: z of the previous state, described by exp(z)/(1+exp(z)) = exp(z/2)/(exp(-z/2) + exp(z/2))
    # * inputs[1]: det_bits of the first three rounds
    # * inputs[2]: det_evts of last two rounds
    # - If include_initial_state is True:
    # * inputs[0]: z of the initial state, described by exp(z)/(1+exp(z)) = exp(z/2)/(exp(-z/2) + exp(z/2))
    # * inputs[1]: z of the previous state, described by exp(z)/(1+exp(z)) = exp(z/2)/(exp(-z/2) + exp(z/2))
    # * inputs[2]: det_bits of the first three rounds
    # * inputs[3]: det_evts of last two rounds
    initial_state = None
    prev2_state = None
    ioffset = 0
    if self.include_initial_state:
      initial_state = inputs[ioffset]
      ioffset += 1
    if self.include_prev2_state:
      prev2_state = inputs[ioffset]
      ioffset += 1
    old_state = inputs[ioffset]
    det_bits = inputs[ioffset+1]
    det_evts = inputs[ioffset+2]
    
    n = arrayops_shape(old_state, 0)

    triplet_state = tf.reshape(self.embedder_det_bits(tf.reshape(det_bits, shape=(n, self.rounds+1, -1))), shape=(n, -1))
    recurrence_z = tf.matmul(triplet_state*2-1, self.get_mapped_weights(self.kernel_weights_triplet_states, self.kernel_weights_triplet_states_swap_map))
    recurrence_z += self.evaluator_det_evts(det_evts)

    states = [ recurrence_z, old_state ]
    if prev2_state is not None:
      states.append(prev2_state)
    if initial_state is not None:
      states.append(initial_state)
    return self.state_correlator(states)



# Specializations of recurrent kernel layers:
# - RCNNLeadInKernel: The warmup kernel in a recurrent chain, which facilitates the transition to recurrent triplet state readout.
# - RCNNRecurrenceKernel: The true recurrent kernel in the recurrent chain, which takes over until the final state is reached.
# Both kernels process the previous state and the initial state,
# but only in RCNNRecurrenceKernel is the previous state obtained from another recurrent-style kernel.
class RCNNLeadInKernel(RCNNRecurrenceBaseKernel):
  ignore_initial_state = False


  @classmethod
  def set_ignore_initial_state(cls, flag):
    cls.ignore_initial_state = flag

  
  def __init__(
      self,
      kernel_parity,
      obs_type, kernel_distance,
      npol = 1,
      use_exp_act = True,
      **kwargs
    ):
    super(RCNNLeadInKernel, self).__init__(
      kernel_parity = kernel_parity,
      obs_type = obs_type,
      kernel_distance = kernel_distance,
      append_name = "RCNNLeadInKernel",
      npol = npol,
      use_exp_act = use_exp_act,
      include_initial_state = not RCNNLeadInKernel.ignore_initial_state,
      include_prev2_state = False,
      **kwargs
    )
class RCNNRecurrenceKernel(RCNNRecurrenceBaseKernel):
  ignore_initial_state = False
  ignore_prev2_state = True


  @classmethod
  def set_ignore_initial_state(cls, flag):
    cls.ignore_initial_state = flag


  @classmethod
  def set_ignore_prev2_state(cls, flag):
    cls.ignore_prev2_state = flag


  def __init__(
      self,
      kernel_parity,
      obs_type, kernel_distance,
      npol = 1,
      use_exp_act = True,
      **kwargs
    ):
    super(RCNNRecurrenceKernel, self).__init__(
      kernel_parity = kernel_parity,
      obs_type = obs_type,
      kernel_distance = kernel_distance,
      append_name = "RCNNRecurrenceKernel",
      npol = npol,
      use_exp_act = use_exp_act,
      include_initial_state = not RCNNRecurrenceKernel.ignore_initial_state,
      include_prev2_state = not RCNNRecurrenceKernel.ignore_prev2_state,
      **kwargs
    )



class RCNNFinalStateKernel(Layer):
  ignore_initial_state = False


  @staticmethod
  def set_ignore_initial_state(flag):
    RCNNFinalStateKernel.ignore_initial_state = flag

    
  def __init__(
      self,
      kernel_parity,
      obs_type, kernel_distance,
      npol = 1,
      use_exp_act = True,
      reset_ignore_initial_state = None,
      **kwargs
    ):
    super(RCNNFinalStateKernel, self).__init__(**kwargs)
    self.rounds_last = 2
    self.embedded_kernel = RCNNEmbeddedKernelChooser.get_kernel(
      kernel_parity = kernel_parity,
      obs_type = obs_type,
      kernel_distance = kernel_distance,
      rounds = self.rounds_last,
      npol = npol,
      do_all_data_qubits = True,
      include_det_bits = True,
      include_det_evts = True,
      n_remove_last_det_evts = 0,
      discard_activation = True, # We are going to add the results directly, so no need for an activation just yet.
      discard_bias = True,
      ignore_first_det_evt_round = True,
      use_exp_act = use_exp_act,
      **kwargs
    )
    self.kernel_distance = kernel_distance
    self.use_exp_act = use_exp_act
    self.is_symmetric = self.embedded_kernel.is_symmetric

    # In case we are constructing through a call to from_config(), we need to have a reset functionality for static variables.
    keep_initial = not RCNNFinalStateKernel.ignore_initial_state
    if reset_ignore_initial_state is not None:
      keep_initial = not reset_ignore_initial_state

    self.n_states = 3 if keep_initial else 2
    self.state_correlator = RCNNStateCorrelatorChooser.get_correlator(
      distance = self.kernel_distance,
      rounds = self.n_states,
      is_symmetric = self.is_symmetric,
      npol = npol,
      use_exp_act = self.use_exp_act
    )
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'kernel_parity': self.embedded_kernel.kernel_parity,
        'obs_type': self.embedded_kernel.obs_type,
        'kernel_distance': self.kernel_distance,
        'npol': self.embedded_kernel.npol,
        'use_exp_act': self.use_exp_act,
        'reset_ignore_initial_state': (self.n_states==2)
      }
    )
    return config


  def call(self, inputs):
    # Input assumptions:
    # * inputs[0]: z of the initial state, described by exp(z)/(1+exp(z)) = exp(z/2)/(exp(-z/2) + exp(z/2))
    # * inputs[1]: z of the previous state, described by exp(z)/(1+exp(z)) = exp(z/2)/(exp(-z/2) + exp(z/2))
    # * inputs[2]: det_bits of the last two rounds
    # * inputs[3]: det_evts of last round + last det_evts that ACTUALLY correspond to the full circuit!
    iin = 0
    initial_state = None
    if not RCNNFinalStateKernel.ignore_initial_state:
      initial_state = inputs[iin]
      iin = iin + 1
    old_state = inputs[iin]; iin = iin + 1

    final_z = self.embedded_kernel(inputs[iin:])
    states = [ final_z, old_state ]
    if not RCNNFinalStateKernel.ignore_initial_state:
      states.append(initial_state)
    return self.state_correlator(states)



class RCNNKernelCollector(Layer):
  def __init__(
      self,
      KernelProcessor,
      obs_type, code_distance, kernel_distance,
      npol = 1,
      **kwargs
    ):
    super(RCNNKernelCollector, self).__init__(**kwargs)
    self.obs_type = obs_type
    self.code_distance = code_distance
    self.kernel_distance = kernel_distance
    self.kernel_half_distance = kernel_distance//2
    self.nshifts = self.code_distance - self.kernel_distance + 1
    self.npol = npol

    self.cnn_kernels = []
    self.unique_kernel_types = get_unique_kernel_types(self.kernel_distance, code_distance)
    for kernel_type in self.unique_kernel_types:
      kernel_parity = kernel_type[0]
      self.cnn_kernels.append(
        KernelProcessor(
          kernel_parity,
          self.obs_type,
          self.kernel_distance,
          self.npol,
          use_exp_act = True,
          **kwargs
        )
      )


  def call(self, all_inputs):
    kernel_outputs = [ None for _ in range(self.nshifts**2)]
    for i, cnn_kernel in enumerate(self.cnn_kernels):
      kernel_idxs = self.unique_kernel_types[i][1]
      for k in kernel_idxs:
        kernel_inputs = []
        for inputs in all_inputs:
          kernel_inputs.append(inputs[:,k])
        kernel_outputs[k] = cnn_kernel(kernel_inputs) # Since we use the exponential activation, we can set the result directly.
    return tf.stack(kernel_outputs, axis=0)



class RCNNInitialStateKernelCollector(RCNNKernelCollector):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance,
      npol = 1,
      **kwargs
    ):
    super(RCNNInitialStateKernelCollector, self).__init__(
      RCNNInitialStateKernel,
      obs_type, code_distance, kernel_distance,
      npol,
      **kwargs
    )
class RCNNInitialDoubletStateKernelCollector(RCNNKernelCollector):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance,
      npol = 1,
      **kwargs
    ):
    super(RCNNInitialDoubletStateKernelCollector, self).__init__(
      RCNNInitialDoubletStateKernel,
      obs_type, code_distance, kernel_distance,
      npol,
      **kwargs
    )
class RCNNLeadInKernelCollector(RCNNKernelCollector):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance,
      npol = 1,
      **kwargs
    ):
    super(RCNNLeadInKernelCollector, self).__init__(
      RCNNLeadInKernel,
      obs_type, code_distance, kernel_distance,
      npol,
      **kwargs
    )
class RCNNRecurrenceKernelCollector(RCNNKernelCollector):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance,
      npol = 1,
      **kwargs
    ):
    super(RCNNRecurrenceKernelCollector, self).__init__(
      RCNNRecurrenceKernel,
      obs_type, code_distance, kernel_distance,
      npol,
      **kwargs
    )
class RCNNFinalStateKernelCollector(RCNNKernelCollector):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance,
      npol = 1,
      **kwargs
    ):
    super(RCNNFinalStateKernelCollector, self).__init__(
      RCNNFinalStateKernel,
      obs_type, code_distance, kernel_distance,
      npol,
      **kwargs
    )



class RCNNKernelCombiner(Layer):
  def __init__(
      self,
      kernel_collector,
      obs_type,
      code_distance, kernel_distance,
      npol = 1,
      allow_inversion = False,
      has_nonuniform_response = False,
      **kwargs
    ):
    super(RCNNKernelCombiner, self).__init__(**kwargs)
    self.kernel_collector = kernel_collector(obs_type, code_distance, kernel_distance, npol, **kwargs)
    self.code_distance = code_distance
    self.kernel_distance = kernel_distance
    self.kernel_half_distance = kernel_distance//2
    self.nshifts = self.code_distance - self.kernel_distance + 1
    self.allow_inversion = allow_inversion
    self.has_nonuniform_response = has_nonuniform_response

    self.unique_kernel_types = get_unique_kernel_types(self.kernel_distance, code_distance)
    dqubit_kernel_contribs = [ [] for _ in range(self.code_distance**2) ]
    for shifty in range(self.nshifts):
      for shiftx in range(self.nshifts):
        ikernel = shiftx+shifty*self.nshifts
        ktype = None
        is_symmetric = False
        for iktype, kernel_type in enumerate(self.unique_kernel_types):
          if ikernel in kernel_type[1]:
            ktype = iktype
            if kernel_type[0][0]==0 and kernel_type[0][1]==0:
              is_symmetric = True
            break
        _, _, flip_x, flip_y = get_kernel_parity_flips(self.nshifts, shiftx, shifty)
        for ky in range(-self.kernel_half_distance,self.kernel_half_distance+1):
          iy = ky if not flip_y else -ky
          jy = self.kernel_half_distance+iy
          if shifty+jy<0:
            continue
          for kx in range(-self.kernel_half_distance, self.kernel_half_distance+1):
            ix = kx if not flip_x else -kx
            jx = self.kernel_half_distance+ix
            if shiftx+jx<0:
              continue
            ox = kx
            oy = ky
            idx_kqubit_r = (oy+self.kernel_half_distance)*self.kernel_distance + (ox+self.kernel_half_distance)
            if is_symmetric and (ox<oy or (ox==oy and ox>0)):
              ox = -ox
              oy = -oy
            idx_kqubit = (oy+self.kernel_half_distance)*self.kernel_distance + (ox+self.kernel_half_distance)
            if is_symmetric:
              idx_kqubit = idx_kqubit - (oy+self.kernel_half_distance)*(oy+self.kernel_half_distance+1)//2
            idx_dqubit = (shiftx+jx) + (shifty+jy)*self.code_distance
            found = False
            for dqkcs in dqubit_kernel_contribs[idx_dqubit]:
              if dqkcs[0][0]==ktype and dqkcs[0][1]==idx_kqubit:
                dqkcs[1].append([ikernel, idx_kqubit_r])
                found = True
                break
            if not found:
              dqubit_kernel_contribs[idx_dqubit].append([[ ktype, idx_kqubit ], [ [ikernel, idx_kqubit_r] ]]) # Kernel type index, qubit index within kernel
    for dqkcs in dqubit_kernel_contribs:
      dqkcs.sort(key=lambda x: x[0][0]*self.kernel_distance**2 + x[0][1])
    self.unique_dqubit_kernel_contribs = []
    for idq, dqkcs in enumerate(dqubit_kernel_contribs):
      type_contribs = []
      kernel_idxs = []
      for ddd in dqkcs:
        type_contribs.append(ddd[0])
        kernel_idxs.append(ddd[1])
      found = False
      #print(f"Data qubit {idq} -> ktype={type_contribs}, kernel indices = {kernel_idxs}")
      dq_kernelidx_map = [idq, kernel_idxs]
      for udkc in self.unique_dqubit_kernel_contribs:
        if type_contribs==udkc[0]:
          found = True
          udkc[1].append(dq_kernelidx_map)
          break
      if not found:
        self.unique_dqubit_kernel_contribs.append([type_contribs, [ dq_kernelidx_map ]])

    print(f"Number of unique contributions: {len(self.unique_dqubit_kernel_contribs)}")

    total_nfracs = 0
    total_nphases = 0
    total_inverters = 0
    self.frac_params = []
    self.phase_params = []
    for iudkc, udkc in enumerate(self.unique_dqubit_kernel_contribs):
      #print(f"Kernel type = {udkc[0]}")
      #for uuu in udkc[1]:
      #  print(f"- Data qubit {uuu[0]} maps to kernels {uuu[1]}")
      np = len(udkc[0])
      nfr = np-1
      nph = np*(np-1)//2
      total_nfracs += nfr
      total_nphases += nph
      if self.allow_inversion:
        total_inverters += np
      if nfr>0:
        udkc.append(
          self.add_weight(
            name=f"TranslationFrac_{iudkc}",
            shape=[ nfr ],
            initializer='zeros',
            trainable=True
          )
        )
      else:
        udkc.append(None)
      if nph>0:
        udkc.append(
          self.add_weight(
            name=f"TranslationPhase_{iudkc}",
            shape=[ nph ],
            initializer='zeros',
            trainable=True
          )
        )
      else:
        udkc.append(None)
      if self.allow_inversion:
        udkc.append(
          self.add_weight(
            name=f"Inverter_{iudkc}",
            shape=[ np ],
            initializer=tf.keras.initializers.Constant(0.549306), # = atanh(1/2)
            trainable=True
          )
        )
      else:
        udkc.append(None)

    print(f"Total number of fractions: {total_nfracs}")
    print(f"Total number of phases: {total_nphases}")
    print(f"Total number of inverters: {total_inverters}")
    self.frac_activation = tf.math.sigmoid # We need a value between 0 and 1.
    self.phase_activation = tf.math.tanh # We actually need cos(phi).
    self.inverter_activation = tf.math.tanh # We actually need 2*tanh, but we will deal with it later.

    self.nonuniform_response_adj = None
    if self.has_nonuniform_response:
      self.nonuniform_response_adj = self.add_weight(
        name=f"NonUniformResponseAdj",
        shape=[ 1, self.code_distance**2 ], # No reason to include a constraint on the sum of these coefficients.
        initializer='zeros',
        trainable=True
      )


  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'obs_type': self.kernel_collector.obs_type,
        'code_distance': self.code_distance,
        'kernel_distance': self.kernel_distance,
        'npol': self.kernel_collector.npol,
        'allow_inversion': self.allow_inversion,
        'has_nonuniform_response': self.has_nonuniform_response
        # That is it. We do not need to save the kernel_collector since this is a base class with a collector that is explicitly defined in daughters.
      }
    )
    return config


  def eval_final_data_qubit_pred_layer(self, data_qubit_final_preds):
    # We assume data_qubit_final_preds is flat along axis=1 and represents z values.
    if self.nonuniform_response_adj is not None:
      data_qubit_final_preds = data_qubit_final_preds + tf.repeat(self.nonuniform_response_adj, arrayops_shape(data_qubit_final_preds, 0), axis=0)
    return data_qubit_final_preds


  def call(self, all_inputs):
    kernel_outputs = self.kernel_collector(all_inputs)
    
    data_qubit_idxs_preds = []
    for udkc in self.unique_dqubit_kernel_contribs:
      #kernel_type_contribs = udkc[0]
      data_qubit_idxs = udkc[1]
      frac_params = udkc[2]
      phase_params = udkc[3]
      inverter_params = udkc[4]
      frac_values = None
      two_phase_values = None
      inverter_values = None
      if frac_params is not None:
        frac_values = self.frac_activation(VariableBounds.clip_zlike(frac_params))
        #frac_values = tf.clip_by_value(self.frac_activation(frac_params), 1e-6, 1.-1e-6)
      if phase_params is not None:
        two_phase_values = self.phase_activation(phase_params)*2
      if inverter_params is not None:
        inverter_values = self.inverter_activation(inverter_params)*2

      for idq_idkqs in data_qubit_idxs:
        idq = idq_idkqs[0]
        idkqs = idq_idkqs[1]
        sum_kouts = None
        sum_inputs = []
        for iktype, idkq in enumerate(idkqs):
          #ktype = kernel_type_contribs[iktype]
          kout_list = []
          for ikq_idxkq in idkq:
            ikq = ikq_idxkq[0]
            idxkq = ikq_idxkq[1]
            single_kernout = kernel_outputs[ikq][:,idxkq]
            if inverter_values is not None:
              single_kernout = tf.math.pow(single_kernout, inverter_values[iktype])
            kout_list.append(single_kernout)
          kout = None
          for iik in range(len(kout_list)):
            kki = kout_list[iik]
            if kout is None:
              kout = kki
            else:
              kout += kki
            for jjk in range(iik+1,len(kout_list)):
              kkj = kout_list[jjk]
              kout += tf.math.sqrt(kki*kkj)*2
          kout /= len(kout_list)
          if frac_params is not None:
            frac = None
            for ifrac in range(min(frac_params.shape[0],iktype+1)):
              frac_tmp = frac_values[ifrac]
              if ifrac!=iktype:
                frac_tmp = 1.-frac_tmp
              if frac is None:
                frac = frac_tmp
              else:
                frac = frac*frac_tmp
            kout = kout*frac
          if sum_kouts is None:
            sum_kouts = kout
          else:
            sum_kouts = sum_kouts + kout
          sum_inputs.append(kout)
        n_sum_inputs = len(sum_inputs)
        if phase_params is not None:
          if n_sum_inputs*(n_sum_inputs-1)//2!=phase_params.shape[0]:
            raise RuntimeError(f"Number of phase parameters {phase_params.shape[0]} does not match the number of inputs {n_sum_inputs}.")
          iphase = 0
          for idx_i1 in range(n_sum_inputs):
            for idx_i2 in range(idx_i1+1, n_sum_inputs):
              two_cos_phase = two_phase_values[iphase]
              sum_kouts = sum_kouts + tf.sqrt(sum_inputs[idx_i1]*sum_inputs[idx_i2])*two_cos_phase
              iphase += 1
        sum_kouts = tf.math.log(VariableBounds.clip_exp(sum_kouts))
        #sum_kouts = tf.math.log(tf.clip_by_value(sum_kouts, 1e-9, 1e9))
        data_qubit_idxs_preds.append([idq, sum_kouts])
    data_qubit_idxs_preds.sort()

    data_qubit_final_preds = tf.concat(
      [ tf.reshape(dqp[1], shape=(arrayops_shape(dqp[1], 0),-1)) for dqp in data_qubit_idxs_preds ],
      axis=1
    )
    return self.eval_final_data_qubit_pred_layer(data_qubit_final_preds)



class RCNNInitialStateKernelCombiner(RCNNKernelCombiner):
  def __init__(
      self,
      obs_type,
      code_distance, kernel_distance,
      npol = 1,
      allow_inversion = False,
      has_nonuniform_response = False,
      **kwargs
    ):
    super(RCNNInitialStateKernelCombiner, self).__init__(
      RCNNInitialStateKernelCollector,
      obs_type,
      code_distance, kernel_distance,
      npol,
      allow_inversion,
      has_nonuniform_response,
      **kwargs
    )
class RCNNInitialDoubletStateKernelCombiner(RCNNKernelCombiner):
  def __init__(
      self,
      obs_type,
      code_distance, kernel_distance,
      npol = 1,
      allow_inversion = False,
      has_nonuniform_response = False,
      **kwargs
    ):
    super(RCNNInitialDoubletStateKernelCombiner, self).__init__(
      RCNNInitialDoubletStateKernelCollector,
      obs_type,
      code_distance, kernel_distance,
      npol,
      allow_inversion,
      has_nonuniform_response,
      **kwargs
    )
class RCNNLeadInKernelCombiner(RCNNKernelCombiner):
  def __init__(
      self,
      obs_type,
      code_distance, kernel_distance,
      npol = 1,
      allow_inversion = False,
      has_nonuniform_response = False,
      **kwargs
    ):
    super(RCNNLeadInKernelCombiner, self).__init__(
      RCNNLeadInKernelCollector,
      obs_type,
      code_distance, kernel_distance,
      npol,
      allow_inversion,
      has_nonuniform_response,
      **kwargs
    )
class RCNNRecurrenceKernelCombiner(RCNNKernelCombiner):
  def __init__(
      self,
      obs_type,
      code_distance, kernel_distance,
      npol = 1,
      allow_inversion = False,
      has_nonuniform_response = False,
      **kwargs
    ):
    super(RCNNRecurrenceKernelCombiner, self).__init__(
      RCNNRecurrenceKernelCollector,
      obs_type,
      code_distance, kernel_distance,
      npol,
      allow_inversion,
      has_nonuniform_response,
      **kwargs
    )
class RCNNFinalStateKernelCombiner(RCNNKernelCombiner):
  def __init__(
      self,
      obs_type,
      code_distance, kernel_distance,
      npol = 1,
      allow_inversion = False,
      has_nonuniform_response = False,
      **kwargs
    ):
    super(RCNNFinalStateKernelCombiner, self).__init__(
      RCNNFinalStateKernelCollector,
      obs_type,
      code_distance, kernel_distance,
      npol,
      allow_inversion,
      has_nonuniform_response,
      **kwargs
    )



class StateDecoder(Layer):
  def __init__(
    self,
    code_distance,
    hidden_specs,
    do_all_data_qubits
  ):
    super(StateDecoder, self).__init__()
    self.code_distance = code_distance
    self.hidden_specs = hidden_specs
    self.do_all_data_qubits = do_all_data_qubits

    noutputs = 1 if not self.do_all_data_qubits else self.code_distance**2
    self.layers_decoder = []
    if hidden_specs is not None:
      for hl in hidden_specs:
        if type(hl)==int:
          self.layers_decoder.append(Dense(hl, activation="relu"))
        elif type(hl)==dict:
          is_activation = hl["is_activation"] if "is_activation" in hl else False
          if not is_activation:
            n_nodes = hl["n_nodes"] # Mandatory
            has_activation = hl["has_activation"] if "has_activation" in hl else True
            activation = None
            if has_activation:
              activation = hl["activation"] if "activation" in hl else "relu"
            self.layers_decoder.append(Dense(n_nodes, activation=activation))
          else:
            self.layers_decoder.append(hl["activation"])
    # Last layer needs to translate z-like quantities to probability-like quantities
    if len(self.layers_decoder)>0:
      self.layers_decoder.append(Dense(noutputs, activation="sigmoid"))
    else:
      self.layers_decoder.append(tf.keras.layers.Activation('sigmoid'))
  

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'code_distance': self.code_distance,
        'hidden_specs': self.hidden_specs,
        'do_all_data_qubits': self.do_all_data_qubits
      }
    )
    return config
  

  def call(self, inputs):
    x = inputs
    for layer in self.layers_decoder:
      x = layer(x)
    return x



class FullCNNModel(Model):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance, rounds,
      hidden_specs,
      npol = 1,
      do_all_data_qubits = False,
      extended_kernel_output = True,
      include_det_evts = True,
      include_last_kernel_dets = False,
      include_last_dets = True,
      has_nonuniform_response = False,
      use_translated_kernels = False,
      **kwargs
    ):
    super(FullCNNModel, self).__init__(**kwargs)
    self.obs_type = obs_type
    self.code_distance = code_distance
    self.kernel_distance = kernel_distance
    self.kernel_half_distance = kernel_distance//2
    self.n_kernel_last_det_evts = (self.kernel_distance**2-1)//2
    self.nshifts = self.code_distance - self.kernel_distance + 1
    self.rounds = rounds
    self.npol = npol
    self.do_all_data_qubits = do_all_data_qubits
    self.extended_kernel_output = extended_kernel_output
    self.include_det_evts = include_det_evts
    self.include_last_kernel_dets = include_last_kernel_dets
    self.include_last_dets = include_last_dets
    self.has_nonuniform_response = has_nonuniform_response
    self.use_translated_kernels = use_translated_kernels
    self.hidden_specs = hidden_specs

    self.cnn_kernels = []
    self.unique_kernel_types = get_unique_kernel_types(self.kernel_distance, code_distance)
    for kernel_type in self.unique_kernel_types:
      n_remove_last_dets = 0
      kernel_parity = kernel_type[0]
      if self.include_det_evts:
        if self.include_last_kernel_dets:
          if self.obs_type=="ZL":
            if kernel_parity[0]==0:
              n_remove_last_dets = 2
            elif kernel_parity[0]==1 and self.code_distance>self.kernel_distance:
              n_remove_last_dets = 1
            elif kernel_parity[0]==-1:
              n_remove_last_dets = 1
          elif self.obs_type=="XL":
            if kernel_parity[1]==0:
              n_remove_last_dets = 2
            elif kernel_parity[1]==1 and self.code_distance>self.kernel_distance:
              n_remove_last_dets = 1
            elif kernel_parity[1]==-1:
              n_remove_last_dets = 1
        else:
          n_remove_last_dets = self.n_kernel_last_det_evts

      self.cnn_kernels.append(
        CNNEmbeddedKernelChooser.get_kernel(
          kernel_parity = kernel_parity,
          obs_type = self.obs_type,
          kernel_distance = self.kernel_distance,
          rounds = self.rounds,
          npol = self.npol,
          do_all_data_qubits = self.do_all_data_qubits or self.extended_kernel_output,
          include_det_bits = True,
          include_det_evts = self.include_det_evts,
          n_remove_last_det_evts = n_remove_last_dets
        )
      )
    
    self.translation_coef_transform = None
    self.translation_coef_transform_act = None
    if self.extended_kernel_output and self.use_translated_kernels:
      self.translation_coef_transform = Dense(self.nshifts)
      self.translation_coef_transform_act = tf.keras.layers.Activation('sigmoid')

    dqubit_kernel_contribs = [ [] for _ in range(self.code_distance**2) ]
    for shifty in range(self.nshifts):
      for shiftx in range(self.nshifts):
        ikernel = shiftx+shifty*self.nshifts
        ktype = None
        is_symmetric = False
        for iktype, kernel_type in enumerate(self.unique_kernel_types):
          if ikernel in kernel_type[1]:
            ktype = iktype
            if kernel_type[0][0]==0 and kernel_type[0][1]==0:
              is_symmetric = True
            break
        _, _, flip_x, flip_y = get_kernel_parity_flips(self.nshifts, shiftx, shifty)
        for ky in range(-self.kernel_half_distance,self.kernel_half_distance+1):
          iy = ky if not flip_y else -ky
          jy = self.kernel_half_distance+iy
          if shifty+jy<0:
            continue
          for kx in range(-self.kernel_half_distance,self.kernel_half_distance+1):
            ix = kx if not flip_x else -kx
            jx = self.kernel_half_distance+ix
            if shiftx+jx<0:
              continue
            ox = kx
            oy = ky
            idx_kqubit_r = (oy+self.kernel_half_distance)*self.kernel_distance + (ox+self.kernel_half_distance)
            if is_symmetric and (ox<oy or (ox==oy and ox>0)):
              ox = -ox
              oy = -oy
            idx_kqubit = (oy+self.kernel_half_distance)*self.kernel_distance + (ox+self.kernel_half_distance)
            if is_symmetric:
              idx_kqubit = idx_kqubit - (oy+self.kernel_half_distance)*(oy+self.kernel_half_distance+1)//2
            idx_dqubit = (shiftx+jx) + (shifty+jy)*self.code_distance
            found = False
            for dqkcs in dqubit_kernel_contribs[idx_dqubit]:
              if dqkcs[0][0]==ktype and dqkcs[0][1]==idx_kqubit:
                dqkcs[1].append([ikernel, idx_kqubit_r])
                found = True
                break
            if not found:
              dqubit_kernel_contribs[idx_dqubit].append([[ ktype, idx_kqubit ], [ [ikernel, idx_kqubit_r] ]]) # Kernel type index, qubit index within kernel
    for dqkcs in dqubit_kernel_contribs:
      dqkcs.sort(key=lambda x: x[0][0]*self.kernel_distance**2 + x[0][1])
    self.unique_dqubit_kernel_contribs = []
    for idq, dqkcs in enumerate(dqubit_kernel_contribs):
      type_contribs = []
      kernel_idxs = []
      for ddd in dqkcs:
        type_contribs.append(ddd[0])
        kernel_idxs.append(ddd[1])
      found = False
      #print(f"Data qubit {idq} -> ktype={type_contribs}, kernel indices = {kernel_idxs}")
      dq_kernelidx_map = [idq, kernel_idxs]
      for udkc in self.unique_dqubit_kernel_contribs:
        if type_contribs==udkc[0]:
          found = True
          udkc[1].append(dq_kernelidx_map)
          break
      if not found:
        self.unique_dqubit_kernel_contribs.append([type_contribs, [ dq_kernelidx_map ]])

    print(f"Number of unique contributions: {len(self.unique_dqubit_kernel_contribs)}")

    total_nfracs = 0
    total_nphases = 0
    self.frac_params = []
    self.phase_params = []
    for iudkc, udkc in enumerate(self.unique_dqubit_kernel_contribs):
      #print(f"Kernel type = {udkc[0]}")
      #for uuu in udkc[1]:
      #  print(f"- Data qubit {uuu[0]} maps to kernels {uuu[1]}")
      np = len(udkc[0])
      nfr = np-1
      nph = np*(np-1)//2
      total_nfracs += nfr
      total_nphases += nph
      if nfr>0:
        udkc.append(
          self.add_weight(
            name=f"TranslationFrac_{iudkc}",
            shape=[ nfr ],
            initializer='zeros',
            trainable=True
          )
        )
      else:
        udkc.append(None)
      if nph>0:
        udkc.append(
          self.add_weight(
            name=f"TranslationPhase_{iudkc}",
            shape=[ nph ],
            initializer='zeros',
            trainable=True
          )
        )
      else:
        udkc.append(None)
    print(f"Total number of fractions: {total_nfracs}")
    print(f"Total number of phases: {total_nphases}")
    self.frac_activation = tf.keras.activations.sigmoid # We actually need cos(phi), so there is an activation function
    self.phase_activation = tf.keras.activations.tanh # We actually need cos(phi), so there is an activation function

    self.nonuniform_response_adj = None
    self.nonuniform_response_kerneval_adj = None
    if self.has_nonuniform_response:
      self.nonuniform_response_adj = self.add_weight(
        name=f"NonUniformResponseAdj",
        shape=[ 1, self.code_distance**2 ],
        initializer='zeros',
        trainable=True
      )
      self.nonuniform_response_kerneval_adj = self.add_weight(
        name=f"NonUniformResponseKernelEvalAdj",
        shape=[ self.nshifts**2 ],
        initializer='zeros',
        trainable=True
      )

    self.noutputs_final = 1 if not self.do_all_data_qubits else self.code_distance**2
    self.first_dense_nout = None
    self.upper_layers = []
    for hs in self.hidden_specs:
      if self.first_dense_nout is None:
        self.first_dense_nout = hs
      self.upper_layers.append(Dense(hs))
      self.upper_layers.append(tf.keras.layers.Activation('relu'))

    if self.first_dense_nout is None: # only happens if there are no hidden layers
      self.first_dense_nout = self.noutputs_final
    self.data_qubit_pred_eval_layer = Dense(self.first_dense_nout, use_bias=False)
    self.upper_layers.append(Dense(self.noutputs_final))
    self.upper_layers.append(tf.keras.layers.Activation('sigmoid'))


  def get_config(self):
    config = super().get_config()
    config.update(
      {
        'obs_type': self.obs_type,
        'code_distance': self.code_distance,
        'kernel_distance': self.kernel_distance,
        'rounds': self.rounds,
        'hidden_specs': self.hidden_specs,
        'npol': self.npol,
        'do_all_data_qubits': self.do_all_data_qubits,
        'extended_kernel_output': self.extended_kernel_output,
        'include_det_evts': self.include_det_evts,
        'include_last_kernel_dets': self.include_last_kernel_dets,
        'include_last_dets': self.include_last_dets,
        'has_nonuniform_response': self.has_nonuniform_response,
        'use_translated_kernels': self.use_translated_kernels
      }
    )
    return config


  def eval_final_data_qubit_pred_layer(self, data_qubit_final_preds):
    # We assume data_qubit_final_preds is flat along axis=1
    if self.nonuniform_response_adj is not None:
      data_qubit_final_preds = data_qubit_final_preds + tf.repeat(self.nonuniform_response_adj, arrayops_shape(data_qubit_final_preds, 0), axis=0)
    return self.data_qubit_pred_eval_layer(data_qubit_final_preds)


  def call(self, all_inputs):
    det_bits = all_inputs[0]
    det_evts = all_inputs[1]
    translation_coefs = all_inputs[2]
    final_det_evts = all_inputs[3]

    kernel_outputs = dict()
    predictor_inputs = []

    predictor_inputs.append(tf.cast(translation_coefs, tf.float32))
    if self.include_det_evts and self.include_last_dets:
      predictor_inputs.append(tf.cast(final_det_evts, tf.float32))
    predictor_inputs = tf.concat(predictor_inputs, axis=1)

    translation_coefs_transformed = None
    if self.extended_kernel_output and self.use_translated_kernels:
      translation_coefs_transformed = self.translation_coef_transform_act(self.translation_coef_transform(translation_coefs))
      translation_coefs_transformed = tf.reshape(translation_coefs_transformed, (arrayops_shape(translation_coefs_transformed, 0), arrayops_shape(translation_coefs_transformed, 1), 1))
      if self.extended_kernel_output:
        translation_coefs_transformed = tf.repeat(translation_coefs_transformed, self.kernel_distance**2, axis=2)

    kernel_eval_adj = None
    if self.nonuniform_response_kerneval_adj is not None:
      kernel_eval_adj = tf.math.exp(self.nonuniform_response_kerneval_adj)

    for i, cnn_kernel in enumerate(self.cnn_kernels):
      kernel_parity = self.unique_kernel_types[i][0]
      kernel_idxs = self.unique_kernel_types[i][1]
      for k in kernel_idxs:
        kernel_input = None
        det_bits_kernel = det_bits[:,k]
        if self.include_det_evts:
          det_evts_kernel = det_evts[:,k,0:-self.n_kernel_last_det_evts]
          if self.include_last_kernel_dets:
            det_evts_kernel_end = det_evts[:,k,-self.n_kernel_last_det_evts:]
            if self.obs_type=="ZL":
              if kernel_parity[0]==0:
                det_evts_kernel_end = det_evts_kernel_end[:,1:-1]
              elif kernel_parity[0]==1 and self.code_distance>self.kernel_distance:
                det_evts_kernel_end = det_evts_kernel_end[:,:-1]
              elif kernel_parity[0]==-1:
                det_evts_kernel_end = det_evts_kernel_end[:,1:]
            elif self.obs_type=="XL":
              if kernel_parity[1]==0:
                det_evts_kernel_end = det_evts_kernel_end[:,1:-1]
              elif kernel_parity[1]==1 and self.code_distance>self.kernel_distance:
                det_evts_kernel_end = det_evts_kernel_end[:,:-1]
              elif kernel_parity[1]==-1:
                det_evts_kernel_end = det_evts_kernel_end[:,1:]
            det_evts_kernel = tf.concat([det_evts_kernel, det_evts_kernel_end], axis=1)
          kernel_input = [ det_bits_kernel, det_evts_kernel ]
        else:
          kernel_input = [ det_bits_kernel ]
        kernel_output = cnn_kernel(kernel_input)
        if self.extended_kernel_output and self.use_translated_kernels:
          shift_x = k % self.nshifts
          shift_y = k // self.nshifts
          k_shift = shift_y if self.obs_type=="ZL" else shift_x
          ksh1 = arrayops_shape(kernel_output, 1)
          kernel_output = tf.math.pow(
              kernel_output,
              1.-translation_coefs_transformed[:,k_shift,0:ksh1]
            )*tf.math.pow(
              1.-kernel_output,
              translation_coefs_transformed[:,k_shift,0:ksh1]
            )
        kernel_output = VariableBounds.clip_prob(kernel_output)
        #kernel_output = tf.clip_by_value(kernel_output, 1e-6, 1.-1e-6)
        kernel_output = kernel_output/(1.-kernel_output)
        if kernel_eval_adj is not None:
          kernel_output = kernel_output * kernel_eval_adj[k]
        kernel_outputs[k] = [i, kernel_output] # [Kernel unique type index, transformed kernel output]
    
    data_qubit_idxs_preds = []
    for udkc in self.unique_dqubit_kernel_contribs:
      kernel_type_contribs = udkc[0]
      data_qubit_idxs = udkc[1]
      frac_params = udkc[2]
      phase_params = udkc[3]
      frac_values = None
      phase_values = None
      if frac_params is not None:
        frac_values = self.frac_activation(VariableBounds.clip_zlike(frac_params))
        #frac_values = tf.clip_by_value(self.frac_activation(frac_params), 1e-6, 1.-1e-6)
      if phase_params is not None:
        phase_values = self.phase_activation(phase_params)

      for idq_idkqs in data_qubit_idxs:
        idq = idq_idkqs[0]
        idkqs = idq_idkqs[1]
        sum_kouts = None
        sum_inputs = []
        for iktype, idkq in enumerate(idkqs):
          #ktype = kernel_type_contribs[iktype]
          kout = None
          for ikq_idxkq in idkq:
            ikq = ikq_idxkq[0]
            idxkq = ikq_idxkq[1]
            if kout is None:
              kout = kernel_outputs[ikq][1][:,idxkq]
            else:
              kout = kout + kernel_outputs[ikq][1][:,idxkq]
          if frac_params is not None:
            frac = None
            for ifrac in range(min(frac_params.shape[0],iktype+1)):
              frac_tmp = frac_values[ifrac]
              if ifrac!=iktype:
                frac_tmp = 1.-frac_tmp
              if frac is None:
                frac = frac_tmp
              else:
                frac = frac*frac_tmp
            kout = kout*frac
          if sum_kouts is None:
            sum_kouts = kout
          else:
            sum_kouts = sum_kouts + kout
          sum_inputs.append(kout)
        n_sum_inputs = len(sum_inputs)
        if phase_params is not None:
          if n_sum_inputs*(n_sum_inputs-1)//2!=phase_params.shape[0]:
            raise RuntimeError(f"Number of phase parameters {phase_params.shape[0]} does not match the number of inputs {n_sum_inputs}.")
          iphase = 0
          for idx_i1 in range(n_sum_inputs):
            for idx_i2 in range(idx_i1+1, n_sum_inputs):
              cos_phase = phase_values[iphase]
              iphase += 1
              sum_kouts = sum_kouts + 2*tf.sqrt(sum_inputs[idx_i1]*sum_inputs[idx_i2])*cos_phase
        #sum_kouts = sum_kouts/(1.+sum_kouts)
        sum_kouts = tf.math.log(VariableBounds.clip_exp(sum_kouts))
        #sum_kouts = tf.math.log(tf.clip_by_value(sum_kouts, 1e-9, 1e9))
        data_qubit_idxs_preds.append([idq, sum_kouts])
    data_qubit_idxs_preds.sort()

    data_qubit_final_preds = tf.concat(
      [ tf.reshape(dqp[1], shape=(arrayops_shape(dqp[1], 0),-1)) for dqp in data_qubit_idxs_preds ],
      axis=1
    )
    eval_dqubit_preds_layer = None

    x = predictor_inputs
    for ll in self.upper_layers:
      x = ll(x)
      if eval_dqubit_preds_layer is None:
        eval_dqubit_preds_layer = self.eval_final_data_qubit_pred_layer(data_qubit_final_preds)
        x = x + eval_dqubit_preds_layer
    return x



class FullRCNNModelEnums:
  prob_no_acc = 0
  prob_undecoded_acc = 1
  prob_decoded_acc = 2

  kInitialStateMode_InitialState = 0
  kInitialStateMode_MinusTwo = 1
  kInitialStateMode_InitialState_MinusTwo = 2



class FullRCNNModel(Model):
  prob_no_acc = 0
  prob_undecoded_acc = 1
  prob_decoded_acc = 2
  
  
  def __init__(
      self,
      obs_type, code_distance, kernel_distance, rounds,
      hidden_specs,
      npol = 1,
      stop_round = None, # One could pass None, -1, or rounds+1 to disable early stopping before the final round.
      has_nonuniform_response = False,
      do_all_data_qubits = False,
      return_all_rounds = False,
      initial_state_mode = FullRCNNModelEnums.kInitialStateMode_InitialState,
      separate_first_round = False,
      allow_combiner_inversion = False,
      decode_per_state_layer = False,
      is_grouped = False,
      probability_accumulation_mode = FullRCNNModelEnums.prob_no_acc,
      **kwargs
    ):
    """
    Initialize the FullRCNNModel object.
    Arguments:
    - obs_type: Type of observable, "ZL" or "XL".
    - code_distance: Distance of the full surface code.
    - kernel_distance: Distance of the kernel.
    - rounds: Number of rounds to run the model.
    - hidden_specs: List of hidden layer specifications.
      Each element can be an integer or a dictionary.
      If it is an integer, it is the number of nodes in a hidden layer.
      If it is a dictionary,
        it must have the key "n_nodes".
        It may further have the keys
        "is_activation" (default False),
        "has_activation" (default True), and
        "activation" (default "relu").
    - npol: Whether we use a linear (=1) or quadratic (>1) relationship between spatial and temporal coordinates.
    - stop_round: Round at which to stop the model. If None, -1, or rounds+1, the model will run for all rounds.
      Valid values are 2<=stop_round<=rounds, and specifying anything in this range will disable the final state layer.
    - has_nonuniform_response: Whether to include a non-uniform response adjustment.
    - do_all_data_qubits: Whether to output all data qubit predictions over the full surface code.
    - return_all_rounds: Whether to return predictions for all rounds.
    - initial_state_mode: By default (option kInitialStateMode_InitialState),
      the last state state correlators receive is the initial state.
      One can request the last state to be the second to previous state (option kInitialStateMode_MinusTwo),
      or elect to pass both (option kInitialStateMode_InitialState_MinusTwo).
    - separate_first_round: By default, there is no prediction for the first round. If this flag is true, there will be one.
    - allow_combiner_inversion: Whether to allow the combiner to include the option to invert the kernel outputs. This option is disabled by default.
    - decode_per_state_layer: By default, we use one type of a decoder layer to decode all states.
      If this flag is True, separate decoder layers will be used for each state type.
    - probability_accumulation_mode: Probability accumulation mode.
      By default (prob_no_acc), the probability of an error is predicted using
        p'(z) = sigmoid(z)
      for the z of that layer.
      If prob_undecoded_acc is chosen, an effective z' is computed before applying the decoding layer such that
        z'[t=0] = z[t=0],
        p'(z'[t]) = p(z[t])*(1-p'(z'[t-1])) + p'(z'[t-1])*(1-p(z[t])) for t>0,
      and decoding is performed over z'[t].
      If, instead, prob_decoded_acc is chosen, the final probability is computed after applying the decoding layer with
        p'[t=0] = p(z[t=0]),
        p'[t] = p(z[t])*(1-p'[t-1]) + p'[t-1]*(1-p(z[t]) for t>0,
      making p(z[t]) from the decoded layer output the evolution probability for each time step t.
    - kwargs: Additional arguments to pass to the Model class.
    Output:
    - FullRCNNModel object, which inherits from the Model class.
    """
    super(FullRCNNModel, self).__init__(**kwargs)
    self.obs_type = obs_type
    self.code_distance = code_distance
    self.kernel_distance = kernel_distance
    self.kernel_half_distance = kernel_distance//2
    self.kernel_n_ancillas = kernel_distance**2-1
    self.kernel_half_n_ancillas = self.kernel_n_ancillas//2
    self.n_last_det_evts = (self.code_distance**2-1)//2
    if self.obs_type=="XL":
      # For even-d surface codes, the number of Z and X measure qubits are not the same,
      # so we have to recompute by subtracting from the total number of measure qubits.
      self.n_last_det_evts = self.code_distance**2 - 1 - self.n_last_det_evts
    self.n_kernel_last_det_evts = (self.kernel_distance**2-1)//2
    self.nshifts = self.code_distance - self.kernel_distance + 1
    self.rounds = rounds
    self.hidden_specs = hidden_specs
    self.npol = npol
    self.stop_round = stop_round
    if self.stop_round is not None and (self.stop_round>self.rounds or self.stop_round<0):
      self.stop_round = None
    self.has_nonuniform_response = has_nonuniform_response
    self.do_all_data_qubits = do_all_data_qubits
    self.return_all_rounds = return_all_rounds
    self.initial_state_mode = initial_state_mode
    self.separate_first_round = separate_first_round
    self.allow_combiner_inversion = allow_combiner_inversion
    self.probability_accumulation_mode = probability_accumulation_mode
    self.decode_per_state_layer = decode_per_state_layer and (self.return_all_rounds or self.probability_accumulation_mode==FullRCNNModelEnums.prob_decoded_acc)
    self.is_grouped = is_grouped

    self.first_round_offset = (0 if self.separate_first_round else 1)

    RCNNInitialStateKernel.set_rounds_first(2 if not self.separate_first_round else 1)
    RCNNLeadInKernel.set_ignore_initial_state(not self.separate_first_round)
    RCNNRecurrenceKernel.set_ignore_prev2_state(self.initial_state_mode != FullRCNNModelEnums.kInitialStateMode_InitialState_MinusTwo)
    RCNNFinalStateKernel.set_ignore_initial_state(self.rounds<=1+self.first_round_offset)

    if self.rounds<1+self.first_round_offset:
      raise RuntimeError(f"The number of rounds must be at least {1+self.first_round_offset}.")
    if self.stop_round is not None and self.stop_round<1+self.first_round_offset:
      raise RuntimeError(f"The stop round must be at least {1+self.first_round_offset}.")

    self.state_shift_map = []
    for shifty in range(self.nshifts):
      for shiftx in range(self.nshifts):
        this_shift_map = []
        for iy in range(-self.kernel_half_distance,self.kernel_half_distance+1):
          jy = self.kernel_half_distance + shifty + iy
          for ix in range(-self.kernel_half_distance,self.kernel_half_distance+1):
            jx = self.kernel_half_distance + shiftx + ix
            this_shift_map.append(jx + jy*self.code_distance)
        self.state_shift_map.append(this_shift_map)

    self.state_decoders = []
    self.layer_initial_state = RCNNInitialStateKernelCombiner(
      obs_type = self.obs_type,
      code_distance = self.code_distance,
      kernel_distance = self.kernel_distance,
      npol = self.npol,
      allow_inversion = self.allow_combiner_inversion,
      has_nonuniform_response = self.has_nonuniform_response
    )
    self.state_decoders.append(
      StateDecoder(
        code_distance = self.code_distance,
        hidden_specs = self.hidden_specs,
        do_all_data_qubits = self.do_all_data_qubits
      )
    )
    self.layer_initial_doublet = None
    self.layer_lead_in = None
    self.layer_recurrence = None
    self.layer_final_state = None
    if self.rounds>1:
      if self.separate_first_round:
        self.layer_initial_doublet = RCNNInitialDoubletStateKernelCombiner(
          obs_type = self.obs_type,
          code_distance = self.code_distance,
          kernel_distance = self.kernel_distance,
          npol = self.npol,
          allow_inversion = self.allow_combiner_inversion,
          has_nonuniform_response = self.has_nonuniform_response
        )
        if self.decode_per_state_layer:
          self.state_decoders.append(
            StateDecoder(
              code_distance = self.code_distance,
              hidden_specs = self.hidden_specs,
              do_all_data_qubits = self.do_all_data_qubits
            )
          )
      if self.rounds>2:
        self.layer_lead_in = RCNNLeadInKernelCombiner(
          obs_type = self.obs_type,
          code_distance = self.code_distance,
          kernel_distance = self.kernel_distance,
          npol = self.npol,
          allow_inversion = self.allow_combiner_inversion,
          has_nonuniform_response = self.has_nonuniform_response
        )
        if self.decode_per_state_layer:
          self.state_decoders.append(
            StateDecoder(
              code_distance = self.code_distance,
              hidden_specs = self.hidden_specs,
              do_all_data_qubits = self.do_all_data_qubits
            )
          )
        if self.rounds>3:
          self.layer_recurrence = RCNNRecurrenceKernelCombiner(
            obs_type = self.obs_type,
            code_distance = self.code_distance,
            kernel_distance = self.kernel_distance,
            npol = self.npol,
            allow_inversion = self.allow_combiner_inversion,
            has_nonuniform_response = self.has_nonuniform_response
          )
          if self.decode_per_state_layer:
            self.state_decoders.append(
              StateDecoder(
                code_distance = self.code_distance,
                hidden_specs = self.hidden_specs,
                do_all_data_qubits = self.do_all_data_qubits
              )
            )
    if self.stop_round is None:
      self.layer_final_state = RCNNFinalStateKernelCombiner(
        obs_type = self.obs_type,
        code_distance = self.code_distance,
        kernel_distance = self.kernel_distance,
        npol = self.npol,
        allow_inversion = self.allow_combiner_inversion,
        has_nonuniform_response = self.has_nonuniform_response
      )
      if self.decode_per_state_layer:
        self.state_decoders.append(
          StateDecoder(
            code_distance = self.code_distance,
            hidden_specs = self.hidden_specs,
            do_all_data_qubits = self.do_all_data_qubits
          )
        )

    self.decoded_outputs = []


  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "obs_type": self.obs_type,
        "code_distance": self.code_distance,
        "kernel_distance": self.kernel_distance,
        "rounds": self.rounds,
        "hidden_specs": self.hidden_specs,
        "npol": self.npol,
        "stop_round": self.stop_round,
        "has_nonuniform_response": self.has_nonuniform_response,
        "do_all_data_qubits": self.do_all_data_qubits,
        "return_all_rounds": self.return_all_rounds,
        "initial_state_mode": self.initial_state_mode,
        "separate_first_round": self.separate_first_round,
        "allow_combiner_inversion": self.allow_combiner_inversion,
        "decode_per_state_layer": self.decode_per_state_layer,
        "probability_accumulation_mode": self.probability_accumulation_mode,
        "is_grouped": self.is_grouped,
      }
    )
    return config


  def set_rounds(self, rounds):
    """
    Set the number of rounds for the model.
    The purpose of this function is to be able to train the model on a smaller number of rounds
    and then evaluate it on a data with more rounds.
    That should work given that this is a recurrent architecture.
    """
    self.rounds = rounds
    if self.rounds<1+self.first_round_offset:
      raise ValueError(f"Invalid rounds value {self.rounds}. Must be at least {1+self.first_round_offset}.")
    if self.stop_round is not None:
      if self.stop_round>self.rounds:
        raise ValueError(f"Invalid stop_round value {self.stop_round}. Please also reset stop_round.")
      else:
        print(f"WARNING: stop_round={self.stop_round} might need to be reset.")


  def set_stop_round(self, stop_round):
    """
    Set the stop round for the model.
    If stop_round is None, the model will run for all rounds.
    Otherwise, if it is in the range [1, rounds], the model will run up to that round.
    Any valid integer value will disable the final state layer.
    """
    self.stop_round = stop_round
    if self.stop_round is not None and (self.stop_round>self.rounds or self.stop_round<1+self.first_round_offset):
      raise ValueError(f"Invalid stop_round value {self.stop_round}. Must be within [{1+self.first_round_offset}, {self.rounds}].")


  def set_probability_accumulation_mode(self, mode):
    """
    Set the probability accumulation mode.
    """
    self.probability_accumulation_mode = mode


  def set_initial_state_mode(self, mode):
    """
    Set the initial state mode.
    """
    self.initial_state_mode = mode


  def decode_state(self, psi, ilayer):
    """
    Decode the z-like state psi to a probability-like quantity.
    """
    return self.state_decoders[ilayer](psi)


  def regroup_state_by_kernel(self, psi):
    """
    Regroup the state psi by kernel strides.
    """
    res = []
    for rmap in self.state_shift_map:
      res.append(tf.gather(psi, rmap, axis=1))
    return tf.stack(res, axis=1)

  
  def get_grouped_det_bits_and_evts_w_final(self, all_inputs):
    """
    Group the stabilizer measurements and detector events into kernels if this is not done already.
    See the call() function for further details on the conventions of the function argument all_inputs.
    """
    if self.is_grouped:
      # In this case, the inputs are already grouped by kernel strides.
      # Faster in training, but less memory-efficient.
      return all_inputs[0], all_inputs[1]
    else:
      # Here, we group the inputs by kernel strides ourselves.
      # Slower in training, but we do not have to worry about storing giant sets of duplicated information.
      # (A much more memory-efficient way would be to operate over pointers - sigh...)
      binary_t, _, idx_t, _ = get_types(self.code_distance, self.rounds, self.kernel_distance)
      use_TF = type(all_inputs[0])!=np.ndarray
      features_det_bits, _, _, _ = group_det_bits_kxk(
        det_bits_dxd = (all_inputs[0] if not use_TF else tf.cast(all_inputs[0], tf.int16)),
        d=self.code_distance, r=self.rounds, k=self.kernel_distance,
        use_rotated_z=(self.obs_type=="ZL"),
        data_bits_dxd=None,
        binary_t=binary_t,
        idx_t=idx_t,
        make_translation_map=False
      )
      features_det_evts = translate_det_bits_to_det_evts(
        d=self.code_distance, r=self.rounds, k=self.kernel_distance,
        obs_type=self.obs_type,
        det_bits_kxk_all=features_det_bits,
        final_det_evts=(all_inputs[1][:, -self.n_last_det_evts:] if not use_TF else tf.cast(all_inputs[1][:, -self.n_last_det_evts:], tf.int16))
      )
      features_det_bits = arrayops_swapaxes(features_det_bits, axis1=0, axis2=1, perm=[1, 0, 2])
      features_det_evts = arrayops_swapaxes(features_det_evts, axis1=0, axis2=1, perm=[1, 0, 2])
      return features_det_bits, features_det_evts


  def call(self, all_inputs):
    """
    Run the model for all rounds and return the final prediction.
    Let's denote the number of rounds with r, the code distance with d, and the kernel distance with k for the rest of the description.
    Arguments:
    - all_inputs: A list of input tensors. There are two possible conventions, determined by the shape of the inputs.
      1) Inputs preprocessed for kernel groupings:
      * all_inputs[0] = det_bits[batch size : kernel stride size : r*(k^2-1)]
        => Stabilizer measurements groupe by kernel strides.
      * all_inputs[1] = det_evts[batch size : kernel stride size : (k^2-1)/2 + (r-1)*(k^2-1) + (k^2-1)/2]
        => Detectors events grouped by kernel strides.
           Note the way the last dimension is written.
           While it is equivalent to r*(k^2-1), we highlight the order of special first and last detector events.
           Note that in contrast to FullCNNModel, the last kernel detector events are from the actual final detectors of the full surface code,
           not the reinterpretation of all_inputs[0] per kernel stride.
      2) Inputs that require kernel groupings:
      * all_inputs[0] = det_bits[batch size : r*(d^2-1)]
        => All stabilizer measurements.
      * all_inputs[1] = det_evts[batch size : r*(d^2-1)]
        => All detector events.
    Output:
    An array of probabilities of size [batch size, R], where R is
    - 1 if do_all_data_qubits is False and return_all_rounds is False.
      The returned value corresponds to the last round evaluated.
    - d^2 if do_all_data_qubits is True and return_all_rounds is False.
      The returned value corresponds to the last round evaluated.
    - r if do_all_data_qubits is False, return_all_rounds is True, and stop_round is None.
      Here, the correspondence between actual (r+1) states and the returned array
      is the map (actual state index, array index) = {(1, 0), (2, 1)), ..., (r, r-1)}.
    - r*d^2 if do_all_data_qubits is True and return_all_rounds is True, and stop_round is None.
    - If stop_round has a value, the output is the same as the above cases, but with the last round now capped by stop_round.
      Please note that specifying any valid 2<=stop_round<=r will avoid running the final state layer.
    If separate_first_round, assume (small-case) r -> r+1 in the returned result from the discussion above
    since we would also be predicting the state of the first round.
    """
    # det_evts_w_final is a modified per-kernel det_evts array
    # that includes real final det_evts
    # instead of kernel reinterpretations of per-kernel det_bits.
    det_bits, det_evts_w_final = self.get_grouped_det_bits_and_evts_w_final(all_inputs)

    # Reset decoded output
    self.decoded_outputs = []

    psi_list = []
    idecoder = 0 if self.decode_per_state_layer else -1
    for r in range(self.rounds+1-self.first_round_offset):
      this_layer = None
      this_bits = None
      this_evts = None
      prev_state = None
      prev2_state = None
      initial_state = None
      if r==0:
        this_layer = self.layer_initial_state
        this_bits = det_bits[:,:,:self.kernel_n_ancillas*(1+self.first_round_offset)]
        this_evts = None if self.separate_first_round else det_evts_w_final[:,:,:self.kernel_half_n_ancillas*3]
      elif r==1 and self.separate_first_round:
        this_layer = self.layer_initial_doublet
        this_bits = det_bits[:,:,:self.kernel_n_ancillas*2]
        this_evts = det_evts_w_final[:,:,:self.kernel_half_n_ancillas*3]
        if self.decode_per_state_layer:
          idecoder += 1
      elif r<self.rounds-self.first_round_offset:
        this_layer = self.layer_lead_in if r==2-self.first_round_offset else self.layer_recurrence
        this_bits = det_bits[:,:,self.kernel_n_ancillas*(r-2+self.first_round_offset):self.kernel_n_ancillas*(r+1+self.first_round_offset)]
        this_evts = det_evts_w_final[:,:,self.kernel_half_n_ancillas + self.kernel_n_ancillas*(r-2+self.first_round_offset):self.kernel_half_n_ancillas + self.kernel_n_ancillas*(r+self.first_round_offset)]
        if self.decode_per_state_layer and (r==2-self.first_round_offset or r==3-self.first_round_offset):
          idecoder += 1
      else:
        this_layer = self.layer_final_state
        this_bits = det_bits[:,:,-self.kernel_n_ancillas*2:]
        this_evts = det_evts_w_final[:,:,-self.kernel_half_n_ancillas*3:]
        if self.decode_per_state_layer:
          idecoder += 1

      if r>0:
        prev_state = psi_list[-1]
        if r>1:
          if self.initial_state_mode == FullRCNNModelEnums.kInitialStateMode_InitialState or self.initial_state_mode == FullRCNNModelEnums.kInitialStateMode_InitialState_MinusTwo:
            initial_state = psi_list[0]
          if self.initial_state_mode == FullRCNNModelEnums.kInitialStateMode_MinusTwo or self.initial_state_mode == FullRCNNModelEnums.kInitialStateMode_InitialState_MinusTwo:
            prev2_state = psi_list[-2]
          # Special case for the lead-in and final layers
          if initial_state is not None and prev2_state is not None and (r==2-self.first_round_offset or r==self.rounds-self.first_round_offset):
            prev2_state = None
      
      this_inputs = []
      if initial_state is not None:
        this_inputs.append(self.regroup_state_by_kernel(initial_state))
      if prev2_state is not None:
        this_inputs.append(self.regroup_state_by_kernel(prev2_state))
      if prev_state is not None:
        this_inputs.append(self.regroup_state_by_kernel(prev_state))
      if this_bits is not None:
        this_inputs.append(this_bits)
      if this_evts is not None:
        this_inputs.append(this_evts)

      this_psi = this_layer(this_inputs)
      if self.probability_accumulation_mode==FullRCNNModelEnums.prob_undecoded_acc and prev_state is not None:
        this_p = tf.math.sigmoid(this_psi)
        prev_p = tf.math.sigmoid(prev_state)
        this_p = VariableBounds.clip_prob(this_p*(-prev_p+1) + (-this_p+1)*prev_p)
        this_psi = tf.math.log(this_p/(-this_p+1))
      psi_list.append(this_psi)
      if self.return_all_rounds or self.probability_accumulation_mode==FullRCNNModelEnums.prob_decoded_acc:
        self.decoded_outputs.append(self.decode_state(this_psi, idecoder))

      if self.stop_round is not None and r==self.stop_round-1-self.first_round_offset:
        break

    res = None
    if self.probability_accumulation_mode==FullRCNNModelEnums.prob_decoded_acc:
      for iout in range(1, len(self.decoded_outputs)):
        self.decoded_outputs[iout] = self.decoded_outputs[iout]*(-self.decoded_outputs[iout-1]+1) + self.decoded_outputs[iout-1]*(-self.decoded_outputs[iout]+1)
    if self.return_all_rounds:
      res = tf.stack(self.decoded_outputs, axis=1)
      res = tf.reshape(res, (arrayops_shape(res, 0), -1))
    elif self.probability_accumulation_mode==FullRCNNModelEnums.prob_decoded_acc:
      res = self.decoded_outputs[-1]
    else:
      res = self.decode_state(psi_list[-1], -1)
    return res
