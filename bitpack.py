def pack_bits(ev_bits, nbits, packed_t):
  res: packed_t = 0
  for i in range(nbits):
    res = res | (packed_t(ev_bits[i]) << i)
  return res

def unpack_bits(bitmap, nbits, binary_t):
  res = []
  for i in range(nbits):
    res.append(binary_t((bitmap>>i)&1))
  return res
