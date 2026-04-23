#!/usr/bin/env python3
"""Debug: Check what's in the matrix file."""

import struct

with open('matrices.bin', 'rb') as f:
    num_scales = struct.unpack('I', f.read(4))[0]
    d = struct.unpack('I', f.read(4))[0]
    K = struct.unpack('I', f.read(4))[0]
    
    print(f"Header:")
    print(f"  num_scales: {num_scales}")
    print(f"  d: {d}")
    print(f"  K: {K}")
    
    scales = []
    for i in range(num_scales):
        scale = struct.unpack('I', f.read(4))[0]
        scales.append(scale)
    print(f"  scales: {scales}")
    
    # Read first few elements of A matrix
    print(f"\nA matrix (first 5 elements):")
    A_first =[]
    for i in range(5):
        val = struct.unpack('f', f.read(4))[0]
        A_first.append(val)
    print(f"  {A_first}")
    
    print(f"\nFile size: {f.seek(0, 2)} bytes")
