#!/usr/bin/env python3
import torch
import os

print("PyTorch Installation Path:")
print(torch.__file__)

print("\nPyTorch Root Directory:")
torch_root = os.path.dirname(torch.__file__)
print(torch_root)

print("\nPossible Include Paths:")
include_path = os.path.join(torch_root, "include")
print(include_path)

print("\nDirectory Contents of PyTorch root:")
for item in os.listdir(torch_root):
    print(item)