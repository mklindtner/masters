import debugpy

print("Starting debugpy server...")
debugpy.listen(("localhost", 5678))  # 0 lets it choose an available port
print("Waiting for debugger to attach...")
debugpy.wait_for_client()

import torch

print("before")
print(torch.cuda.is_available())
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

for i in range(2,10):
    print(i-(i-1))

print("after")