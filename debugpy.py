import torch
import debugpy
import socket
import os

print(f"Process ID: {os.getpid()}")


# Find a free port dynamically
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # Bind to an available port
free_port = sock.getsockname()[1]  # Get the assigned port
sock.close()  # Release the port so debugpy can use it

print(f"Using port {free_port} for debugging...")

# # Start debugpy with the dynamically chosen port
# debugpy.listen(("0.0.0.0", free_port))  # Allow external connections
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # Pause execution until VS Code attaches
# print("Debugger attached!")

# # Your existing code
# print("before")
# import torch
# print(torch.cuda.is_available())
# print("after")