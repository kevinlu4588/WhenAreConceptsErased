import torch

# Check if PyTorch is installed and environment is working
print("Conda environment is working!")

# Check for GPU availability
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")