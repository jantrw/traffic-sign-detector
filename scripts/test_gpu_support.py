import torch

print("Device:", torch.cuda.get_device_name(0))

x = torch.randn(3, 3).to("cuda")
print("Tensor on GPU:", x)