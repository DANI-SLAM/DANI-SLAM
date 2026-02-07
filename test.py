import torch 

model = torch.jit.load("/sly_slam/superpoint_new.pt")

print(model)