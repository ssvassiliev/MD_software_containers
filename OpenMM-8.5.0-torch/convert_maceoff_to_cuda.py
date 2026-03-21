import torch

model = torch.load(
    "MACE-OFF24_medium.model",
    map_location="cpu",
    weights_only=False  
)

# Move to CUDA and save
model = model.to("cuda:0")
torch.save(model, "MACE-OFF24_medium_cuda.pt")
