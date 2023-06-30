import torch
d=torch.load("/home2/khanhnd/fairseq/checkpoints/checkpoint_best.pt")
print(d["model"].keys())