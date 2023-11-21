import torch
from utils.evaluation import LPIPS
img1 = torch.rand(1, 1, 32, 32)
img2 = torch.rand(1, 1, 32, 32)
print(LPIPS(img1,img2))