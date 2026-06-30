import torch
import math

q = torch.randn(1,4,32)
k = torch.randn(1,4,32)
v = torch.randn(1,4,32)


kt = k.transpose(-2,-1)

scores = torch.matmul(q,kt)/math.sqrt(32)

softmax_scores = torch.nn.functional.softmax(scores, dim=-1)
attention_output = torch.matmul(softmax_scores, v)
print(attention_output)