import torch
import math

'''
in the dimensions of the tensors,
1,4,32
1=> batch size (eg: one sentence prcessesd at a time)
4=> number of tokens(eg: 4 words in the sentence)
32=> embedding size (eg: each word is represented by a 32 dimensional vector ; math representation of the word)
'''
q = torch.randn(1,4,32)
k = torch.randn(1,4,32)
v = torch.randn(1,4,32)

'''
transpose happens in the last two dimensions of the tensor.
that is batch size even after trans will be same but the number of tokens and embedding size will be swapped.
thus the key and query can be multiplied to finally get the attention scores.
'''
kt = k.transpose(-2,-1)

scores = torch.matmul(q,kt)/math.sqrt(32)

softmax_scores = torch.nn.functional.softmax(scores, dim=-1)


'''
formula used:

attention =  softmax((QK^T)/sqrt(d_k))V

'''
attention_output = torch.matmul(softmax_scores, v)
print(attention_output)