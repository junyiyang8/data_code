import torch

import torch.nn.functional as F
def gram_schmidt(vectors):
    orthogonalized = []
    for vector in vectors:
        for ortho_vector in orthogonalized:
            projection = torch.dot(vector, ortho_vector) / torch.dot(ortho_vector, ortho_vector)
            vector = vector - projection * ortho_vector
        vector = vector / torch.norm(vector)
        orthogonalized.append(vector)
    return torch.stack(orthogonalized)

def compute_bpc(logits, targets):
    loss = F.cross_entropy(logits, targets, reduction='mean')
    bpc = loss / torch.log(torch.tensor(2.0))
    return bpc

