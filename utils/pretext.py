import torch
import numpy as np
def rand_bbox(size, pratio, maxpratio):
    '''
    size: size of the image tensor
    '''
    W = size[2]
    H = size[3]
    psize = round(pratio * W)
    maxpsize = round(maxpratio * W)

    r = np.random.randint(psize, maxpsize + 1)
    # uniform
    cx = np.random.randint(W-r)
    cy = np.random.randint(H-r)
    bbx1 = cx
    bby1 = cy
    bbx2 = cx + r
    bby2 = cy + r
    return bbx1, bby1, bbx2, bby2

def lorotE(inputs):
    idx = torch.randint(4, size=(inputs.size(0),))
    idx2 = torch.randint(4, size=(inputs.size(0),))
    r = inputs.size(2) // 2
    r2 = inputs.size(2)
    for i in range(inputs.size(0)):
        if idx[i] == 0:
            w1 = 0
            w2 = r
            h1 = 0
            h2 = r
        elif idx[i] == 1:
            w1 = 0
            w2 = r
            h1 = r
            h2 = r2
        elif idx[i] == 2:
            w1 = r
            w2 = r2
            h1 = 0
            h2 = r
        elif idx[i] == 3:
            w1 = r
            w2 = r2
            h1 = r
            h2 = r2
        inputs[i][:, w1:w2, h1:h2] = torch.rot90(inputs[i][:, w1:w2, h1:h2], idx2[i], [1, 2])
    rotlabel = idx * 4 + idx2
    rotlabel = rotlabel.cuda()
    return inputs, rotlabel

def lorotI(inputs):
    '''
    inputs: image tensor (B, C, W, H)
    '''
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), 0.05, 0.5)
    idx2 = torch.randint(4, size=(inputs.size(0),))
    for i in range(inputs.size(0)):
        inputs[i][:, bbx1:bbx2, bby1:bby2] = torch.rot90(inputs[i][:, bbx1:bbx2, bby1:bby2], idx2[i], [1, 2])
    rotlabel = idx2
    rotlabel = rotlabel.cuda()
    return inputs, rotlabel

def rotation(inputs):
    '''
    inputs: image tensor (B, C, W, H)
    '''
    idx2 = torch.randint(4, size=(inputs.size(0),))
    for i in range(inputs.size(0)):
        inputs[i] = torch.rot90(inputs[i], idx2[i], [1, 2])
    rotlabel = idx2
    rotlabel = rotlabel.cuda()
    return inputs, rotlabel