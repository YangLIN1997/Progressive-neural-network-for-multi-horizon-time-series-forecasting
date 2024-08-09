import torch

class PMask():
    def __init__(self, B, label_len,pred_len, indices, steps,device="cpu"):
        self._mask = torch.ones((B,1,1,label_len+pred_len), dtype=torch.bool,device=device)
        self._mask[:, 0, 0, :label_len] = 0
        self._mask = self._mask.repeat(1,1,label_len+pred_len,1)
        k=indices.shape[-1]
        src = torch.ones((B,label_len+pred_len), dtype=torch.bool,device=device)
        indices-=1
        src[:,:label_len]=0
        for i in range(steps):
            a = indices[:,-k:]+1
            a[a>=label_len+pred_len]=label_len+pred_len-1
            indices=torch.cat((indices,a),dim=-1)
            src = src.scatter_(-1,a,0)
            for j in range(k):
                self._mask[torch.arange(0, B), 0,a[:, j]]=src

        self._mask = self._mask.to(device)
    @property
    def mask(self):
        return self._mask

class PMask_test():
    def __init__(self, B, label_len,pred_len, indices,steps,device="cpu"):

        self.k=indices.shape[-1]
        self.a = indices-1
        self.steps=steps
        self.src = torch.ones((B,label_len+pred_len), dtype=torch.bool,device=device)
        self.indices=indices-1
        self.src[:,:label_len]=0

        self._mask = torch.ones((B,1,1,label_len+pred_len), dtype=torch.bool,device=device)
        self._mask[:, 0, 0, :label_len] = 0
        self._mask = self._mask.repeat(1,1,label_len+pred_len,1)
        self.label_len=label_len
        self.pred_len=pred_len
        self._mask = self._mask.to(device)

    @property
    def mask(self):
        return self._mask

    def forward(self):
        self.a = self.indices[:,-self.k:]+1
        self.a[self.a>=self.label_len+self.pred_len]=self.label_len+self.pred_len-1
        self.indices=torch.cat((self.indices,self.a),dim=-1)
        self.src = self.src.scatter_(-1,self.a,0)
        # print(self.a[0],self._mask[0,0,10:,10:])
        for j in range(self.k):
            self._mask[torch.arange(0, self.a.shape[0]), 0, self.a[:, j]] = self.src
        # print(self.a[0],self._mask[0,0])
        # print(9999999999999999999999)
        return


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            # print((torch.triu(torch.ones(mask_shape), diagonal=1))[0])
            # print(self._mask[0])
            # print(self._mask.shape)
            # print(interd.shape)
            # self._mask = torch.logical_and(self._mask,interd)
            # print(self._mask[0])

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            # self._mask = (1-torch.triu(torch.ones(mask_shape), diagonal=1)).bool().to(device)
            # print((torch.triu(torch.ones(mask_shape), diagonal=1))[0])
    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dytpe=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask