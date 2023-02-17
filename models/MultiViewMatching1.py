import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewMatching(nn.Module):
    def __init__(self, ):
        super(MultiViewMatching, self).__init__()

    def cos_batch_torch(self, x, y):
        "Returns the cosine distance batchwise"
        # x is the feature: bs * d
        # y is the feature: bt * d
        # return: bs * bt
        # print(x.size())

        bs = x.size(0)
        D = x.size(1)
        assert(x.size(1)==y.size(1))
        x = x.contiguous().view(bs, D)
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.mm(x, torch.transpose(y,0,1))
        cos_dis = 1 - cos_dis
        #return cosine.transpose(1,0)

        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold

        return torch.nn.functional.relu(res)

    def IPOT_torch_batch_uniform(self, C, bs, bt, beta=0.5, iteration=50):
        # C is the distance matrix
        # c: bs by bt
        sigma = torch.ones(bt, 1).cuda()/float(bt)
        T = torch.ones(bs, bt).cuda()
        A = torch.exp(-C/beta).float().cuda()
        for t in range(iteration):
            Q = A * T # bs * bt
            for k in range(1):
                delta = 1 / (bs * torch.mm(Q, sigma))
                a = torch.mm(torch.transpose(Q,0,1), delta)
                sigma = 1 / (float(bt) * a)
            T = delta * Q * sigma.transpose(1,0)

        return T.detach()

    def GW_distance_node(self, X, Y, lamda=0.5, iteration=5, OT_iteration=20):
        '''
        :param X, Y: Source and target featuers , batchsize by embed_dim
        :param p, q: probability vectors
        :param lam: regularization
        :return: GW distance
        '''
        Cs = self.cos_batch_torch(X, Y).float().cuda() 

        bs = Cs.size(0) 
        T = self.IPOT_torch_batch_uniform(Cs, X.size(0), Y.size(0), iteration=iteration)
        temp = torch.mm(torch.transpose(Cs,0,1), T)
        distance = self.batch_trace(temp, bs)
        return distance

    def batch_trace(self, input_matrix, bs):
        a = torch.eye(input_matrix.size(0)).cuda()
        # print(a.shape, input_matrix.shape)
        b = a * input_matrix
        return torch.sum(torch.sum(b,-1),-1)

    def forward(self, imgs, caps):
        # caps -- (num_caps, dim), imgs -- (num_imgs, r, dim)
        num_caps  = caps.size(0)
        num_imgs, r = imgs.size()[:2]
        
        scores = torch.matmul(imgs, caps.t()) 
        scores_ot = self.GW_distance_node(imgs, caps) 
        return scores, scores_ot

        if num_caps == num_imgs:
            scores = torch.matmul(imgs, caps.t()) #(num_imgs, r, num_caps)
            #scores = scores.max(1)[0]  #(num_imgs, num_caps)
        else:   
            scores = []
            score_ids = []
            for i in range(num_caps):
                cur_cap = caps[i].unsqueeze(0).unsqueeze(0)  #(1, 1, dim)
                cur_cap = cur_cap.expand(num_imgs, -1, -1)   #(num_imgs, 1, dim)
                cur_score = torch.matmul(cur_cap, imgs.transpose(-2, -1)).squeeze()    #(num_imgs, r)
                cur_score = cur_score.max(1, keepdim=True)[0]   #(num_imgs, 1)
                scores.append(cur_score)
            scores = torch.cat(scores, dim=1)   #(num_imgs, num_caps)

        return scores