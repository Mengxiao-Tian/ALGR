from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch as th

class DiversityLoss(object):
    def __init__(self, attention_hop):
        super(DiversityLoss, self).__init__()
        self.attention_hop = attention_hop
        self.coeff=0.1#args.penalty_coeff

        self.I = Variable(torch.zeros(128, self.attention_hop, self.attention_hop)).cuda()
        for p in range(128):
            for q in range(self.attention_hop):
                self.I.data[p][q][q] = 1

    def cal_loss(self, attention_map):
        attention_map_T = torch.transpose(attention_map, 1, 2).contiguous()
        diversity_loss = self.Frobenius(torch.bmm(attention_map, attention_map_T) - self.I[:attention_map.size(0)])

        return self.coeff * diversity_loss

    def Frobenius(self, mat):
        size = mat.size()
        if len(size) == 3:  # batched matrix
            ret = (torch.sum(torch.sum((mat ** 2), 1), 1) + 1e-10) ** 0.5
            return torch.sum(ret) / size[0]
        else:
            raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score



class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, scores):
        # scores = torch.matmul(imgs, caps.t())
        diag = scores.diag()

        # Reducing the score on diagonal so there are not selected as hard negative
        scores = (scores - 2 * torch.diag(scores.diag()))

        sorted_cap, _ = torch.sort(scores, 0, descending=True)
        sorted_img, _ = torch.sort(scores, 1, descending=True)

        # Selecting the nmax hardest negative examples
        max_c = sorted_cap[:self.nmax, :]
        max_i = sorted_img[:, :self.nmax]

        # Margin based loss with hard negative instead of random negative
        neg_cap = torch.sum(torch.clamp(max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0))
        neg_img = torch.sum(torch.clamp(max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0))
        loss = neg_cap + neg_img

        return loss


class TripletLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class TripletLoss_adj(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(
            self, margin=0.2,
            max_violation=True,
            weight=1., beta=0.999, smooth=20,
        ):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.weight = weight
        self.max_violation = max_violation
        self.beta = beta

        self.loss_softmax = SoftmaxLoss(smooth=smooth)

        self.iteration = 0
        self.k = 0

    def adjust_k(self, ):
        """
            Update loss hyper-parameter k
            linearly from intial_k to 1 according to
            the number of epochs
        """
        self.iteration += 1

        '''
        if self.max_violation:
            self.k = 1
            return 1.
        '''
        #self.k = (1.-self.beta**np.float(self.iteration))
        self.k = 1
        return self.k

    def forward(self, scores ):

        lst = self.loss_softmax(scores)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask#.cuda()
        I = I.to(cost_s.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_t = cost_s.sum()
        cost_im_t = cost_im.sum()

        k = self.adjust_k()

        cost_all_k = (cost_s_t + cost_im_t) * (1. - k)

        # keep the maximum violating negative for each query
        cost_s_max = cost_s.max(1)[0]
        cost_im_max = cost_im.max(0)[0]

        cost_hard_k = (cost_s_max.sum() + cost_im_max.sum()) * k

        total_loss = cost_all_k + cost_hard_k

        return total_loss * self.weight + lst

    def __repr__(self):
        return((
            f'ContrastiveLoss (margin={self.margin}, '
            f'device={self.device}, '
            f'similarity_fn={self.sim}, '
            f'weight={self.weight}, '
            f'max_violation={self.max_violation}, '
            f'beta={self.beta})'
        ))



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, scores_OT):
        # x is the word embedding of sentence
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores, scores_OT = xattn_score_t2i_OT(im, s, s_l, x, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores, scores_OT = xattn_score_i2t_OT(im, s, s_l, x, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)

        scores = scores + 0.1*scores_OT
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_OT = scores_OT.diag().view(im.size(0), 1)
        d1_OT = diagonal_OT.expand_as(scores_OT)
        d2_OT = diagonal_OT.t().expand_as(scores_OT)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_s_OT = (self.margin + scores_OT - d1_OT).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        cost_im_OT = (self.margin + scores_OT - d2_OT).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_OT = cost_s_OT.masked_fill_(I, 0)
        cost_im_OT = cost_im_OT.masked_fill_(I, 0)

        alpha = self.opt.alpha  # .cuda()
        # cost_s = cost_s + alpha * cost_s_OT
        # cost_im = cost_im + alpha * cost_im_OT

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            cost_s_OT = cost_s_OT.max(1)[0]
            cost_im_OT = cost_im_OT.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class DiversityRegularization(nn.Module):
    """
    Compute diversity regularization
    """
    def __init__(self, smry_k, batch_size):
        super(DiversityRegularization, self).__init__()
        self.smry_k = smry_k
        self.batch_size = batch_size
        self.I = torch.eye(smry_k).unsqueeze(0).repeat(batch_size, 1, 1).cuda() #(bs, k, k)

    def forward(self, smry_mat):
        bs = smry_mat.size(0)
        smry_mat = F.normalize(smry_mat, dim=1)   #(bs, num_r, k)
        diversity_loss = torch.matmul(smry_mat.transpose(1, 2), smry_mat)   #(bs, k, k)
        if bs != self.batch_size:
            I = torch.eye(self.smry_k).unsqueeze(0).repeat(bs, 1, 1).cuda()
        else:
            I = self.I
        diversity_loss = diversity_loss - I
        diversity_loss = (diversity_loss ** 2).sum()
        return diversity_loss