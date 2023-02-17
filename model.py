import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math
import torch.optim as optim

from models import TextEncoder
from models import PositionEncoder
from models import AGSA
from models import Summarization
from models import MultiViewMatching
from loss import TripletLoss, DiversityRegularization, AngularLoss, ContrastiveLoss_all
from gae import GVAE
from intra_inter_attention import Attention, intra_relation


class Tripletnet(nn.Module):
    def __init__(self, opt):
        super(Tripletnet, self).__init__()
        self.gvae = GVAE(opt.embed_size, opt.embed_size, opt.embed_size // 2, dropout=0.0)
        self.imagepostselfattention = Attention(opt.embed_size, 'dot') 

    def forward(self, x, y, adj_v, adj_t, volatile):
        x_attn = intra_relation(x, x, 8)
        x = torch.bmm(x_attn, x)
        mean_x, logvar_x = self.gvae.encode(x, adj_v)
        mean_y,logvar_y = self.gvae.encode_text(y, adj_t) 

        if volatile == False:
            latent_x = self.gvae.reparameterize(mean_x, logvar_x)
            latent_y = self.gvae.reparameterize(mean_y,logvar_y) 

            latent_x_final = self.imagepostselfattention(latent_x, latent_y)
            latent_y_final = self.imagepostselfattention(latent_y, latent_x)  

            reconstruct_x = self.gvae.decode1(latent_x_final, x)
            reconstruct_y = self.gvae.decode(latent_y_final, y)  

            return mean_x, logvar_x, reconstruct_x,\
            mean_y, logvar_y, reconstruct_y
        else:
            return F.normalize(mean_x, p=2, dim=-1), F.normalize(mean_y, p=2, dim=-1)


def annealing_fn(annealing_strategy, step, k, x, m):
    if annealing_strategy == 'logistic':
        return m * float(1 / (1 + np.exp(-k * (step - x))))
    elif annealing_strategy == 'linear':
        return m * min(1, step / x)
 
class GVAEloss(nn.Module):
    ''' Compute the loss within each batch
    '''
    def __init__(self, alpha = 0.1, n_nodes = 128):
        super(GVAEloss, self).__init__()
        self.alpha = alpha
        self.n_nodes = n_nodes
    def forward(self, mu, logvar, epoch): 
        annealing_strategy = 'logistic'
        step = epoch
        k = 0.001
        x = 10
        m = 1.0

        kl_weight = annealing_fn(annealing_strategy, step, k, x, m)
        kld = -0.5 / self.n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return kl_weight * kld


def l2norm(X, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class EncoderImagePrecompSelfAttn(nn.Module):

    def __init__(self, img_dim, embed_size, head, smry_k, drop=0.0):
        super(EncoderImagePrecompSelfAttn, self).__init__()
        self.embed_size = embed_size

        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
        self.position_enc = PositionEncoder(embed_size) 
        self.agsa = AGSA(1, embed_size, h=head, is_share=False, drop=drop)
        #self.mvs = Summarization(embed_size, smry_k)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, boxes, imgs_wh):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        fc_img_emd = l2norm(fc_img_emd)  #(bs, num_regions, dim)
        posi_emb = self.position_enc(boxes, imgs_wh)    #(bs, num_regions, num_regions, dim)

        # Adaptive Gating Self-Attention
        self_att_emb = self.agsa(fc_img_emd, posi_emb)    #(bs, num_regions, dim)
        self_att_emb = l2norm(self_att_emb)
        return self_att_emb 

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompSelfAttn, self).load_state_dict(new_state)

class CAMERA(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecompSelfAttn(opt.img_dim, opt.embed_size, \
                                    opt.head, opt.smry_k, drop=opt.drop)
        self.txt_enc = TextEncoder(opt.bert_config_file, opt.init_checkpoint, \
                                    opt.embed_size, opt.head, drop=opt.drop)
        self.tripnet = Tripletnet(opt) 
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.tripnet.cuda()  
            cudnn.benchmark = True

        self.mvm = MultiViewMatching()
        # Loss and Optimizer
        self.crit_ranking = TripletLoss(margin=opt.margin, max_violation=opt.max_violation)
        self.criterion_g = GVAEloss(alpha =0.1, n_nodes=opt.batch_size) 

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters()) 
        params = [p for p in params if p.requires_grad]
        if opt.measure:
            params += list(self.tripnet.parameters())  

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        if self.opt.measure: 
            state_dict += [self.tripnet.state_dict()] 
        return state_dict

    def load_state_dict(self, state_dict): 
        model_dict_img = self.img_enc.state_dict()
        pretrained_dict_img = {k: v for k, v in state_dict[0].items() if k in model_dict_img}
        model_dict_img.update(pretrained_dict_img)
        self.img_enc.load_state_dict(model_dict_img)

        model_dict_txt = self.txt_enc.state_dict()
        pretrained_dict_txt = {k: v for k, v in state_dict[1].items() if k in model_dict_txt}
        model_dict_txt.update(pretrained_dict_txt)
        self.txt_enc.load_state_dict(model_dict_txt)

        if len(state_dict)>2: 
            new_state_dict = OrderedDict()
            for k, v in state_dict[2].items():
                new_state_dict[k] = v
            self.tripnet.load_state_dict(new_state_dict, strict=False)
            new_state_dict = OrderedDict() 


    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()  
        if self.opt.measure: 
            self.tripnet.train()  

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()  
        if self.opt.measure: 
            self.tripnet.eval()   

    
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1)) # D
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


    def preprocess_adj(self, adj):
        # abj: [ns1xns2, ns2xns2....]
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        max_length = max([a.shape[0] for a in adj]) # a: node_size_a x node_size_a
        # mask: example_num x max_length x 1
        mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding #

        for i in range(adj.shape[0]):
            adj_normalized = self.normalize_adj(adj[i]) # no self-loop
            pad = max_length - adj_normalized.shape[0] # padding for each epoch
            adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
            mask[i,:adj[i].shape[0],:] = 1.
            adj[i] = adj_normalized

        return adj

    def get_adj1(self, dep, lengths, bs):  
        lens = 32#max(lengths) 

        adj = torch.zeros((bs, lens, lens), dtype=torch.int64)#.cuda()
        for index in range(bs):
            for i, pair in enumerate(dep[index]): 
                if i == 0 or pair[0] >= lens or pair[1] >= lens:
                    continue
                adj[:,pair[0], pair[1]] = 1
                adj[:,pair[1], pair[0]] = 1
            
        adj = adj.clone() + torch.eye(lens).expand(bs, lens, lens)#.cuda()
        #return torch.from_numpy(adj).cuda().float() 
        adj = adj.float().numpy()
        adj_norm= self.preprocess_adj(adj)

        adj_norm = torch.tensor(adj_norm).cuda().float().clone()  

        # # generate adj label
        adj_label = adj
        adj_label = torch.FloatTensor(adj_label)

        return adj_norm, adj_label

    def get_adj2(self, bs, lens):
        batch_adj = torch.ones(bs, lens, lens)
        # normalization
        w_norm = batch_adj.pow(2).sum(1, keepdim=True).pow(1. / 2)
        adj_norm = batch_adj.div(w_norm)

        # # generate adj label
        adj_label = batch_adj
        adj_label = torch.FloatTensor(adj_label)

        return adj_norm, adj_label


    def forward_emb(self, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """
        images, boxes, imgs_wh, input_ids, lengths, ids, attention_mask, token_type_ids = batch_data
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            boxes = boxes.cuda()
            imgs_wh = imgs_wh.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        # Forward
        cap_emb = self.txt_enc(input_ids, attention_mask, token_type_ids, lengths)
        img_emb = self.img_enc(images, boxes, imgs_wh)

        bs = cap_emb.size(0) 
        cap_adj_norm, cap_adj_label = self.get_adj2(bs, cap_emb.size(1))
        img_adj_norm, img_adj_label = self.get_adj2(bs, img_emb.size(1))


        if volatile==False:
            mean_x, logvar_x, reconstruct_x,\
            mean_y, logvar_y, reconstruct_y = self.tripnet(img_emb, cap_emb, img_adj_norm.cuda(), cap_adj_norm.cuda(), volatile)
            return mean_x, logvar_x, reconstruct_x,\
            mean_y, logvar_y, reconstruct_y, \
            img_emb, cap_emb, \
            img_adj_label, cap_adj_label
        else:
            latent_x_img, latent_x_cap = self.tripnet(img_emb, cap_emb, img_adj_norm.cuda(), cap_adj_norm.cuda(), volatile) 
            latent_x_cap = self.readout(latent_x_cap, cap_adj_label) 
            latent_x_img = self.readout(latent_x_img, img_adj_label) 
            return latent_x_img, latent_x_cap

    def guiyihua(self, cap, img):
        img = torch.mean(img, 1)
        img = F.normalize(img, dim=-1) 
        cap = torch.mean(cap, 1)
        cap = F.normalize(cap, p=2, dim=-1)
        return cap, img

    def readout(self, f1, adj1):
        
        f1_list = []
        for i in range(f1.shape[0]):
            sample_size1 = adj1[i].size(0)
            f_tmp = torch.mean(f1[i, :sample_size1], 0)
            f_tmp = f_tmp.view(f_tmp.shape[0], -1)
            f1_list.append(f_tmp.unsqueeze(0))

        f1 = torch.cat(f1_list, 0)
        f1 = f1.view(f1.shape[0], -1)
        
        f1 = F.normalize(f1, dim=-1)
        return f1

    def pearson_sim(self, x1, x2):
        # x1: batch_size, n_dim
        # x2: batch_size, n_dim
        EPS = 1e-8
        x1_norm = (x1 - x1.mean(axis=-1, keepdims=True)) / (x1.std(axis=-1, keepdims=True) + EPS)
        x2_norm = (x2 - x2.mean(axis=-1, keepdims=True)) / (x2.std(axis=-1, keepdims=True) + EPS)

        numerator = (x1_norm * x2_norm).sum()
        denominator = torch.sqrt((x1_norm ** 2).sum() * (x2_norm ** 2).sum())
        return numerator / (denominator + EPS)

    def kl_final(self, mu_video, mu_text, logvar_video, logvar_text, epoch):
        annealing_strategy = 'logistic'
        step = epoch
        k = 0.01
        x = 1000
        m = 1.0

        kl_weight = annealing_fn(annealing_strategy, step, k, x, m)
        distance = torch.sqrt(torch.sum((mu_video - mu_text) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_video.exp()) - torch.sqrt(logvar_text.exp())) ** 2, dim=1))
        distance = kl_weight * distance.sum()
        return distance

    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        mean_x, logvar_x, reconstruct_x,\
            mean_y, logvar_y, reconstruct_y, \
            img_emb, cap_emb, \
            img_adj_label, cap_adj_label = self.forward_emb(batch_data)
        bs = mean_x.size(0)

        
        if epoch < 13:
            cap, img = self.guiyihua(ori_x_cap, ori_x_img)
            sim_mat_1 = self.mvm(img, cap) 
            loss = self.crit_ranking(sim_mat_1)
            self.logger.update('Loss', loss.item(), bs) 
        
        else:
            loss_recon_cap = F.smooth_l1_loss(reconstruct_y, cap_emb,reduce=None)
            loss_recon_img = F.smooth_l1_loss(reconstruct_x, img_emb,reduce=None) 
            loss_recon = loss_recon_cap + loss_recon_img 

            # structure loss
            loss_gvae_cap = self.criterion_g(mean_y, logvar_y, epoch) 
            loss_gvae_img = self.criterion_g(mean_x, logvar_x, epoch)  
            loss_gvae =  loss_gvae_cap + loss_gvae_img
            loss_gvae_final = loss_recon + loss_gvae 

            latent_x_cap_norm = self.readout(mean_y, cap_adj_label) 
            latent_x_img_norm = self.readout(mean_x, img_adj_label)   
             
            # bidirectional triplet ranking loss
            sim_mat = self.mvm(latent_x_img_norm, latent_x_cap_norm)
            ranking_loss = self.crit_ranking(sim_mat) 

            #Distribution Alignment
            distribution_loss = self.kl_final(torch.mean(mean_x, 1), torch.mean(mean_y, 1), torch.mean(logvar_x, 1), torch.mean(logvar_y, 1), epoch)
            
            loss = ranking_loss + loss_gvae_final + distribution_loss

        self.logger.update('Loss', loss.item(), bs)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            if isinstance(self.params[0], dict):
                params = []
                for p in self.params:
                    params.extend(p['params'])
                clip_grad_norm(params, self.grad_clip)
            else:
                clip_grad_norm(self.params, self.grad_clip)

        self.optimizer.step()


