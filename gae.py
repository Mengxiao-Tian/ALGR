import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.utils.weight_norm as weightnorm
import math
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        # init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
        init.orthogonal_(m.weight.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = nn.Parameter(torch.nn.init.uniform_(tensor=torch.empty(shape), a=-init_range, b=init_range))
    return initial

class gru_unit(nn.Module):
    def __init__(self, output_dim, act, dropout_p):
        super(gru_unit,self).__init__()
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.act = act
        self.z0_weight = glorot([self.output_dim, self.output_dim]) # nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        self.z1_weight = glorot([self.output_dim, self.output_dim])
        self.r0_weight = glorot([self.output_dim, self.output_dim])
        self.r1_weight = glorot([self.output_dim, self.output_dim])
        self.h0_weight = glorot([self.output_dim, self.output_dim])
        self.h1_weight = glorot([self.output_dim, self.output_dim])
        self.z0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.z1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.r1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h0_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.h1_bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self,support, x, mask):
        support = self.dropout(support)
        a = torch.matmul(support, x)
        # updata gate
        z0 = torch.matmul(a, self.z0_weight) + self.z0_bias
        z1 = torch.matmul(x, self.z1_weight) + self.z1_bias
        z = torch.sigmoid(z0+z1)
        # return z
        # reset gate
        r0 = torch.matmul(a, self.r0_weight) + self.r0_bias
        r1 = torch.matmul(x, self.r1_weight) + self.r1_bias
        r = torch.sigmoid(r0+r1)
        # update embeddings
        h0 = torch.matmul(a, self.h0_weight) + self.h0_bias
        h1 = torch.matmul(r*x, self.h1_weight) + self.h1_bias
        h = self.act(mask * (h0 + h1))
        return h*z + x*(1-z)

class GraphLayer(nn.Module):
    """Graph layer."""
    def __init__(self,input_dim,
                      output_dim,
                      act=nn.Tanh(),
                      dropout_p = 0.,
                      gru_step = 2):
        super(GraphLayer, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(1-self.dropout_p)
        self.gru_step = gru_step
        self.gru_unit = gru_unit(output_dim = self.output_dim,
                                 act = self.act,
                                 dropout_p = self.dropout_p)
        # self.dropout
        self.encode_weight = glorot([self.input_dim, self.output_dim])
        self.encode_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.gc = GraphConvolution(input_dim, input_dim, 0.0, act=lambda x: x)

    def forward(self, feature, support):
        mask = self.softmax(feature)
        #feature = self.dropout(feature)
        # encode inputs
        #encoded_feature = torch.matmul(feature, self.encode_weight) + self.encode_bias
        #output = mask * self.act(encoded_feature)
        output = self.gc(feature, support)
        # convolve
        for _ in range(self.gru_step):
            output = self.gru_unit(support, output, mask)
        return output

class GVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1 =128, hidden_dim2 =64, dropout =0.0):
        super(GVAE, self).__init__() 
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        self.gc2 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        #GraphConvolution(input_feat_dim, hidden_dim1, bias=False, dropout=dropout, act=F.relu)
        # self.gc2 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        # self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.fc = nn.Linear(hidden_dim1, input_feat_dim)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.GraphLayer1 = GraphLayer( 
            input_dim = input_feat_dim,
            output_dim = hidden_dim1,
            act = torch.nn.Tanh(),
            dropout_p = 0.2,
            gru_step = 1
        )
        self.GraphLayer2 = GraphLayer( 
            input_dim = input_feat_dim,
            output_dim = hidden_dim1,
            act = torch.nn.Tanh(),
            dropout_p = 0.2,
            gru_step = 1
        )

    def encode(self, x, adj):
        # hidden1 = self.gc1(x, adj)
        # print('encode', self.gc1(x, adj)[0])
        # exit(0)
        return self.gc1(x, adj), self.gc2(x, adj)

    def encode_text(self, x, adj):
        return self.GraphLayer1(x, adj), self.GraphLayer2(x, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def conditional_softmax(self, x_true, x_pred):
        Tensor = torch.cuda.FloatTensor
        x_pred = torch.exp(x_pred)
        x_pred_conditional = Tensor(np.zeros(shape=x_pred.shape))

        for i,(x_t,x_p) in enumerate(zip(x_true, x_pred)):
            rule_seq = torch.argmax(x_t, dim=0) 
            divisor = torch.sum(x_p, dim=0)
            x_pred_conditional[i] = x_p / divisor
        return x_pred_conditional
        
    def decode(self, mu, x):
        rec_x = self.fc(F.tanh(mu))
        rec_x_new = self.conditional_softmax(x, rec_x) 
        return rec_x_new
    
    def decode1(self, mu, x):
        rec_x = self.fc(F.tanh(mu))
        rec_x = F.sigmoid(rec_x)
        return rec_x
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        z_t = z.permute(0, 2, 1).contiguous()
        # print(z.shape, z_t.shape) 
        # print(m.shape)
        # exit(0)
        adj = self.act(torch.matmul(z, z_t))
        return adj

'''
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training) 
        support = torch.matmul(input, self.weight) 
        output = torch.matmul(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

'''

class GraphDiffusionConvolution(Module):
    def __init__(self, in_features, out_features, k=2, bias=True):
        super(GraphDiffusionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.k = k
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features*self.k)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def graph_diffusion_convolution_fn(self, inputs, adj, weight, bias=None, k=2):

        batch_size = inputs.shape[0] 
        support = torch.bmm(inputs, weight.expand(batch_size, -1, -1))
        output = support.clone()
        adj_ = adj
        for i in range(k):
            output += torch.bmm(adj_.expand(batch_size, -1, -1), support) 
            adj_ = torch.bmm(adj_, adj)
        if bias is not None:
            output = output + bias
        output = output.squeeze(dim=2)
        return output
    
    def forward(self, inputs, DW):
        return self.graph_diffusion_convolution_fn(inputs, DW, self.weight, self.bias, self.k)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features 
        self.similarity_function = 'cosine'
        self.W_a = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_a.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(negative_slope=-0.2)


    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        h_prime = input.clone()
        adj_ = adj
        # map input to h
        for i in range(1):
            e = self.leakyrelu(self.compute_similarity_matrix(h_prime))
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj_ > 0, e, zero_vec)
            attention = nn.functional.softmax(attention, dim=2)
            h_prime = torch.matmul(attention, h_prime)
            adj_ = torch.bmm(adj_, adj)
        h_prime = h_prime + self.bias
        return nn.functional.elu(h_prime)

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.W_a), X.permute(0, 2, 1))
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
        elif self.similarity_function == 'cosine':
            X = torch.matmul(X, self.W_a)
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            A= (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError
        return A


'''
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

        self.linear.apply(weights_init_kaiming)
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        # support = torch.mm(input, self.weight)
        support = self.linear(input)
        try:
            output = torch.spmm(adj, support)
        except:
            output = support
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
'''

'''
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, act=lambda x: x, dropout=0.0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, training = self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

'''