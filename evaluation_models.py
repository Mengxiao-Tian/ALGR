from __future__ import print_function
import os, sys
import pickle
import argparse

import torch
import numpy
#from datacoco import get_test_loader
from data import get_test_loader
import time
import numpy as np
from tqdm import tqdm
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#from modelori import CAMERA
#from modelglo1 import CAMERA
from model import CAMERA
from model import CAMERA as CAMERA1
from collections import OrderedDict
from misc.utils import print_options
from evaluation import AverageMeter, LogCollector, encode_data, calItr
from evaluation import i2t
from evaluation import t2i
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np 
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import os
from matplotlib import patheffects as PathEffects
from sklearn import metrics

#KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans

#TSNEAlgo
from sklearn.manifold import TSNE 
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw, ImageColor


def evalrank_single(model_path, data_path=None, split='test', fold5=False):
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print('Best model: Epoch = {}, Eiters = {}, Rsum = {:.2f}, R1 = {:.2f}'\
                .format(checkpoint['epoch'], checkpoint['Eiters'], checkpoint['best_rsum'], checkpoint['best_r1']))
    if data_path is not None:
        opt.data_path = data_path

    print_options(opt)
    model = CAMERA(opt)

    ckpt_model = checkpoint['model']
    # load model state
    model.load_state_dict(ckpt_model)

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, opt.batch_size, opt.workers, opt)
    print('Computing results...')
    imgs, caps = encode_data(model, data_loader)
    # np.save('imgs_latent.npy', imgs)
    # np.save('caps_latent.npy', caps)
    # print(imgs.shape, caps.shape)
    print('#Images: %d, #Captions: %d' %(imgs.shape[0] / 5, caps.shape[0]))
    # exit(0)
    
    if not fold5:
        imgs = numpy.array([imgs[i] for i in range(0, len(imgs), 5)])
        #imgs1 = numpy.array([imgs1[i] for i in range(0, len(imgs1), 5)])
        sims = calItr(model, imgs, caps, shard_size=opt.batch_size * 5)
        # no cross-validation, full evaluation
        r, rt = i2t(sims, return_ranks=True)
        ri, rti = t2i(sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            imgs_block, caps_block = imgs[i * 5000:(i + 1) * 5000], caps[i * 5000:(i + 1) * 5000]
            imgs_block = numpy.array([imgs_block[i] for i in range(0, len(imgs_block), 5)])
            sims = calItr(model, imgs_block, caps_block, shard_size=opt.batch_size * 5)
            r, rt0 = i2t(sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(sims, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

def evalrank_ensemble(model_path, model_path2, data_path=None, split='test', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print('Best model 1: Epoch = {}, Eiters = {}, Rsum = {:.2f}, R1 = {:.2f}'\
                .format(checkpoint['epoch'], checkpoint['Eiters'], checkpoint['best_rsum'], checkpoint['best_r1']))
    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']
    print('Best model 2: Epoch = {}, Eiters = {}, Rsum = {:.2f}, R1 = {:.2f}'\
                .format(checkpoint2['epoch'], checkpoint2['Eiters'], checkpoint2['best_rsum'], checkpoint2['best_r1']))

    if data_path is not None:
        opt.data_path = data_path

    
    model = CAMERA(opt)
    model2 = CAMERA1(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])
    model2.load_state_dict(checkpoint2['model'])

    #get_test_loader(split, opt.data_name, opt.batch_size, opt.workers, opt)
    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, opt.batch_size, opt.workers, opt)

    print('Computing results...')
    t0 = time.time()

    imgs, caps = encode_data(model, data_loader)
    imgs2, caps2 = encode_data(model2, data_loader)
    if not fold5:
        # no cross-validation, full evaluation
        imgs = numpy.array([imgs[i] for i in range(0, len(imgs), 5)])
        #imgs1 = numpy.array([imgs1[i] for i in range(0, len(imgs1), 5)])
        sims = calItr(model, imgs, caps, shard_size=opt.batch_size * 5) 
        imgs2 = numpy.array([imgs2[i] for i in range(0, len(imgs2), 5)])
        #imgs22 = numpy.array([imgs22[i] for i in range(0, len(imgs22), 5)])
        sims2 = calItr(model2, imgs2, caps2, shard_size=opt2.batch_size * 5)
        sims = (sims + sims2) / 2

        r, rt = i2t(sims, return_ranks=True)
        ri, rti = t2i(sims, return_ranks=True)

        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            imgs_block, caps_block = imgs[i * 5000:(i + 1) * 5000], caps[i * 5000:(i + 1) * 5000]
            imgs_block = numpy.array([imgs_block[i] for i in range(0, len(imgs_block), 5)])
            sims = calItr(model, imgs_block, caps_block, shard_size=opt.batch_size * 5)   
            imgs2_block, caps2_block = imgs2[i * 5000:(i + 1) * 5000], caps2[i * 5000:(i + 1) * 5000]
            imgs2_block = numpy.array([imgs2_block[i] for i in range(0, len(imgs2_block), 5)])
            sims2 = calItr(model2, imgs2_block, caps2_block, shard_size=opt2.batch_size * 5)   
            sims = (sims + sims2) / 2 

            r, rt0, sims = i2t(sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(sims, return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

