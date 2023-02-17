import torch
import evaluation_models


# flickr
evaluation_models.evalrank_single('ALGR_f30k/model_best.pth.tar', data_path='./data/VSRN_data/', split="test", fold5=False)

# coco
evaluation_models.evalrank_single("ALGR_coco/model_best.pth.tar", data_path='./data/VSRN_data/', split="testall", fold5=False)
evaluation_models.evalrank_single("ALGR_coco/model_best.pth.tar", data_path='./data/VSRN_data/', split="testall", fold5=True)
