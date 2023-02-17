This is implementation for the paper "Adaptive Latent Graph Representation Learning for Image-Text Matching" (ALGR, TIP 2022). 
It is built on top of [CAMERA](https://github.com/LgQu/CAMERA) and [SCAN](https://github.com/kuanghuei/SCAN).


## Requirements 
* Python 3.6
* Pytorch 1.8.1
* NumPy 1.19.1
* torchvision 0.9.1

## Download data
We use [CAMERA's]() data. The image features can be download [here](https://drive.google.com/drive/u/1/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC). The positions of detected boxes can be download [here](https://drive.google.com/file/d/1K9LnWJc71dK6lF1BJMPlbkIu_vYmHjVP/view?usp=sharing)


## BERT model

We use the BERT code from [BERT-pytorch](https://github.com/huggingface/pytorch-transformers). Please following [here](https://github.com/huggingface/pytorch-transformers/blob/4fc9f9ef54e2ab250042c55b55a2e3c097858cb7/docs/source/converting_tensorflow_models.rst) to convert the Google BERT model to a PyTorch save file `$BERT_PATH`.

## Training new models
For MSCOCO:

Run `script_coco.sh`

For Flickr30K:

Run `script_f30k.sh`


## Evaluate trained models

python evaluate_models.py


## Reference

```
@article{tian2022adaptive,
          title={Adaptive Latent Graph Representation Learning for Image-Text Matching},
          author={Tian, Mengxiao and Wu, Xinxiao and Jia, Yunde},
          journal={IEEE Transactions on Image Processing},
          year={2022},
          publisher={IEEE}
        }
```

