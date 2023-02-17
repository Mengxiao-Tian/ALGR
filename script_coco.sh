DATA_PATH=/home/tmx/disk/data/VSRN_data/
BERT_PATH=./bert_model/uncased_L-12_H-768_A-12/

python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --data_name coco_precomp --logger_name ALGR_coco --max_violation --num_epochs 30 --lr_update 20

