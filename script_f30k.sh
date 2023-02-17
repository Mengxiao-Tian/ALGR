BERT_PATH=./bert_model/uncased_L-12_H-768_A-12/
DATA_PATH=/home/tmx/disk/share_data/VSRN_data/

python train.py --data_path $DATA_PATH --bert_path $BERT_PATH --data_name f30k_precomp --logger_name ALGR_f30k --max_violation --num_epochs 30 --lr_update 10

