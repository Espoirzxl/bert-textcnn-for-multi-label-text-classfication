# @Author  : Jiangshang_jiu
# @Time    : 2022-08-14 15:08
# ----- coding: UTF-8 -----

# 预训练bert参数位置
bert_config_path = './chinese_bert_wwm_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = './chinese_bert_wwm_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = './chinese_bert_wwm_L-12_H-768_A-12/vocab.txt'

# 数据集位置
train_dataset_path = "./data/multi-classification-train.txt"
test_dataset_path = "./data/multi-classification-test.txt"

# 训练参数
epochs = 10
batch_size = 4
max_len = 256
learning_rate = 3e-5
