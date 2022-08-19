# @Author  : Jiangshang_jiu
# @Time    : 2022-08-14 15:50
# ----- coding: UTF-8 -----
import config
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from bert_textcnn_model import build_bert_textcnn_model
from keras_bert import Tokenizer, load_vocabulary

# 加载bert字典，构造分词器。
token_dict = load_vocabulary(config.bert_dict_path)
tokenizer = Tokenizer(token_dict)


# 用以加载数据
def load_data(txt_file_path):
    text_list = []
    label_list = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            label_list.append(line[0].split('|'))
            text_list.append(line[1])
    return text_list, label_list


# 对文本编码
def encoding_text(content_list):
    token_ids = []
    segment_ids = []
    for line in tqdm(content_list):
        token_id, segment_id = tokenizer.encode(first=line, max_len=config.max_len)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
    encoding_res = [np.array(token_ids), np.array(segment_ids)]
    return encoding_res


if __name__ == "__main__":
    # 读取训练集与测试集
    train_content_x, train_label_y = load_data(config.train_dataset_path)
    test_content_x, test_label_y = load_data(config.test_dataset_path)

    # 打乱训练集的数据
    index = [i for i in range(len(train_content_x))]
    random.shuffle(index)  # 打乱索引表
    # 按打乱后的索引，重新组织训练集
    train_content_x = [train_content_x[i] for i in index]
    train_label_y = [train_label_y[i] for i in index]

    # 对训练集与测试集的文本编码
    train_x = encoding_text(train_content_x)
    test_x = encoding_text(test_content_x)

    # 对标签集编码
    mlb = MultiLabelBinarizer()
    mlb.fit(train_label_y)
    pickle.dump(mlb, open('./data/mlb.pkl', 'wb'))

    train_y = np.array(mlb.transform(train_label_y))
    test_y = np.array(mlb.transform(test_label_y))

    model = build_bert_textcnn_model(config.bert_config_path, config.bert_checkpoint_path, len(mlb.classes_))
    model.summary()
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=config.batch_size, epochs=config.epochs)
    model.save("./model/bert_textcnn.h5")

    # 训练过程可视化
    # 绘制训练loss和验证loss的对比图
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()
    # 绘制训练acc和验证acc的对比图
    plt.subplot(2, 1, 2)
    epochs = len(history.history['accuracy'])
    plt.plot(range(epochs), history.history['accuracy'], label='acc')
    plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
    plt.legend()
    # 保存loss与acc对比图
    plt.savefig("./model/bert-textcnn-loss-acc.png")





