# @Author  : Jiangshang_jiu
# @Time    : 2022-08-15 9:35
# ----- coding: UTF-8 -----
import config
import pickle
import numpy as np
from tqdm import tqdm
from keras_bert import get_custom_objects
from keras.models import load_model
from keras_bert import load_vocabulary, Tokenizer
from sklearn.metrics import hamming_loss, classification_report

# 加载bert字典，构造分词器。
token_dict = load_vocabulary(config.bert_dict_path)
tokenizer = Tokenizer(token_dict)

# 加载训练好的模型
model = load_model('./model/bert_textcnn.h5', custom_objects=get_custom_objects())
mlb = pickle.load(open('./data/mlb.pkl', 'rb'))


def load_data(txt_file_path):
    text_list = []
    label_list = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            label_list.append(line[0].split('|'))
            text_list.append(line[1])
    return text_list, label_list


def predict_single_text(text):
    token_id, segment_id = tokenizer.encode(first=text, max_len=config.max_len)
    prediction = model.predict([[token_id], [segment_id]])[0]

    indices = [i for i in range(len(prediction)) if prediction[i] > 0.5]
    lables = [mlb.classes_.tolist()[i] for i in indices]
    one_hot = np.where(prediction > 0.5, 1, 0)
    return one_hot, lables


def evaluate():
    test_x, test_y = load_data(config.test_dataset_path)
    true_y_list = mlb.transform(test_y)

    pred_y_list = []
    pred_labels = []
    for text in tqdm(test_x):
        pred_y, label = predict_single_text(text)
        pred_y_list.append(pred_y)
        pred_labels.append(label)

    # 计算accuracy，一条数据的所有标签全部预测正确则1，否则为0。
    test_len = len(test_y)
    correct_count = 0
    for i in range(test_len):
        if test_y[i] == pred_labels[i]:
            correct_count += 1
    accuracy = correct_count / test_len

    print(classification_report(true_y_list, pred_y_list, target_names=mlb.classes_.tolist(), digits=4))
    print("accuracy:{}".format(accuracy))
    print("hamming_loss:{}".format(hamming_loss(true_y_list, pred_y_list)))


if __name__ == "__main__":
    evaluate()
