# @Author  : Jiangshang_jiu
# @Time    : 2022-08-17 11:08
# ----- coding: UTF-8 -----
import config
import pickle
from keras_bert import get_custom_objects
from keras.models import load_model
from keras_bert import load_vocabulary, Tokenizer


# 加载bert字典，构造分词器。
token_dict = load_vocabulary(config.bert_dict_path)
tokenizer = Tokenizer(token_dict)

# 加载训练好的模型
model = load_model('./model/bert_textcnn.h5', custom_objects=get_custom_objects())
mlb = pickle.load(open('./data/mlb.pkl', 'rb'))


# 预测单个句子的标签
def predict_single_text(text):
    token_id, segment_id = tokenizer.encode(first=text, max_len=config.max_len)
    prediction = model.predict([[token_id], [segment_id]])[0]

    indices = [i for i in range(len(prediction)) if prediction[i] > 0.5]
    lables = [mlb.classes_.tolist()[i] for i in indices]
    return "|".join(lables)


if __name__ == "__main__":
    text = "美的置业：贵阳项目挡墙垮塌致8人遇难已责令全面停工"
    result = predict_single_text(text)
    print(result)
