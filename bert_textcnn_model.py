# @Author  : Jiangshang_jiu
# @Time    : 2022-08-10 22:15
# ----- coding: UTF-8 -----
import config
import keras
from keras_bert import load_trained_model_from_checkpoint


def textcnn(inputs):
    # 选用3、4、5三个卷积核进行特征提取，最后拼接后输出用于分类。
    kernel_size = [3, 4, 5]
    cnn_features = []
    for size in kernel_size:
        cnn = keras.layers.Conv1D(
            filters=256,
            kernel_size=size,
            activation='relu',
        )(inputs)  # shape=[batch_size,maxlen-2,256]
        cnn = keras.layers.GlobalMaxPooling1D()(cnn)  # shape=[batch_size,256]
        cnn_features.append(cnn)

    # 对kernel_size=3、4、5时提取的特征进行拼接
    output = keras.layers.concatenate(cnn_features, axis=-1)  # [batch_size,256*3]
    # 返回textcnn提取的特征结果
    return output


def build_bert_textcnn_model(config_path, checkpoint_path, class_nums):
    """
    :param config_path: bert_config.json所在位置。
    :param checkpoint_path: bert_model.ckpt所在位置。
    :param class_nums: 最终模型的输出的维度（分类的类别）。
    :return:返回搭建好的模型。
    """
    # 加载预训练好的bert
    bert = load_trained_model_from_checkpoint(
        config_file=config_path,
        checkpoint_file=checkpoint_path,
        seq_len=None
    )

    # 取出[cls]，可以直接用于分类，也可以与其它网络的输出拼接。
    cls_features = keras.layers.Lambda(
        lambda x: x[:, 0],
        name='cls'
    )(bert.output)  # shape=[batch_size,768]

    # 去除第一个[cls]和最后一个[sep]，得到输入句子的embedding，用作textcnn的输入。
    word_embedding = keras.layers.Lambda(
        lambda x: x[:, 1:-1],
        name='word_embedding'
    )(bert.output)  # shape=[batch_size,maxlen-2,768]

    # 将句子的embedding，输入textcnn，得到经由textcnn提取的特征。
    cnn_features = textcnn(word_embedding)  # shape=[batch_size,cnn_output_dim]

    # 将cls特征与textcnn特征进行拼接。
    all_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)  # shape=[batch_size,cnn_output_dim+768]

    # 应用dropout缓解过拟合的现象，rate一般在0.2-0.5。
    all_features = keras.layers.Dropout(0.2)(all_features)  # shape=[batch_size,cnn_output_dim+768]

    # 降维
    dense = keras.layers.Dense(units=256, activation='relu')(all_features)  # shape=[batch_size,256]

    # 输出结果
    output = keras.layers.Dense(
        units=class_nums,
        activation='sigmoid'
    )(dense)  # shape=[batch_size,class_nums]

    # 根据输入和输出构建构建模型
    model = keras.models.Model(bert.input, output, name='bert-textcnn')

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(config.learning_rate),
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # 可以单独运行用于查看模型结构输出模型结构图
    config_path = config.bert_config_path
    checkpoint_path = config.bert_checkpoint_path
    model = build_bert_textcnn_model(config_path, checkpoint_path, 65)
    model.summary()
    keras.utils.plot_model(model, to_file='./model/model.png', show_shapes=True)
