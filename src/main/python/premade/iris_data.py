import argparse
import pandas as pd
import tensorflow as tf
import iris_data

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


if __name__ == '__main__':
    # Feature columns describe how to use the input.
    # 创建特征列
    my_feature_columns = []
    y_name = 'Species'
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    # print(train)
    # print(train_y)
    for key in train_x.keys():
        # print(train_x.get_values(key))
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    for index, value in enumerate(my_feature_columns):
        # 0 _NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)
        print(index, value)
    # 使用 hidden_units 参数来定义神经网络内每个隐藏层中的神经元数量。,列表长度表示隐藏层的数量
    # n_classes 参数指定了神经网络可以预测的潜在值的数量。由于鸢尾花问题将鸢尾花品种分为 3 类，因此我们将 n_classes 设置为 3。
    # 指定分类器
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)
    # train_feature 是 Python 字典，其中：

    # 每个键都是特征的名称。
    # 每个值都是包含训练集中每个样本的值的数组。
    # train_y包含训练集中每个样本的标签值的数组
    # batch_size 定义批次大小的整数。
    # steps 参数指示 train 在完成指定的迭代次数后停止训练
    # 训练模型
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--train_steps', default=1000, type=int,
                        help='number of training steps')

    args = parser.parse_args([])
    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)
    # 调用 classifier.evaluate 与调用 classifier.train 类似。
    # 最大的区别在于，classifier.evaluate 必须从测试集（而非训练集）中获取样本。
    # 换言之，为了公正地评估模型的效果，用于评估模型的样本一定不能与用于训练模型的样本相同。
    # eval_input_fn 函数负责提供来自测试集的一批样本

    ##评估模型的效果
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(train_x, train_y,
                                                 args.batch_size))
    print(eval_result)

    ##开始预测数据
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    #predict 方法返回一个 Python 可迭代对象，为每个样本生成一个预测结果字典。此字典包含几个键
    # probabilities 键存储的是一个由三个浮点值组成的列表，每个浮点值表示输入样本是特定鸢尾花品种的概率
    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))
    #print(len(list(predictions)))
    #class_ids 键存储的是一个 1 元素数组，用于标识可能性最大的品种。
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        print(pred_dict)
        class_id = pred_dict['class_ids'][0]

        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))