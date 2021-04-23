####自定义评估函数
import tensorflow as tf
class NN(tf.keras.metrics.Metric):
    #计算正确预测的个数
    def __init__(self,name='MM',**kwargs):
        super(CatgoricalTP, self).__init__(name=name,**kwargs)
        self.MM=self.add.weight(name='MM',initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        values = tf.equal(tf.cast(y_pred, 'int32'), tf.cast(y_true, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weights = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)

        self.MM.assign_add(tf.reduce_sum(values))
    def result(self):
        return self.MM

    def reset_states(self):
        self.MM.assign(0.)
if __name__ == '__main__':

