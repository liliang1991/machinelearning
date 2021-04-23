import tensorflow as tf
if __name__ == '__main__':
    x=tf.random.normal([1,6,6,3])
    layer=tf.keras.layers.MaxPool2D(strides=2)
    print(layer(x).shape)