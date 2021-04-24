import tensorflow  as tf

def train_modell(model, inputs, outputs, model_path):
    history = model.fit(
        inputs,
        outputs,
        epochs=1,
        verbose=1
    )
    # 保存参数
    model.save(model_path)
    #模型加载函数
def load_model(model_save):
    model = tf.keras.models.load_model(model_save)
    return model