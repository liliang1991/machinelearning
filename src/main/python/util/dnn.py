from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# sys.path.append("D:\\IdeaProjects\\ultra-nlp\\ultra-nlp-tensorflow\\src")

import argparse
import time

import os
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

from ultra.classification.low_level.dnn_test_eval import evaluation
from ultra.classification.low_level.input_data import SparkData

from ultra.classification.low_level import dnn_inference
from ultra.classification.low_level.dnn_train import Train
from ultra.common.enums import WorkType

FLAGS = None


def train(learning_rate=0.01, num_epochs=3, shape=[10,10,10]
          , batch_size=100, data_dir="/tmp/data", model_dir ="/tmp/model",reg_factor=0.001):
   with tf.Graph().as_default():
        # get traindata
        train_data = SparkData(data_dir, WorkType.train.value, num_epochs
                               , batch_size, shape[0])
        label_batch, feature_batch = train_data.get_data()
        im = dnn_inference.Inference(shape, feature_batch, WorkType.train)
        logits = im.inference()

        feature_holder = tf.placeholder(tf.float32, [None, shape[0]], "dnn_input")
        logits_predict = dnn_inference.Inference(shape, feature_holder, WorkType.test).inference()
        predict_tensor = tf.arg_max(logits_predict, 1,name="dnn_output")

        input = {signature_constants.PREDICT_INPUTS: tf.saved_model.utils.build_tensor_info(feature_holder)}
        output = {signature_constants.PREDICT_OUTPUTS: tf.saved_model.utils.build_tensor_info(predict_tensor)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=input,outputs=output,method_name=signature_constants.PREDICT_METHOD_NAME)
        tf.summary.histogram("logits", logits)
        train_object = Train(logits, label_batch, learning_rate,reg_factor)

        # not have regularization yet
        train_op, loss, reg_loss, global_step = train_object.fit()
        # The op for initializing the variables.

        validate_data = SparkData(data_dir,WorkType.validate.value, num_epochs*10, FLAGS.validate_batch_size, shape[0])
        label_batch_validate, feature_batch_validate = validate_data.get_data()
        im_validate = dnn_inference.Inference(shape, feature_batch_validate, WorkType.validate)
        logits_validate = im_validate.inference()

        valid_accuracy = evaluation(logits_validate, tf.to_int32(label_batch_validate))/FLAGS.validate_batch_size
        tf.summary.scalar("validation_accuracy",valid_accuracy)
        train_accuracy = evaluation(logits, tf.to_int32(label_batch))/batch_size
        tf.summary.scalar("train_accuracy",train_accuracy)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        #增加保存模型的builer
        model_builder= tf.saved_model.builder.SavedModelBuilder(export_dir=model_dir)


        with tf.Session() as sess:
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            sess.run(init_op)
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start_time = time.time()
            ckpt_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            try:
                while not coord.should_stop():
                    _, loss_value, reg_loss_value, step, feature = sess.run([train_op, loss, reg_loss, global_step,feature_batch])
                    # Print an overview fairly often.
                    if step % 100 == 0:
                        duration = time.time() - start_time
                        start_time = time.time()
                        print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
                        print('Setp %d: reg_loss = %.4f' % (step, reg_loss_value))
                        print('Step %d: validate accuracy = %.2f ' % (step, sess.run(valid_accuracy)))
                        print('Step %d: train accuracy = %.2f ' % (step, sess.run(train_accuracy)))
                        # predict = sess.run(predict_tensor,feed_dict={feature_holder:feature})
                        # print(predict)
                        # Update the events file.
                        summary_str = sess.run(summary)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # saver.save(sess, model_file, global_step=step)

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                saver.save(sess, ckpt_file)
                model_builder.add_meta_graph_and_variables(sess,
                        tags=[tf.saved_model.tag_constants.SERVING]
                        , signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
                model_builder.save()
            # Wait for threads to finish.
            coord.join(threads)


def main(_):
    print("shape is %s" % eval(FLAGS.shape))
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    #删除上次保存的模型文件
    if tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    train(FLAGS.learning_rate, FLAGS.num_epochs, eval(FLAGS.shape)
          , FLAGS.batch_size, data_dir= FLAGS.data_dir
          , model_dir=FLAGS.model_dir, reg_factor=FLAGS.reg_factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--shape',
        type=str,
        default='[11963,100,10]',
        help='包括输入、隐藏层、输出在内的神经网络完整形状，使用array格式，例如“[200,100,10]”'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=200,
        help='Batch size.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='E:\\tfrecord',
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/ultra/dnn',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/ultra/model',
        help='保存dnn模型文件的路径，可以支持hdfs'
    )
    parser.add_argument(
        '--reg_factor',
        type=float,
        default='0.001',
        help='损失函数正则损失项的系数'
    )
    parser.add_argument(
        '--validate_batch_size',
        type=int,
        default=200,
        help=('验证集的数据多少')
    )
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main, argv=None)