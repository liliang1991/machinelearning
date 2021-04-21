##tf图像处理
import tensorflow  as tf
import os
import matplotlib.pyplot as plt
def show_pic(pic,name,cmap=None):
    '''显示图像'''
    plt.imshow(pic,cmap=cmap)
    plt.axis('off')   # 打开坐标轴为 on
    # 设置图像标题
    plt.title('%s'%(name))
    plt.show()
if __name__ == '__main__':
    image_path="/home/moon/Desktop/index.jpeg"
    ##转换为字节数据
    image_bytes=tf.io.read_file(image_path)
    image_b64 = tf.io.encode_base64(image_bytes)
    ##编码
    image_b64 = tf.io.decode_base64(image_b64)
    image_decode = tf.io.decode_image(image_b64)
    h,w,c=image_decode.shape
    print("before resize image hight:{},width:{},channels:{}".format(h,w,c))
    #图像放缩
    image_resize=tf.image.resize(
    image_decode,
    (618,618),
    method="bilinear",
)
    h,w,c=image_resize.shape
    print("after resize image hight:{},width:{},channels:{}".format(h,w,c))
    #显示图像
    image_resize=image_resize/255
    plt.imshow(image_resize)
    plt.show()
    #上下翻转
    image_flip=tf.image.flip_up_down(image_decode)
    plt.imshow(image_flip)
    plt.show()
    #对角线翻转
    image_tans=tf.image.transpose(image_decode)
    plt.imshow(image_tans)
    plt.show()
    #图像旋转(只能旋转90、180、270、360度)
    image_rot=tf.image.rot90(image_decode,1)
    plt.imshow(image_rot)
    plt.show()
    #对比度调整
    image_contrast=tf.image.adjust_contrast(image_decode,3)
    plt.imshow(image_contrast)
    plt.show()
    # 中心裁剪，裁剪比例为60%
    cropped_img1 = tf.image.central_crop(image_decode, central_fraction=0.6)
    show_pic(cropped_img1,'cropped_img1')
    #  转换为灰度
    gray_img = tf.image.rgb_to_grayscale(image_decode)
    # 将三通道压缩成单通道
    gray_img = tf.squeeze(gray_img)
    show_pic(gray_img,'img','gray')
    float_img = tf.image.convert_image_dtype(image_decode, tf.float32)   #  图像数据类型转换，并归一化到[0,1]之间
    hsv_img = tf.image.rgb_to_hsv(float_img)   # 对rgb图像进行hsv转换，输入的图像数据类型必须是half,float16,float32,float64)
    show_pic(hsv_img,'hsv_img','hsv')

    # 改变图像饱和度
    saturated_img= tf.image.adjust_saturation(image_decode,8)
    show_pic(saturated_img,'saturated_img')