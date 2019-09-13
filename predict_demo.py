# coding:utf8
import tensorflow as tf
from pprint import pprint
import numpy as np
import cv2 as cv
import argparse
import pickle
import imutils
import time
import glob
import os

'''
    模型分类推理
        python predict_demo-wangqiang.py \
          --pb_file /xx/xx.h5 \
          --img_dir /xx/img_dir \
          --label_file /xx/xx.pickle \
          --img_crop_width 96 \
          --img_crop_height 96
'''

def parser():
    parser = argparse.ArgumentParser('JN')
    parser.add_argument("--pb_file", type=str, required=True, help="pb_file name")
    parser.add_argument('--img_dir', type=str, required=True, help='The test data')
    parser.add_argument("--label_file", type=str, required=True, help="Category label file:/xxx/xx.pickle")
    parser.add_argument('--img_crop_width', type=int, required=True, help='Image crop width during testing.')
    parser.add_argument('--img_crop_height', type=int, required=True, help='Image crop height during testing.')
    args = parser.parse_args()
    return args

def inference(args):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        pb_file = args.pb_file
        with open(pb_file, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            input_x = sess.graph.get_tensor_by_name("conv2d_1_input:0")
            #pprint(input_x)
            out_sigmoid = sess.graph.get_tensor_by_name("activation_7/Sigmoid:0")
            #pprint(out_sigmoid)

            img_format = ['jpg', 'JPG', 'png', 'PNG']
            for filename in glob.iglob(args.img_dir + os.sep + "**" + os.sep + "*.*", recursive=True):
                pprint(filename)
                if filename.split('.')[-1] in img_format:
                    # 加载图片
                    #image = cv.imread(filename)
                    image = cv.imdecode(np.fromfile(filename, dtype=np.uint8), 1)  # 读取包含中文路径或者中文字符的图片
                    output = imutils.resize(image, width=400) # 用于最后的显示---搞到400x400大小

                    # 预处理
                    image = cv.resize(image, (args.img_crop_width, args.img_crop_height))
                    image = image.astype("float") / 255.0

                    # 加载类别标签
                    mlb = pickle.loads(open(args.label_file, "rb").read())

                    start_time = time.time()
                    # [-1, height, width, channel]
                    img_out_sigmoid = sess.run(out_sigmoid,
                                               feed_dict={input_x: np.reshape(image,
                                                                              [-1, args.img_crop_height,
                                                                               args.img_crop_width, image.shape[2]])})
                    pprint('====> img_out_sigmoid: {}'.format(img_out_sigmoid))
                    runtime = time.time() - start_time
                    pprint('====> run time：%f' % (runtime * 1000) + 'ms')

                    # np.argsort() ---> https://www.cnblogs.com/yyxf1413/p/6253995.html
                    # idxs = np.argsort(img_out_sigmoid[0]) # 所有预测概率(对应索引值)从低到高排序
                    # idxs = np.argsort(img_out_sigmoid[0])[::-1] # 所有预测概率(对应索引值)从高到低排序
                    idxs = np.argsort(img_out_sigmoid[0])[::-1][:2]  # 所有预测概率从高到低排序，取前两个最大(索引)值
                    pprint('=====> idxs: {}'.format(idxs))

                    # 循环遍历高置信度类标签的索引
                    for (i, j) in enumerate(idxs):
                        print(i, j)
                        # 构建标签并在图像上绘制标签
                        label = "{}: {:.2f}%".format(mlb.classes_[j], img_out_sigmoid[0][j] * 100)
                        cv.putText(output, label, (10, (i * 30) + 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # show the output image
                    cv.imshow("Output", output)
                    cv.waitKey(0)

if __name__ == '__main__':
    args = parser()
    assert args.pb_file, '`pb_file` missing.'
    assert args.img_dir, '`img_dir` missing.'
    assert args.label_file, '`label_file` missing.'
    assert args.img_crop_width, '`img_crop_width` missing.'
    assert args.img_crop_height, '`img_crop_height` missing.'

    inference(args)