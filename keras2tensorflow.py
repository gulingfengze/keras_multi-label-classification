# coding:utf8
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from pprint import pprint
import argparse

'''
    keras .h5 模型转换到tensorflow .pb
    python keras2tensorflow-jn.py \
      --h5_file /xx/xx.h5 \
      --save_dir /xx/xx \
      --pb_file frozen_inference_graph.pb
'''

def parser():
    parser = argparse.ArgumentParser('JN')
    parser.add_argument("--h5_file", type=str, required=True, help="h5_file:/xxx/xx.h5")
    parser.add_argument("--save_dir", type=str, required=True, help="The path to store pb files")
    parser.add_argument("--pb_file", type=str, default='frozen_inference_graph.pb', help="pb_file name")
    args = parser.parse_args()
    return args

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    # 将会话状态冻结为已删除的计算图,创建一个新的计算图,其中变量节点由在会话中获取其当前值的常量替换.
    # session要冻结的TensorFlow会话,keep_var_names不应冻结的变量名列表,或者无冻结图中的所有变量
    # output_names相关图输出的名称,clear_devices从图中删除设备以获得更好的可移植性
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        # 从图中删除设备以获得更好的可移植性
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        # 用相同值的常量替换图中的所有变量
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def run(args):
    # 加载模型：权重+网络结构
    h5_file = args.h5_file
    K.set_learning_phase(0)
    net_model = load_model(h5_file)
    # 打印输入输出节点名称---> 后续推理需要
    pprint('Input_node_Names: {}'.format(net_model.input.name))   # conv2d_1_input:0
    pprint('Output_node_Names: {}'.format(net_model.output.name)) # activation_7/Sigmoid:0
    # 获得当前图
    sess = K.get_session()
    # 冻结图
    frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])
    graph_io.write_graph(frozen_graph, args.save_dir, args.pb_file, as_text=False)
    pprint('=========> The model transformation finished!')
    pprint(K.get_uid())

if __name__ == '__main__':
    args = parser()
    assert args.h5_file, '`h5_file` missing.'
    assert args.save_dir, '`save_dir` missing.'
    assert args.pb_file, '`pb_file` missing.'

    run(args)