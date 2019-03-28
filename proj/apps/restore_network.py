import tensorflow as tf
import numpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

saver_name = '/localtmp/yuting/cnn_approx/saver2.meta'
saver_path = '/localtmp/yuting/cnn_approx'

data = numpy.load('test.npz')

sess = tf.Session()
saver = tf.train.import_meta_graph(saver_name)
saver.restore(sess, tf.train.latest_checkpoint(saver_path))
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")
output = graph.get_tensor_by_name("output:0")

ans_list = []
for file in data.files:
    ans = output.eval(feed_dict={x: data[file]}, session=sess)
    ans_list.append(ans)
    
numpy.savez('ans.npz', *ans_list)