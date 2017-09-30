import tensorflow as tf
import numpy as np

np_array = np.random.rand(3, 2)
print(np_array)

sess = tf.Session()
with sess.as_default():
    tensor = tf.constant(np_array)
    print(tensor)
    numpy_array_2 = tensor.eval()
    print(numpy_array_2)
