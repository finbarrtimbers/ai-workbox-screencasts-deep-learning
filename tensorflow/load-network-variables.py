import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# We load the MNIST dataset using a helper function; in future lessons we'll
# cover how to load other datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def create_mnist_model():
    # create tf variables and initialize them
    # x is going to be our data, which are 28 x 28 pixel images. We represent them
    # here as a series of N 784 length vectors (784 == 28 * 28)
    # For the shape of x, we set it to be a None x 784 size tensor as we don't know
    # the number of images we're inputting, but we know each image is a 784
    # dimensional vector
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # Because we're using a ReLU activation function, you want your variables to be
    # initialized to a positive value, which is why we have a bias of 0.1
    # See Deep Learning, by Goodfellow et. al, for more insight into the theory
    W = tf.get_variable("weights", shape=[784, 10],
                        initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable("bias", [10],
                        initializer=tf.constant_initializer(0.1))

    # the output of our simple model
    y = tf.nn.relu(tf.matmul(x, W) + b)

    return x, y

# Our model output (logits)
x, y = create_mnist_model()

# the placeholder to contain the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# We'll use this to make predictions with our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print("Model accuracy after random initialization: ")
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

print("Restoring model...")

saver = tf.train.Saver()
saver.restore(sess, "model.ckpt")

print("Model accuracy after restoration: ")
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
