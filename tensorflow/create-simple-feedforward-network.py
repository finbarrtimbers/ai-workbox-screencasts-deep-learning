import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# We load the MNIST dataset using a helper function; in future lessons we'll
# cover how to load other datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

# the placeholder to contain the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# our scoring function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)

# To train the model, we use gradient descent, with a learning step of 0.5
# What this does is tells Tensorflow to use Gradient Descent, with a learning
# rate of 0.001, to find the weights that minimize the cross_entropy of our model
learning_rate = 0.001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# We run our training for 1000 steps
MAX_STEPS = 50
for step in range(MAX_STEPS):
    print(f"training step: {step}/{MAX_STEPS}")

    # at each step, we get a batch of 100 random images frmo our training set
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # we then run those images through our model
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
