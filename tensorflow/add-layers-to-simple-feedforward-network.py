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
W1 = tf.get_variable("weights_1", shape=[784, 392],
                    initializer=tf.glorot_uniform_initializer())
b1 = tf.get_variable("bias_1", [392],
                    initializer=tf.constant_initializer(0.1))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# our second relu layer
W2 = tf.get_variable("weights_2", shape=[392, 196],
                    initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable("bias_2", [196],
                    initializer=tf.constant_initializer(0.1))
h2 =  tf.nn.relu(tf.matmul(h1, W2) + b2)

# our third relu layer
W3 = tf.get_variable("weights_3", shape=[196, 10],
                    initializer=tf.glorot_uniform_initializer())
b3 = tf.get_variable("bias_3", [10],
                    initializer=tf.constant_initializer(0.1))

y = tf.nn.relu(tf.matmul(h2, W3) + b3)

# our first convolutional layer
#conv1_input = tf.reshape(x, [-1, 28, 28, 1])

#conv1 = tf.layers.conv2d(inputs=conv1_input,
#                         filters=3,
#                         kernel_size=[5, 5],
#                         padding='valid')
#print(conv1.get_shape())
#conv1 = tf.reshape(conv1, [-1, 1728])
#W1 = tf.get_variable("weights", shape=[1728, 10],
#                    initializer=tf.glorot_uniform_initializer())
#y = tf.nn.relu(tf.matmul(conv1, W1) + b1)


# the placeholder to contain the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# our scoring function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)

# To train the model, we use gradient descent, with a learning step of 0.5
# What this does is tells Tensorflow to use Gradient Descent, with a learning
# rate of 0.001, to find the weights that minimize the cross_entropy of our model
learning_rate = 0.001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# We'll use this to make predictions with our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# We run our training for 50 steps
MAX_STEPS = 50
for step in range(MAX_STEPS):
    #print(f"training step: {step}/{MAX_STEPS}")

    # at each step, we get a batch of 100 random images frmo our training set
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # we then run those images through our model
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # we want to see how good our model is
    if step % 10 == 0:
        print("model accuracy: ")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))
print("final model accuracy: ")
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
