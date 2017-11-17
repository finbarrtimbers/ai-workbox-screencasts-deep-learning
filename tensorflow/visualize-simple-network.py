import tensorflow as tf
import datetime
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

# Our first layer
W = tf.get_variable("weights", shape=[784, 10],
                    initializer=tf.glorot_uniform_initializer())
b = tf.get_variable("bias", [10],
                    initializer=tf.constant_initializer(0.1))

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

# We'll use this to make predictions with our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

# We want to record accuracy and see how it changes over time
tf.summary.scalar('accuracy', accuracy)

time_string = datetime.datetime.now().isoformat()
experiment_name = "two-hidden-layers-fully-connected-network" + time_string

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(f'./train/{experiment_name}', sess.graph)
test_writer = tf.summary.FileWriter(f'./test/{experiment_name}')

tf.global_variables_initializer().run()


# We run our training for 1000 steps
MAX_STEPS = 1000
for step in range(MAX_STEPS):
    print(f"training step: {step}/{MAX_STEPS}")

    # we want to see how good our model is
    if step % 10 == 0:
        # Every ten steps, record the accuracy of the model against the test set
        summary, acc = sess.run([merged, accuracy],
                                feed_dict={x: mnist.test.images,
                                           y_: mnist.test.labels})
        test_writer.add_summary(summary, step)
        print(f'Step {step}; Model accuracy: {acc}')
    else:
        # at each step, we get a batch of 100 random images frmo our training set
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # we then run those images through our model
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

        # Every step, record the accuracy of the model against the train set
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs,
                                                               y_: batch_ys})
        train_writer.add_summary(summary, step)
