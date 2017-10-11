import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'training',
                    'Directory where data to run model is found')
flags.DEFINE_string('checkpoint', None,
                    'If it exists, loads model from checkpoint')
flags.DEFINE_boolean('train', False, 'If True trains model for 100 steps')
FLAGS = flags.FLAGS

print(FLAGS.data_dir)
print(FLAGS.train)
print(FLAGS.checkpoint)
quit()

# https://gist.github.com/eerwitt/518b0c9564e500b4b50f
# Download the data from here, unzip into two folders" training, and test:
#
# https://www.kaggle.com/scolianni/mnistasjpg/data
#
train_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/trainingSample/*/*.jpg")
)
test_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("data/testSample/*.jpg")
)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
train_image_reader = tf.WholeFileReader()
test_image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, train_image_file = train_image_reader.read(train_filename_queue)
_, test_image_file = test_image_reader.read(test_filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
train_image = tf.image.decode_jpeg(train_image_file)
train_image = tf.image.decode_jpeg(test_image_file)

# Generate batches
num_preprocess_threads = 1
min_queue_examples = 256
train_images = tf.train.shuffle_batch(
    [train_image],
    batch_size=32,
    num_threads=1
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)
test_images = tf.test.shuffle_batch(
    [test_image],
    batch_size=32,
    num_threads=1
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)

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

# We'll use this to make predictions with our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

if FLAGS.checkpoint:


# We run our training for 1000 steps
if not FLAGS.test:
    MAX_STEPS = 50
    for step in range(MAX_STEPS):
        print(f"training step: {step}/{MAX_STEPS}")

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
