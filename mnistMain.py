import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 55,000개의 학습데이터 = mnist.train
# 10,000개의 테스트 데이터 mnist.text
# 5,000개의 검증데이터 mnist.validation


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



#Embedding 2
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = 'logsWithClass'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

images = tf.Variable(mnist.test.images, name='images')

with open(metadata, 'w') as metadata_file:
    for row in mnist.test.labels:
        metadata_file.write('%s\n' % row)

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)



# https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector
# Here first we create a TensoFlow variable (images) and then save it using tf.train.Saver.
# After executing the code we can launch TensorBoard by issuing tensorboard --logdir=logs command and
# opening localhost:6006 in a browser.



#Embedding 1
# LOG_DIR = 'logs'
# images = tf.Variable(mnist.test.images, name='images')
#
# with tf.Session() as sess:
#     saver = tf.train.Saver([images])
#
#     sess.run(images.initializer)
#     saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))
