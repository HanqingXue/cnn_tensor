import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from loadData import *

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", shape=[None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# train data and get results for batches
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# train the data
train, trainLabel = prasedata()
test,  testLabel  = praseTestdata()
batch_size = 50
n_epochs   = 20

output = open('output.txt', 'w')

for epoch_i in range(n_epochs):
    start = 0
    end = batch_size
    for i in range(len(train) // batch_size):
        batch_xs = train[start:end]
        batch_ys = trainLabel[start:end]
        start += batch_size
        end += batch_size
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("accuracy", sess.run(accuracy, feed_dict={x: test, y_: testLabel}))
    prediction=tf.argmax(y,1)
    labels = prediction.eval(feed_dict={x: test}, session=sess)
    print ("predictions", prediction.eval(feed_dict={x: test}, session=sess))
    if epoch_i == 7:
    	print (list(labels))
    	for label in labels:
    		output.write(str(label) + '\n')
