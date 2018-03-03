CREATE OR REPLACE LOADER trainmodel (model_path string) LANGUAGE PYTHON 
{   
import pickle
import tensorflow as tf
import numpy as np
# numpyimage blob, label integer ,model_id integer, model_path string, model_name string, batch_size float,learning_rate float, epoch float
train_images = _conn.execute("SELECT data,superclass FROM cifar100 WHERE train=True;")
test_images = _conn.execute("SELECT data,superclass FROM cifar100 WHERE train=False;")
xs = []
ys = []
xstest = []
ystest = []
for i in range(len(train_images['data'])):
    xs.append(pickle.loads(train_images['data'][i]))
    ys.append(train_images['superclass'][i])
xs = np.array(xs)
# Xtr_rows becomes 50000 x 3072, This is due so it can fit the Weights Matrix.
images_train = xs.reshape(xs.shape[0], 32 * 32 * 3) 
for i in range(len(test_images['data'])):
    xstest.append(pickle.loads(test_images['data'][i]))
    ystest.append(test_images['superclass'][i])
xstest = np.array(xstest)
# Xtr_rows becomes 50000 x 3072, This is due so it can fit the Weights Matrix.
images_test = xstest.reshape(xstest.shape[0], 32 * 32 * 3) 
# Parameter definitions
batch_size = 100
learning_rate = 0.5
epoch = 10000
#batch_size = [100,1000,10000]
#learning_rate = [0.5,0.05,0.005]
#epoch = [10000,100000,1000000]
# Get exactly the same result on each run
print("top")
for superclass in range(1):
  accuracy = 0
  best_batch_size = 0
  best_learning_rate = 0
  best_epoch = 0
  labels_train = np.array(ys)
  labels_train[labels_train==superclass] = -1
  labels_train[labels_train!=superclass] = 0
  labels_train[labels_train==-1] = 1
  labels_test = np.array(ystest)
  labels_test[labels_test==superclass] = -1
  labels_test[labels_test!=superclass] = 0
  labels_test[labels_test==-1] = 1
  np.random.seed(1)
  # Define input placeholders, since they will need to be accessed again we need to define names.
  images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072], name='images_placeholder')
  labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='labels_placeholder')
  # Define variables (these are the values we want to optimize)
  weights = tf.Variable(tf.zeros([3072, 2]))
  biases = tf.Variable(tf.zeros([2]))
  # Define the classifier's result
  logits = tf.matmul(images_placeholder, weights) + biases
  #Similar to the placeholdes we need to define a name for the logits in order to access them again later.
  tf.add_to_collection("logits", logits)
  # Define the loss function
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    labels=labels_placeholder))
  # Define the training operation
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  print("top2")
  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
   # Repeat max_steps times
    for i in range(epoch):
      # Generate input data batch
      indices = np.random.choice(images_train.shape[0], batch_size)
      images_batch = images_train[indices]
      labels_batch = labels_train[indices]
      # Perform a single training step
      sess.run(train_step, feed_dict={images_placeholder: images_batch,labels_placeholder: labels_batch})
      #Actually perform the classification for the images
      classifiedtype =sess.run(tf.argmax(logits, 1), feed_dict={images_placeholder:images_test})
      print("top3")
    if np.sum(classifiedtype == labels_test) > accuracy:
      print(accuracy)
      accuracy = np.sum(classifiedtype == labels_test)
      best_batch_size = batch_size
      best_learning_rate = learning_rate
      best_epoch = epoch
      #Save Trained Session
      saver = tf.train.Saver()
      saver.save(sess, model_path+str(superclass))
  _emit.emit( {'name': str(superclass), 'model_path': model_path, 'batch_size': best_batch_size, 'learning_rate':best_learning_rate, 'epoch':best_epoch, 'image_superclass_id' : superclass})
};