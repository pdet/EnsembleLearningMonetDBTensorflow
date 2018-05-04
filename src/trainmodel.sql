CREATE OR REPLACE AGGREGATE trainmodel(sclass INTEGER, model_path STRING)
RETURNS INTEGER
LANGUAGE PYTHON_MAP
{
import cPickle as pickle
import tensorflow as tf
import numpy as np
model_path = model_path[0]
train_images = _conn.execute("SELECT data, superclass FROM cifar100 WHERE train=True LIMIT 4;")
test_images = _conn.execute("SELECT data, superclass FROM cifar100 WHERE train=False LIMIT 1;")
xs = []
ys = []
xstest = []
ystest = []
for i in range(len(train_images['data'])):
    xs.append(pickle.loads(str(train_images['data'][i])))
    ys.append(train_images['superclass'][i])
xs = np.array(xs)
images_train = xs.reshape(xs.shape[0], 32 * 32 * 3) 
for i in range(len(test_images['data'])):
    xstest.append(pickle.loads(str(test_images['data'][i])))
    ystest.append(test_images['superclass'][i])
xstest = np.array(xstest)
images_test = xstest.reshape(xstest.shape[0], 32 * 32 * 3) 

batch_size = [100,1000,10000]
learning_rate = [0.5,0.05,0.005]
epochs = [100,1000,10000]

for superclass in sclass:
    accuracy = -1
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
    images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072], name='images_placeholder')
    labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='labels_placeholder')
    weights = tf.Variable(tf.zeros([3072, 2]))
    biases = tf.Variable(tf.zeros([2]))
    logits = tf.matmul(images_placeholder, weights) + biases
    tf.add_to_collection("logits", logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels_placeholder))
    with tf.Session() as sess:
        for batch in batch_size:
            for learning in learning_rate:
                for epoch in epochs:
                    train_step = tf.train.GradientDescentOptimizer(learning).minimize(loss)
                    sess.run(tf.global_variables_initializer())
                    for i in range(epoch):
                        indices = np.random.choice(images_train.shape[0], batch)
                        images_batch = images_train[indices]
                        labels_batch = labels_train[indices]
                        sess.run(train_step, feed_dict={images_placeholder: images_batch,labels_placeholder: labels_batch})
                        classifiedtype = sess.run(tf.argmax(logits, 1), feed_dict={images_placeholder:images_test})
                        new_accuracy = np.sum(classifiedtype == labels_test)
                        if new_accuracy > accuracy:
                            accuracy = new_accuracy
                            best_batch_size = batch
                            best_learning_rate = learning
                            best_epoch = epoch
                            saver = tf.train.Saver()
                            mpath = model_path+str(superclass)
                            saver.save(sess, mpath)
                            globals()[mpath] = sess
    _conn.execute("""
       INSERT INTO classificationmodel 
        (name, model_path, batch_size, learning_rate, epoch, image_superclass_id)
       VALUES
        ('%s', '%s', %d, %f, %d, %d);
        """ % (str(superclass), model_path, best_batch_size, best_learning_rate, best_epoch, superclass))
return 1;
};