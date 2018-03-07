import os
import cPickle
import pickle
import tensorflow as tf
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def specializedModel(main_path,superclass,ys,ystest,images_train,images_test):
    model_path = main_path + "/tensorflowmodels/"

    batch_size = [100,1000,10000]
    learning_rate = [0.5,0.05,0.005]
    epochs = [20]
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
                        classifiedtype =sess.run(tf.argmax(logits, 1), feed_dict={images_placeholder:images_test})
                    if np.sum(classifiedtype == labels_test) > accuracy:
                        accuracy = np.sum(classifiedtype == labels_test)
                        best_batch_size = batch_size
                        best_learning_rate = learning_rate
                        best_epoch = epoch
                        saver = tf.train.Saver()
                        saver.save(sess, model_path+str(superclass))

def classify(model_path,model_name,test_images):
    model_path = model_path + "/tensorflowmodels/"
    sess=tf.Session()    
    new_saver = tf.train.import_meta_graph(model_path+model_name+'.meta')
    new_saver.restore(sess, model_path +model_name)
    graph = tf.get_default_graph()
    images_placeholder = graph.get_tensor_by_name("images_placeholder:0")
    labels_placeholder = graph.get_tensor_by_name("labels_placeholder:0")
    logits = tf.get_collection("logits")[0]
    classifiedtype =sess.run(tf.argmax(logits, 1), feed_dict={images_placeholder:test_images})


def run(main_path):
    folder_path = main_path +"/cifar-100-python"
    train_set = unpickle(os.path.join(folder_path, 'train'))
    test_set = unpickle(os.path.join(folder_path, 'test'))

    train_set['class'] = train_set.pop('fine_labels')
    train_set['superclass'] = train_set.pop('coarse_labels')
    test_set['class'] = test_set.pop('fine_labels')
    test_set['superclass'] = test_set.pop('coarse_labels')
    xs = []
    ys = []
    xstest = []
    ystest = []
    for i in range(len(train_set['data'])):
        xs.append(train_set['data'][i])
        ys.append(train_set['superclass'][i])
    xs = np.array(xs)
    images_train = xs.reshape(xs.shape[0], 32 * 32 * 3) 
    for i in range(len(test_set['data'])):
        xstest.append(test_set['data'][i])
        ystest.append(test_set['superclass'][i])
    xstest = np.array(xstest)
    images_test = xstest.reshape(xstest.shape[0], 32 * 32 * 3) 
    for i in range(20):
        print(i)
        specializedModel(main_path,i,ys,ystest,xs,xstest)
    for i in range(20):
        print(i)
        classify(main_path,str(i),xstest)