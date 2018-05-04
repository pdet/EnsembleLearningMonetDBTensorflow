CREATE OR REPLACE FUNCTION classification (model_path string, model_name string)
RETURNS INTEGER
LANGUAGE PYTHON {
import tensorflow as tf
import pickle
import numpy as np
from multiprocessing.pool import ThreadPool

test_images = _conn.execute("SELECT data,superclass FROM cifar100 WHERE train=False;")
xs = []
ys = []
for i in range(len(test_images['data'])):
    xs.append(pickle.loads(test_images['data'][i]))
    ys.append(test_images['superclass'][i])
xs = np.array(xs)
images_test = xs.reshape(xs.shape[0], 32 * 32 * 3) 

def classify_images(model_name):
	mpath = model_path[0]+"/"+model_name+'.meta'
	if mpath not in globals():
		sess=tf.Session()    
		new_saver = tf.train.import_meta_graph(mpath+'.meta')
		new_saver.restore(sess, mpath)
		globals()[mpath] = sess
	else:
		sess = globals()[mpath]
	graph = tf.get_default_graph()
	images_placeholder = graph.get_tensor_by_name("images_placeholder:0")
	labels_placeholder = graph.get_tensor_by_name("labels_placeholder:0")
	logits = tf.get_collection("logits")[0]
	classifiedtype =sess.run(tf.argmax(logits, 1), feed_dict={images_placeholder:images_test})

p = ThreadPool(len(model_name))
p.map(classify_images, model_name)

return 0
};
