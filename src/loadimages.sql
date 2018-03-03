CREATE OR REPLACE LOADER loadClass(folder_path STRING) LANGUAGE PYTHON {
import os
import cPickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
labels = unpickle(os.path.join(folder_path, 'meta'))
_emit.emit({'id': numpy.arange(len(labels['fine_label_names'])),
    'label': labels['fine_label_names']})
};

CREATE OR REPLACE LOADER loadSuperclass(folder_path STRING) LANGUAGE PYTHON {
import os
import cPickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


labels = unpickle(os.path.join(folder_path, 'meta'))

_emit.emit({'id': numpy.arange(len(labels['coarse_label_names'])),
    'label': labels['coarse_label_names']})
};


CREATE OR REPLACE LOADER loadImages(folder_path STRING) LANGUAGE PYTHON {
import os
import cPickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


train_set = unpickle(os.path.join(folder_path, 'train'))
test_set = unpickle(os.path.join(folder_path, 'test'))

del train_set['batch_label']
del train_set['filenames']
del test_set['batch_label']
del test_set['filenames']

train_set['class'] = train_set.pop('fine_labels')
train_set['superclass'] = train_set.pop('coarse_labels')
test_set['class'] = test_set.pop('fine_labels')
test_set['superclass'] = test_set.pop('coarse_labels')


train_set['data'] = [cPickle.dumps(x) for x in train_set['data']]
test_set['data'] = [cPickle.dumps(x) for x in test_set['data']]
i=0
for x in train_set['data']:
    _emit.emit({'data': x, 'class': train_set['class'][i], 'superclass': train_set['superclass'][i],'train':True})
    i = i + 1
i=0     
for x in test_set['data']:
    _emit.emit({'data': x, 'class': train_set['class'][i], 'superclass': train_set['superclass'][i],'train':False})
    i = i + 1
};