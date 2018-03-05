import urllib2
import os
import tarfile
import inspect
import time
import sys

MAIN_PATH =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
sys.path.append(MAIN_PATH+ "/src")
import std_tensorflow

print("Cleaning Database")
os.system('mclient '+ MAIN_PATH+'/src/dropschema.sql')

print("Downloading Cifar 100")
response = urllib2.urlopen('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
zipcifar= response.read()
with open(MAIN_PATH+"/cifar100.tar.gz", 'w') as f:
    f.write(zipcifar)

tar = tarfile.open(MAIN_PATH+"/cifar100.tar.gz", "r:gz")
tar.extractall()
tar.close()

print("Creating Database")
os.system('mclient '+ MAIN_PATH+'/src/schema.sql')
os.system('mclient '+ MAIN_PATH+'/src/loadimages.sql')
sql = "COPY LOADER INTO image_class FROM loadClass(\'"+MAIN_PATH+"/cifar-100-python\');"
os.system('mclient -s ' +"\"" + sql +"\"")
sql = "COPY LOADER INTO image_superclass FROM loadSuperclass(\'"+MAIN_PATH+"/cifar-100-python\');"
os.system('mclient -s ' +"\"" + sql +"\"")
sql = "COPY LOADER INTO cifar100 FROM loadImages(\'"+MAIN_PATH+"/cifar-100-python\');"
os.system('mclient -s ' +"\"" + sql +"\"")
start_time_monet = time.time()
print("Training Models MonetDB/Tensorflow")
modeldir = os.path.join(MAIN_PATH, "databasemodels")
os.system('mkdir '+ modeldir)
os.system('mclient '+ MAIN_PATH+'/src/trainmodel.sql')
sql = """
CREATE TABLE classification_temporary AS
SELECT trainmodel(id, '%s')
FROM image_superclass
WITH DATA;
INSERT INTO classificationmodel
SELECT * FROM
unpackmodels((SELECT * FROM classification_temporary));
DROP TABLE classification_temporary;
""" % (modeldir,)
os.system('mclient -s ' +"\"" + sql.replace("\n", " ") +"\"")
end_time_monet = time.time()
os.system('mkdir '+ MAIN_PATH+'/tensorflowmodels')
start_time_tensor = time.time()
print("Classifying Models MonetDB/Tensorflow")
os.system('mclient '+ MAIN_PATH+'/src/classification.sql')
sql = "SELECT classification (model_path,name) FROM classificationmodel;"
os.system('mclient -s ' +"\"" + sql +"\"")

print("Training/Classifying Models Standard Tensorflow")
std_tensorflow.run(MAIN_PATH)
end_time_tensor = time.time()

print("--- %s MonetDB seconds ---" % (end_time_monet - start_time_monet))
print("--- %s TensorFlow seconds ---" % (end_time_tensor - start_time_tensor))