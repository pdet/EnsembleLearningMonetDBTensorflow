import urllib2
import os
import tarfile
import inspect
import time
import sys

MAIN_PATH =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
sys.path.append(MAIN_PATH+ "/src")
import std_tensorflow

# print("Cleaning Database")
# os.system('mclient '+ MAIN_PATH+'/src/dropschema.sql')

# print("Downloading Cifar 100")
# response = urllib2.urlopen('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
# zipcifar= response.read()
# with open(MAIN_PATH+"/cifar100.tar.gz", 'w') as f:
#     f.write(zipcifar)

# tar = tarfile.open(MAIN_PATH+"/cifar100.tar.gz", "r:gz")
# tar.extractall()
# tar.close()

# print("Creating Database")
# os.system('mclient '+ MAIN_PATH+'/src/schema.sql')
# os.system('mclient '+ MAIN_PATH+'/src/loadimages.sql')
# sql = "COPY LOADER INTO image_class FROM loadClass(\'"+MAIN_PATH+"/cifar-100-python\');"
# os.system('mclient -s ' +"\"" + sql +"\"")
# sql = "COPY LOADER INTO image_superclass FROM loadSuperclass(\'"+MAIN_PATH+"/cifar-100-python\');"
# os.system('mclient -s ' +"\"" + sql +"\"")
# sql = "COPY LOADER INTO cifar100 FROM loadImages(\'"+MAIN_PATH+"/cifar-100-python\');"
# os.system('mclient -s ' +"\"" + sql +"\"")
# start_time_monet = time.time()
# print("Training Models MonetDB/Tensorflow")
# os.system('mkdir '+ MAIN_PATH+'/databasemodels')
# os.system('mclient '+ MAIN_PATH+'/src/trainmodel.sql')
# sql = " COPY LOADER INTO classificationmodel FROM trainmodel((select distinct \'"+MAIN_PATH+"/databasemodels\', id from image_superclass));"
# os.system('mclient -s ' +"\"" + sql +"\"")
# end_time_monet = time.time()
os.system('mkdir '+ MAIN_PATH+'/tensorflowmodels')
# start_time_tensor = time.time()

print("Training Models Standard Tensorflow")
std_tensorflow.run(MAIN_PATH)
end_time_tensor = time.time()

print("--- %s MonetDB seconds ---" % (end_time_monet - start_time_monet))
print("--- %s TensorFlow seconds ---" % (end_time_tensor - start_time_tensor))