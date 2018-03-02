import urllib2
import os
import tarfile
import inspect
MAIN_PATH =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
print("Downloading Cifar 100")
response = urllib2.urlopen('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
zipcifar= response.read()
with open("cifar100.tar.gz", 'w') as f:
    f.write(zipcifar)

tar = tarfile.open("cifar100.tar.gz", "r:gz")
tar.extractall()
tar.close()

print("Creating Database")
os.system('mclient src/schema.sql')
os.system('mclient src/loadimages.sql')
sql = "COPY LOADER INTO image_class FROM loadClass(\'"+MAIN_PATH+"cifar-100-python\');"
os.system('mclient -s ' + sql)
sql = "COPY LOADER INTO image_superclass FROM loadSuperclass(\'"+MAIN_PATH+"cifar-100-python\');"
os.system('mclient -s ' + sql)
sql = "COPY LOADER INTO cifar100 FROM loadImages(\'"+MAIN_PATH+"cifar-100-python\');"
os.system('mclient -s ' + sql)

print("Training Models MonetDB/Tensorflow")


print("Classification MonetDB/Tensorflow")

print("Training Models Standard Tensorflow")


print("Classification Standard Tensorflow")

