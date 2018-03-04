# EnsembleLearningMonetDBTensorflow

Use MonetDB/Python(i.e.,vectorized python User-Defined Functions (UDFs)) to perform ensemble learning. This example showcases how machine learning models can be stored inside MonetDB, allowing for better management of models and enabling model meta-analysis using relational queries. In addition, we utilize the automatic parallelization of the database to train/classify in parallel.

## Dependencies
Tensorflor
MonetDB

Start Mserver: mserver5 --set gdk_nr_threads=20 --forcemito --set embedded_py=true

Execute python experiment.py
