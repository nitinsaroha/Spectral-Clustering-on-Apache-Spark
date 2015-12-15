# Stochastic-Gradient-Descent-for-Matrix-Factorization-on-spark
Implementation of Spectral Clustering in Apache Spark.

Dataset used is synthetic data, generated on-the-fly using random number generators (specifically,
the scikit-learn samples generators); they don’t represent any “real” data

Used Matplot Library for plotting the clusters

#### How to Run
  **$** **spark-submit ~/absolute/path/to/the/directory/spectral_clustering.py 3 10 1.0 a5_data/blobs.txt**
  
- Argument 1 is the number of clusters
- Argument 2 is the upper bound
- Argument 3 is the value of gamma

*Note:- spark-submit should be in path*