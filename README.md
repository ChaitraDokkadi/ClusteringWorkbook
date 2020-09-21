# ClusteringWorkbook
Implemented K-means, Hierarchical Agglomerative clustering with Min approach, density-based, mixture model, and spectral clustering algorithms and compared the results between the 5 clustering algorithms

## K-means clustering
	Steps to run K-means for given data in cho.txt file:
	  1. Open cmd in current folder which contains k-means.py and cho.txt file
	  2. Type the following command in the cmd:
			'python k-means.py'
	  3. Enter file name 
		e.g. cho.txt
	  4. Enter number of clusters
		e.g. 5
	  5.To enter centroids, select Y or N. If input is N, centroids will be selected at random.
		e.g. Y
	  6.Enter centroids indices one after the other
		e.g. 35

## HAC algorithm
	Steps to run HAC algorithm for given data in cho.txt file:
	  1. Open cmd in current folder which contains HAC.py and cho.txt file
	  2. Type the following command in the cmd:
			'python HAC.py'
	  3. Enter file name 
		e.g. cho.txt
	  4. Enter number of clusters
		e.g. 5

## Density based clustering:
	Steps to run DBSCAN clustering algorithm for cho.txt dataset:
	  1. Open cmd in current folder which contains density_clustering.py and cho.txt file
	  2. Type the following command in the cmd:
			'python density_clustering.py'
	  3. Enter file name 
		e.g. cho.txt
	  4. Enter minimum number of points
		e.g. 5
	  5. Enter eps
		e.g. 1.4
	  6. The clustering plot will be outputted with Jaccard coefficient and Rand Index
  
## Spectral Clustering:
	Steps to run spectral clustering algorithm for cho.txt dataset:
	  1. Open cmd in current folder which contains spectral_clustering.py and cho.txt file
	  2. Type the following command in the cmd:
			'python spectral_clustering.py'
	  3. Enter file name 
		e.g. cho.txt
	  4. Enter number of clusters
		e.g. 5
	  5. Enter sigma
		e.g. 0.8
	  6. To enter centroids, select Y or N. If input is N, centroids will be selected at random.
		e.g. Y
	  7. Enter centroids indices one after the other
		e.g. 35
	  8. The clustering plot will be outputted with Jaccard coefficient and Rand Index

## GMM algorithm
	Steps to run GMM for given data in cho.txt file:
	  1. Open cmd in current folder which contains GMM_clustering.py and cho.txt file
	  2. Type the following command in the cmd:
			'python GMM_clustering.py'
	  3. Enter file name 
		e.g. cho.txt
	  4. Enter number of clusters
		e.g. 5
	  5. The clustering plot will be outputted with Jaccard coefficient and Rand Index
