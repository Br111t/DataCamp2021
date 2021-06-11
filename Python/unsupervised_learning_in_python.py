'''
Clustering for datase exploration
-Unsupervised learning finds patterns in data
-E.g., clustering customers by their purchases 
-Copressing the data using purchase patterns (dimension reduction)

Supervised vs. Unsupervised learning 
-Supervised learning finds patterns for a prediced task 
-E.g., classify tumors as benign or cancerous (labels)
-Unsupervised learning find patterns in data 
-... but without a specific prediction task in mind 

Arrays, features & samples 
* 2D Numpy array 
* Columns are measuremets (the features)
* Rows represent iris plants (the samples)

Iris data is 4-dimensional 
* Iris samples are pooints in 4 dimensional space 
* Dimension = number of features 
* Dimension too high to visualize!
 .. but unsupervised learning gives insight 
 
 k-means clustering 
 -finds clusters of samples 
 -Number of clusters must be specified 
 -Implemented in sklearn ("scikit-learn")
 
 from sklearn.cluster impor KMeans
 model = KMeans(n_clusters=3)
 model.fit(samples) ##Fits the model to the data by locating and remembering the regions were the different classes occur 
 labels = model.predict(samples)##this returns a cluster lable for each sample indicating to which cluster a sample belongs 
 
 Cluster labls for new sampes 
 * New samples can be assigned to existing clusters 
 * k-means remembers the mean of each cluster (the "centroids")
 * Finds the nearest centroid to each new sample 
 
 New samples can be passed to the predict method 
 
 new_labels = model.predict(new_samples)
 
 Generating a scatter plot
 import matplotlib.pyplot as plt
 xs = samples[:,0]
 ys = samples[:,2]
 plt.scatter(xs,ys, c=labels)
 plt.show
 '''
 
 ###Clustering 2D points 
 # Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

###Inspect your clustering 
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y, marker='D', s=50)
plt.show()


'''
Evaluating a cluster 
* Can check correspondence with e.g. iris species 
* ... but what if there are no species to check against?
* Measure quality of a clustering 
* Informs choice of how many clusters to look for 

Iriscluster vs species 
* k-means found 3 clusters amongst the iris samples 
* D the clusters correspond to the species 


Cross tabulation with pandas 
species     setosa      versicolor      virginica
labels
0           0           2               36
1           50          0               0
2           0           48              14


species of each samples is given as strings 

import pandas as pd 
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
ct = pd.crosstab(df['labels'], df['species'])
print(ct)


In most datasets the samples are not labeled by species 
How can the quality of the clustering be evaluated in these cases 

Measuring clustering quality 
* Using only samples and their cluster labels 
* A good clustering has tight clusters
* Samples in each cluster bunched together 

Inertia measures clustering quality 
* Measuring how spread out the clusters are (lower is better)
* Distance from each sample to centroid of its cluster 
* After fit(), available as attribute inertia_

from sklean.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)


How many clusters to choose?
* A good clustering has tight clusters (so low inertia)
* ... but not to many clusters!
* Choose an "elbow" in the ineertia plot
* Where inertia begins to decrease more slowly
* E.g., for iris dataset, 3 is a good choice
'''

###How many clusters of grain?

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


###Evaluating the grain clustering 
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)


'''
Piedmont wines dataset 
* 178 samples from 3 distinct varieties of red wine: Barolo, Grignolino and Barera
* Features measure chemical compoosition e.g. alcohol content 
* Visual propertie like "color intensity"

Clustering the wines 
from sklearn.cluset import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)
df = pd.DataFrame({'labels':labels, 
                    'varieties':varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

Feature variances 
*The wine dataset have very different variances!
*Variance of a feature measures spread of its values  

To give every feature a chance, the data needs to be transformed to the features have equal variance 

StandardScaler
* In kmeans:feature variance = feature influence 
* StandardScalar transforms each featur to have mean 0 and variance 1

sklearn StandardScaler 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)

Similar methods 
* StandardScaler and KMeans have similar methods 
* Use fit() / transform() with StandardScaler
* Use fit() / predict()with KMeans

StandardScaler, then KMeans
* need to perform two steps: StandardScaler, then KMeans
* Use sklearn pipeline to combne multiple steps 
* Data flows from one step into the next 

Pipelines combie multiple steps 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipline import make_pipeline
pipeline = make_pipline(scaler, kmeans)
pipeline.fit(samples)
labels = pipline.predict(samples)
'''

###Scaling fish data for clustering 

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)


###Clustering stocks using KMeans
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels, 'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)


###Clustering stock using KMeans
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)


###Which stocks move together 
# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))


