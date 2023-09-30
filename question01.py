#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:25:46 2023

@author: chchasong
"""

#1.	Retrieve and load the Olivetti faces dataset.
from sklearn.datasets import fetch_olivetti_faces

# Fetch the Olivetti faces dataset
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=96)
X = olivetti_faces.images  # Images
y = olivetti_faces.target  # Labels

# Display 25 faces from the dataset
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.ravel()

for i in range(25):
    axes[i].imshow(X[i], cmap='gray')
    axes[i].set_title(f"Label: {y[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()


#2.	Split the training set, a validation set, 
#and a test set using stratified sampling to ensure that there are the same number of images per person in each set. 
#Provide your rationale for the split ratio.
from sklearn.model_selection import train_test_split

# Split into training and temporary set (80:20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=96)

print(f"Temporary set: {X_temp.shape}, {y_temp.shape}")

# Split the temporary set into validation and test sets (50:50) so that the final ratio becomes 60:20:20
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=96)


print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")


#Reshape the data 
X_train = X_train.reshape((X_train.shape[0], -1))
X_val = X_val.reshape((X_val.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

#3.	Using k-fold cross validation, train a classifier to predict which person is represented in each picture, 
#and evaluate it on the validation set. 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

clf = SVC(kernel='linear', C=1)

try:
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, error_score='raise')
    print(f"Cross Validation Scores: {cv_scores}")
    print(f"Average 5 Fold CV Score: {np.mean(cv_scores)}")
except Exception as e:
    print(f"An error occurred: {e}")

# Fit the model to the complete training set
clf.fit(X_train, y_train)

# Evaluate the model on the validation set
val_score = clf.score(X_val, y_val)
print(f"Validation Set Score: {val_score}")

# plot the cross-validation scores 
plt.bar(range(len(cv_scores)), cv_scores)
plt.title("Cross-Validation Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()

#4.	Use K-Means to reduce the dimensionality of the set. 
#Provide your rationale for the similarity measure used to perform the clustering. 
#Use the silhouette score approach to choose the number of clusters. 

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_clusters_list = range(2, 15)

best_n_clusters = 0
best_silhouette = -1

silhouette_scores = []
# Calculate silhouette scores for different numbers of clusters
for n_clusters in n_clusters_list:  
    kmeans = KMeans(n_clusters=n_clusters, random_state=96)
    cluster_labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    print(f"For n_clusters = {n_clusters}, Silhouette score is {silhouette_avg}")
    silhouette_scores.append(silhouette_avg)    
    if silhouette_avg > best_silhouette:
        best_n_clusters = n_clusters
        best_silhouette = silhouette_avg

#Plot the silhouette scores for different numbers of clusters 
plt.plot(n_clusters_list, silhouette_scores)
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
        
# Fit the KMeans model with the best number of clusters    
kmeans = KMeans(n_clusters=best_n_clusters, random_state=96)
kmeans.fit(X_train)

#Transform the data points to their corresponding cluster centers
X_train_reduced = kmeans.transform(X_train)
X_val_reduced = kmeans.transform(X_val)
X_test_reduced = kmeans.transform(X_test)

print(f"Optimal number of clusters: {best_n_clusters}")

#5.	Use the set from step (4) to train a classifier as in step (3)    

#Train the same classifier as in step 3 but on the reduced dataset
clf_reduced = SVC(kernel='linear', C=1)

try:
    # Perform 5-fold cross-validation on the reduced training set
    cv_scores_reduced = cross_val_score(clf_reduced, X_train_reduced, y_train, cv=5, error_score='raise')
    print(f"Cross Validation Scores on reduced set: {cv_scores_reduced}")
    print(f"Average 5 Fold CV Score on reduced set: {np.mean(cv_scores_reduced)}")
except Exception as e:
    print(f"An error occurred: {e}")

# Fit the model to the complete training set
clf_reduced.fit(X_train_reduced, y_train)

# Evaluate the model on the reduced validation set
val_score_reduced = clf_reduced.score(X_val_reduced, y_val)
print(f"Validation Set Score on reduced set: {val_score_reduced}")

#Plot the cross-validation scores for the reduced dataset
plt.bar(range(len(cv_scores_reduced)), cv_scores_reduced)
plt.title("Cross-Validation Scores on Reduced Set")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()

#6.	Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to the Olivetti Faces dataset for clustering. 
#Preprocess the images and convert them into feature vectors, 
#then use DBSCAN to group similar images together based on their density. 
#Provide your rationale for the similarity measure used to perform the clustering, considering the nature of facial image data. 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#Standardize the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)

dbscan_labels_train = dbscan.fit_predict(X_train_scaled)

unique_labels = np.unique(dbscan_labels_train)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  
print(f"Number of clusters: {n_clusters}")

if n_clusters > 1:  
    silhouette_avg = silhouette_score(X_train_scaled, dbscan_labels_train)
    print(f"Silhouette score is {silhouette_avg}")
else:
    print("Cannot calculate silhouette score for a single cluster.")


#Scatter plot for DBSCAN
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels_train, cmap='rainbow')
plt.title("DBSCAN Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()