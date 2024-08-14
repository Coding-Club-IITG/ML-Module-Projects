import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class DBSCAN:
    def __init__(self,eps:float,minPts:int) -> None:
        super().__init__()
        self.eps=eps
        self.minPts=minPts
        
    #Function to find index of neighbors of a point. All points whose distances are lower than eps from the given point are neighbors
    def _findNeighbors(self,data,coreCandiIndex):
        neighbour=[]
        for pointIdx in range(len(data)):
            if (np.linalg.norm(data.iloc[pointIdx]-data.iloc[coreCandiIndex]))<=self.eps:
                neighbour.append(pointIdx)
        return neighbour

    #Function to check if points are density connected or not and then assign them the same cluster labels
    def _createCluster(self,data,neighbours,coreCandiIndex,labels,clusterLabel):
        labels[coreCandiIndex]=clusterLabel
        i=0
        while i < len(neighbours):
            pointIdx=neighbours[i]
            if (labels[pointIdx]==-1):
                labels[pointIdx]=clusterLabel
            elif (labels[pointIdx]==0):
                labels[pointIdx]=clusterLabel
                pointNeighbours=self._findNeighbors(data,pointIdx)
                if (len(pointNeighbours)>=self.minPts):
                    neighbours=neighbours+pointNeighbours
            i+=1

    #Fit method
    def fit(self,data):
        labels=[0]*len(data)
        clusterLabel=0
        for i in range(len(data)):
            if (labels[i]!=0):
                continue
            neighbour=self._findNeighbors(data,i)
            if (len(neighbour)<self.minPts):
                labels[i]= -1
            else:
                clusterLabel+=1
                self._createCluster(data,neighbour,i,labels,clusterLabel)
        return labels

    #Plot method for our dbscan clustered data
    def plotClusters(self,df,labels):
        plt.figure(figsize=(10,10))
        plt.scatter(df[0],df[1],c=labels)
        plt.title('DBSCAN Clustering',fontsize=20)
        plt.xlabel('Feature 1',fontsize=14)
        plt.ylabel('Feature 2',fontsize=14)
        plt.show()

    #Curve to find out efficient eps vlue
    def optimizationCurve(self,df):
        neigh = NearestNeighbors(n_neighbors=self.minPts)
        nbrs = neigh.fit(df[[0,1]])
        distances, indices = nbrs.kneighbors(df[[0,1]])
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.figure(figsize=(10,10))
        plt.plot(distances)
        plt.title('K-distance Graph',fontsize=20)
        plt.xlabel('Data Points sorted by distance',fontsize=14)
        plt.ylabel('Epsilon',fontsize=14)
        plt.show()