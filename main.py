import sys
import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyspark.ml.feature import PCA
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler


class KmeansModel:

    def __init__(self, csv_file, num_cores=2):

        # Create pySpark session
        self.pandas_df = None
        self.kmeans = None
        self.centers = None
        self.predictions = None
        self.evaluator = None
        self.spark = SparkSession.builder.master(f'local[{num_cores}]').appName("Mutant Clustering Kmeans").getOrCreate()

        # Make schema.
        self.schema = StructType([
            StructField("Mutant", StringType(), True),
            StructField("Cell elongation", IntegerType(), True),
            StructField("Shape disruption", IntegerType(), True),
            StructField("Cell lysis", IntegerType(), True),
            StructField("Fluorescence loss", IntegerType(), True),
            StructField("High fluorescence", IntegerType(), True),
            StructField("Thylakoid disruption", IntegerType(), True),
            StructField("Structure formation",IntegerType(), True),
            StructField("Bacteriostasis", IntegerType(), True)
        ])


        # Read the data from csv
        try:
            self.df = self.spark.read.csv(csv_file, header=False, schema=self.schema)
            self.df.printSchema()  # Print schema
            self.df.show(5) # Print dataframe head
        except Exception as e:
            print(f"Error reading CSV: {e}")
            self.spark.stop()
            raise

        # Features to normalize
        self.featureCols = ["Cell elongation", "Shape disruption", "Shape disruption", "Cell lysis", "Fluorescence loss",
                        "High fluorescence", "Thylakoid disruption", "Structure formation", "Bacteriostasis"]

        # Create feature vector column
        self.vectorAssembler = VectorAssembler(inputCols = self.featureCols, outputCol = 'features')
        # Convert into dense vector
        self.processed_data = self.vectorAssembler.transform(self.df)
        # print vector head
        # processed_data.select('features').show(5)

        # clustering scoring
        self.silhouette_score=[]


    def fit_kmeans(self):
        # Clustering scoring
        silhouette_score = []
        evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features',
                                        metricName='silhouette', distanceMeasure='squaredEuclidean')

        for i in range(2, 10):
            kmeans = KMeans(featuresCol='features', k=i)
            model = kmeans.fit(self.processed_data)
            prediction = model.transform(self.processed_data)
            score = evaluator.evaluate(prediction)
            silhouette_score.append(score)
            print(f'Silhouette Score for k = {i} is {score}')

        # Plot silhouette scores
        plt.plot(range(2, 10), silhouette_score)
        plt.xlabel('k')
        plt.ylabel('silhouette score')
        plt.title('Silhouette Score')
        plt.show()
        plt.clf()

        # Find the highes K value from silhouette list.
        # print(silhouette_score)
        optimal_k = silhouette_score.index(np.max(silhouette_score)) + 2
        print(f'Optimal number of clusters based on silhouette score: {optimal_k}')

        # Train final K-means model with optimal number of clusters
        self.kmeans = KMeans(featuresCol='features', k=optimal_k)
        self.model = self.kmeans.fit(self.processed_data)
        self.predictions = self.model.transform(self.processed_data)

        # Printing cluster centers
        centers = self.model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)

        self.pandas_df = self.predictions.toPandas()


    def stop(self):
        print('Stopping pySpark')
        self.spark.stop()  # Stop the Spark session

class Plotting:
    def __init__(self, predictions):
        self.predictions = predictions

    def perform_pca(self):
        pca = PCA(k=3, inputCol='features', outputCol='pca_features')
        pca_model = pca.fit(self.predictions)
        pca_res = pca_model.transform(self.predictions)
        return pca_res

    def plot_2d(self, pca_res):
        # Convert PCA features into separate columns for easier plotting using DataFrame
        pandas_df = self.predictions.toPandas()
        pca_df = pca_res.select('pca_features').toPandas()
        pandas_df['pca_x'] = pca_df['pca_features'].apply(lambda x: x[0])
        pandas_df['pca_y'] = pca_df['pca_features'].apply(lambda x: x[1])

        # Function to slightly offset the labels
        def offset_coordinates(x, y, offset=0.05):
            return x + np.random.uniform(-offset, offset), y + np.random.uniform(-offset, offset)

        # 2D Plotting with offsets
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(pandas_df['pca_x'], pandas_df['pca_y'], c=pandas_df['prediction'], cmap='viridis')

        # Add mutant names
        for i, row in pandas_df.iterrows():
            x_offset, y_offset = offset_coordinates(row['pca_x'], row['pca_y'])
            plt.text(x_offset, y_offset, row['Mutant'], fontsize=9, alpha=0.7)

        plt.title('2D K-Means Clustering of Mutants')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster Label')
        plt.show()

    def plot_3d(self, pca_res):
        # Convert PCA features into separate columns
        pandas_df = self.predictions.toPandas()
        pca_df = pca_res.select('pca_features').toPandas()
        pandas_df['pca_x'] = pca_df['pca_features'].apply(lambda x: x[0])
        pandas_df['pca_y'] = pca_df['pca_features'].apply(lambda x: x[1])
        pandas_df['pca_z'] = pca_df['pca_features'].apply(lambda x: x[2])

        # Function to slightly offset the labels
        def offset_coordinates_3d(x, y, z, offset=0.05):
            return (
                x + np.random.uniform(-offset, offset),
                y + np.random.uniform(-offset, offset),
                z + np.random.uniform(-offset, offset)
            )

        # 3D Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pandas_df['pca_x'], pandas_df['pca_y'], pandas_df['pca_z'], c=pandas_df['prediction'],
                             cmap='viridis')

        # Add mutant names as annotations with offsets
        for i, row in pandas_df.iterrows():
            x_offset, y_offset, z_offset = offset_coordinates_3d(row['pca_x'], row['pca_y'], row['pca_z'])
            ax.text(x_offset, y_offset, z_offset, row['Mutant'], fontsize=9, alpha=0.7)

        ax.set_title('3D K-Means Clustering of Mutants')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        fig.colorbar(scatter, label='Cluster Label')
        plt.show()


if __name__ == '__main__':

    print('Running pySpark')
    print('-----------------------------')
    print("Packages versions:")
    print('numpy      :', np.__version__)
    print('pandas     :', pd.__version__)
    print('matplotlib :', sys.modules['matplotlib'].__version__)
    print('seaborn    :', np.__version__)
    print('pyspark    :', pyspark.__version__)
    # Initialize KMeans model
    kmeans_model = KmeansModel('./barcode data.csv')
    # Fit the model
    kmeans_model.fit_kmeans()
    # Get plotting instance
    plotting = Plotting(kmeans_model.predictions)
    # Run PCA
    pca_res = plotting.perform_pca()
    # plot data
    plotting.plot_2d(pca_res)
    plotting.plot_3d(pca_res)


    # Stop the Spark session
    kmeans_model.spark.stop()


