import sys
import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.feature import PCA
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler


class KmeansModel:

    def __init__(self, csv_file, num_cores=2):

        # Create pySpark session
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


    def evaluate_clustering(self):
        self.evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features',
                                             metricName='silhouette', distanceMeasure='squaredEuclidean')
        for i in range(2, 10):
            kmeans = KMeans(featuresCol='features', k=i)
            model = kmeans.fit(self.processed_data)
            prediction = model.transform(self.processed_data)
            score = self.evaluator.evaluate(prediction)
            self.silhouette_score.append(score)
            print(f'Silhouette Score for k = {i} is {score}')

        return self.silhouette_score


    def fit_kmeans(self, k_iter):
        # Trains a k-means model.
        kmeans = KMeans(featuresCol='features', k=k_iter)
        model = kmeans.fit(self.processed_data)
        self.predictions = model.transform(self.processed_data)
        self.centers = model.clusterCenters()
        # Printing cluster centers
        print("Cluster Centers: ")
        for center in self.centers:
            print(center)

        return self.predictions


    def stop(self):
        self.spark.stop()  # Stop the Spark session

class Plotting:
    def __init__(self, predictions):
        self.predictions = predictions

    def perform_pca(self):
        pca = PCA(k=2, inputCol='features', outputCol='pca_features')
        pca_model = pca.fit(self.predictions)
        pca_res = pca_model.transform(self.predictions)
        return pca_res

    def plot_2d(self, pca_res):
        # Convert PCA features into separate columns for easier plotting
        pandas_df = self.predictions.toPandas()
        pandas_df['pca_x'] = pca_res.select('pca_features').rdd.map(lambda x: x[0][0]).collect()
        pandas_df['pca_y'] = pca_res.select('pca_features').rdd.map(lambda x: x[0][1]).collect()

        # Function to slightly offset the labels
        def offset_coordinates(x, y, offset=0.05):
            return x + np.random.uniform(-offset, offset), y + np.random.uniform(-offset, offset)

        # 2D Plotting with offsets
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(pandas_df['pca_x'], pandas_df['pca_y'], c=pandas_df['prediction'], cmap='viridis')

        # Add mutant names as annotations with offsets
        for i, row in pandas_df.iterrows():
            x_offset, y_offset = offset_coordinates(row['pca_x'], row['pca_y'])
            plt.text(x_offset, y_offset, row['Mutant'], fontsize=9, alpha=0.7)

        plt.title('2D K-Means Clustering of Mutants')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Cluster Label')
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

    # Evaluate clustering
    kmeans_model.evaluate_clustering()

    # Fit KMeans model with the desired number of clusters
    predictions = kmeans_model.fit_kmeans(k_iter=8)

    # Initialize Plotting class
    plotting = Plotting(predictions)

    # Perform PCA and plot results
    pca_result = plotting.perform_pca()
    plotting.plot_2d(pca_result)

    # Stop Spark session
    kmeans_model.spark.stop()

