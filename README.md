# K-Means Clustering and Visualization with PySpark

This project implements a PySpark-based K-Means clustering pipeline for analyzing mutant phenotypes and visualizing clustering results in 2D and 3D. It integrates PCA for dimensionality reduction and generates interactive visualizations.

## Features

- K-Means clustering with optimal `k` determined by silhouette scores.
- PCA-based dimensionality reduction for 2D and 3D visualization.
- Scalable data processing and machine learning with PySpark.

## Prerequisites

1. Python 3.7+ installed.
2. Required Python libraries:
   - `pyspark`
   - `numpy`
   - `pandas`
   - `matplotlib`

Install dependencies using:

```bash
pip install pyspark numpy pandas matplotlib
```

## Usage

1. Clone the repository
```bash
git clone <repository_url> cd <repository_directory>
```
2. Prepare the input data
The input data should be in a CSV file format. Update the path to your CSV file in the KmeansModel class initialization:
```bash
kmeans_model = KmeansModel('./barcode data.csv')
```
Ensure your CSV file has the following columns:

Mutant	Cell elongation	Shape disruption	Cell lysis	Fluorescence loss	High fluorescence	Thylakoid disruption	Structure formation	Bacteriostasis
3. Run the script
```bash
python <script_name>.py
```
4. Output visualizations
The script will generate:

A silhouette score plot for choosing the optimal number of clusters.
2D and 3D scatter plots of PCA-reduced clustering results, annotated with mutant names.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

This project demonstrates PySpark's MLlib capabilities for clustering and data visualization. Adapt it for your high-dimensional datasets as needed.
