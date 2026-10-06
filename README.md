## Customer Age and Spending Segmentation
A machine learning project that uses K-Means clustering to group customers according to their age and spending behavior.
## Project Overview
This project analyzes customer data using two features:
-Customer age
-Spending score
The goal is to identify groups of customers with similar characteristics. These customer segments can help businesses understand their audience and design more targeted marketing strategies.
## Dataset
The project uses:
customer_age_spending_score.csv
| Column         | Description                |
| -------------- | -------------------------- |
| CustomerID     | Unique customer identifier |
| Age            | Age of the customer        |
| Spending_Score | Customer spending score    |
CustomerID  Age  Spending_Score
1           19   39
2           21   81
3           20   6
4           23   77
5           31   40
The analysis uses Age and Spending_Score as the clustering features.
## Technologies Used
Python
Pandas
Matplotlib
Scikit-learn
StandardScaler
K-Means clustering
Google Colab
## Project Workflow
1.Import the required Python libraries.
2. Load the customer dataset.
3.Inspect the first and last rows.
4.Select age and spending score as features.
5.Standardize the feature values.
6.Test different numbers of clusters.
7.Calculate WCSS for each value of K
8.Use the Elbow Method to identify a suitable number of clusters.
## Data Preprocessing
Because age and spending score may have different scales, the features are standardized before clustering:
from sklearn.preprocessing import StandardScaler

X = df[["Age", "Spending_Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Standardization helps ensure that one feature does not dominate the clustering process simply because of its numerical scale.

## K-Means Clustering
K-Means clustering groups observations by assigning each customer to the nearest cluster center.
The project evaluates cluster counts from 1 through 10:
from sklearn.cluster import KMeans

wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

    ## Elbow Method
The Within-Cluster Sum of Squares, or WCSS, is plotted against the number of clusters:
import matplotlib.pyplot as plt

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

The resulting graph shows a sharp reduction in WCSS at the beginning, followed by smaller improvements as more clusters are added. The elbow point provides a practical choice for the number of customer segments
## Why Customer Segmentation Matters
1.Customer segmentation can help businesses:
2.Identify high-spending customers.
3.Find younger customers with strong purchasing behavior.
4.Detect low-spending customer groups.
5.Create personalized marketing campaigns.
6.Improve customer retention strategies.
7.Allocate advertising budgets more effectively.
For example, a business could create separate campaigns for customers with high spending scores and customers who may need stronger incentives to purchase.
## Example Clustering Code
After selecting the desired number of clusters, the final K-Means model can be created as follows:
optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df.head())

## Possible Customer Segments
Depending on the selected value of K, the clusters may represent groups such as:
Young customers with high spending scores.
Young customers with low spending scores.
Older customers with moderate spending behavior.
High-value customers.
Customers with limited engagement.
The exact meaning of each cluster should be determined by examining the average age and spending score within each group.
## Future Improvements
Plot the final customer clusters on a scatter plot.
Add cluster centers to the visualization.
Calculate average age and spending score for every cluster.
Use silhouette analysis to validate the cluster count.
Include annual income as an additional feature.
Build a customer segmentation dashboard.
Export the labeled dataset for marketing analysis.
Test alternative clustering methods such as DBSCAN or hierarchical clustering.
## Conclusion
This project demonstrates how unsupervised machine learning can be used to segment customers based on age and spending behavior. K-Means clustering and the Elbow Method provide a practical foundation for discovering customer groups and supporting data-driven marketing decisions.

