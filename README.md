# mall-customer-segmentation
mall-customer-segmentation
# Mall Customer Segmentation Using Unsupervised Learning

## Project Overview

This project applies unsupervised learning techniques to segment mall customers based on their annual income and spending behavior. The goal is to identify distinct customer groups that can inform targeted marketing strategies and business decisions.

**Dataset**: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (200 customers, 5 features)

## Key Findings

Five distinct customer segments were identified:

| Segment | Size | Income | Spending | Description |
|---------|------|--------|----------|-------------|
| Mainstream | 40.5% | ~55k | ~50 | Average customers, largest group |
| VIP | 19.5% | ~87k | ~82 | High-value customers, priority for retention |
| Impulsive | 11% | ~26k | ~79 | Low income but high spending, monitor for risk |
| Conservative | 17.5% | ~88k | ~17 | Wealthy but low spending, growth opportunity |
| Budget-Conscious | 11.5% | ~26k | ~21 | Price-sensitive, target with discounts |

## Methods Compared

| Model | Clusters | Silhouette Score | Notes |
|-------|----------|------------------|-------|
| **K-Means** | 5 | 0.5547 | ✓ Selected - simple and effective |
| DBSCAN | 4 | 0.6452 | 36% noise points, unsuitable |
| GMM | 5 | 0.5537 | Soft clustering with probabilities |
| Hierarchical | 5 | 0.5538 | Dendrogram visualization |

## Project Structure
```
├── mall-customer-segmentation.ipynb    # Main analysis notebook
├── README.md                           # Project documentation
└── images/                             # Visualization outputs
    ├── elbow_method.png
    ├── silhouette_scores.png
    ├── kmeans_clusters.png
    ├── gmm_confidence.png
    ├── dendrogram.png
    └── business_segments.png
```

## Methodology

1. **Exploratory Data Analysis**
   - Distribution analysis (age, income, spending score)
   - Correlation matrix revealing weak feature correlations
   - Scatter plot showing natural cluster patterns

2. **Preprocessing**
   - Feature selection: Annual Income + Spending Score
   - StandardScaler normalization for distance-based algorithms

3. **Model Development**
   - K-Means: Elbow method + Silhouette validation → K=5
   - DBSCAN: Grid search over eps and min_samples
   - GMM: BIC/AIC model selection → n=5
   - Hierarchical: Ward linkage with dendrogram analysis

4. **Evaluation & Comparison**
   - Silhouette score comparison across methods
   - Visual inspection of cluster assignments
   - Business interpretability assessment

## Key Visualizations

### Cluster Selection Validation
- Elbow method shows clear inflection at K=5
- Silhouette score peaks at K=5 (0.5547)
- BIC score minimized at n=5 components

### GMM Soft Clustering (Differentiator)
GMM provides probability assignments for each customer, identifying boundary cases with lower confidence scores. This enables nuanced marketing strategies for customers who may respond to multiple segment approaches.

## Business Recommendations

- **VIP Segment**: Premium products, exclusive memberships, personal outreach
- **Conservative Wealthy**: Value demonstration, quality emphasis, content marketing
- **Impulsive Buyers**: Installment plans, budget tools, responsible spending education
- **Mainstream**: Standard promotions, loyalty programs, seasonal campaigns
- **Budget-Conscious**: Discounts, clearance sales, price matching

## Technologies Used

- Python 3.x
- pandas, numpy
- scikit-learn (KMeans, DBSCAN, GaussianMixture, AgglomerativeClustering)
- matplotlib, seaborn
- scipy (dendrogram, linkage)

## How to Run

1. Clone this repository
2. Open the notebook in Kaggle or Jupyter
3. If running locally, install dependencies:
```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy
```
4. Download the dataset from Kaggle and update the file path

## Author

Qihang Zhang

MSAI Program, University of Colorado Boulder

## License

MIT License
