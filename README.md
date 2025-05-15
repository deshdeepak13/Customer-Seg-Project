
# 🧠 Intelligent Customer Segmentation System

A cutting-edge, full-stack solution for segmenting customers using unsupervised machine learning, visualizing insights, and serving real-time personalized offers via a REST API. This project demonstrates end-to-end problem solving—from data exploration through model deployment—designed for production-grade performance and maintainability.

---

## 🎯 Problem Statement

Traditional customer grouping relies on static rules (e.g., demographic buckets) that miss nuanced behaviors. We need a dynamic, data-driven way to:

1. **Identify distinct customer segments** based on spending, engagement, and profile attributes.
2. **Automate real-time segment inference** for new users.
3. **Deliver tailored offers/discounts** that maximize engagement and conversion.

---

## 🚀 High-Level Solution

1. **Exploratory Data Analysis**  
   - Inspect distributions, correlations, and outliers across numeric features (age, income, spending score, etc.).  
   - Visualize categorical breakdowns (gender, preferred category) to understand base rates.

2. **Preprocessing Pipeline**  
   - **Deduplication** & missing-value handling.  
   - **Standard Scaling** of numeric features to zero mean and unit variance.  
   - **One-Hot Encoding** for categorical fields.

3. **Dimensionality Reduction**  
   - Use **PCA** to project high-dimensional feature space into two principal components for visualization and noise reduction.

4. **Unsupervised Clustering**  
   - Apply **KMeans**:
     - Use **Elbow Method** (within-cluster sum of squares) and **Silhouette Score** to choose optimal `k`.  
     - Train final model to discover 3 meaningful segments.

5. **Segment Profiling & Business Rules**  
   - Analyze cluster centers and box plots to label segments (e.g., “High Spender,” “Value Seeker,” “Loyal Regular”).  
   - Map each segment to a curated set of offers.

6. **Model Deployment with FastAPI**  
   - Serialize `StandardScaler`, `PCA`, and `KMeans` objects with `joblib`.  
   - Expose a `/predict` endpoint that ingests raw user data, runs the preprocessing pipeline, and returns segment + offers in JSON.

7. **Interactive Frontend (React)**  
   - Build a responsive form that collects user inputs.  
   - Submit via Axios to the FastAPI API.  
   - Display real-time segment and personalized offers.

---

## 📂 Repository Structure

```

.
├── app_api/
│   ├── app/                          # FastAPI application module
│   │   ├── main.py                   # API entry point with clustering logic
│   │   └── model/
│   │       ├── scaler.pkl            # Pre-fitted StandardScaler for input normalization
│   │       ├── pca.pkl               # PCA transformer for dimensionality reduction
│   │       ├── kmeans_model.pkl      # Trained KMeans clustering model
│   │       └── model_columns.json    # Column order used during training
│   ├── requirements.txt              # Python dependencies for the API
│   └── Dockerfile                    # Containerization setup for deployment
│
├── customer_segmentation_data.csv/     # Dataset 
|
|
├── main.ipynb/     # Python Notebook 
│   
│
└── README.md       # Comprehensive documentation & setup instructions

````

---

## 🔬 Code Analysis & Key Snippets

### 1. Data Scaling & PCA  
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fit on training data
scaler = StandardScaler().fit(X_numeric)
X_scaled = scaler.transform(X_numeric)

pca = PCA(n_components=2, random_state=42).fit(X_scaled)
X_pca = pca.transform(X_scaled)
````

* **Insight**: Scaling ensures features with different units don’t dominate clustering.
* **PCA** reduces noise and enables 2D visualizations.

### 2. KMeans Clustering & Validation

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Determine optimal k
inertias, sil_scores = [], []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42).fit(X_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, km.labels_))
```

* **Elbow vs. Silhouette**: Combine both to pick a stable cluster count.

### 3. FastAPI Predict Endpoint

```python
@app.post("/predict")
def predict(data: UserData):
    df = pd.DataFrame([data.dict()])
    # One-hot encode & reindex
    df = pd.get_dummies(df).reindex(columns=model_columns, fill_value=0)
    # Pipeline
    scaled = scaler.transform(df)
    reduced = pca.transform(scaled)
    cluster = kmeans.predict(reduced)[0]
    return {"segment": segment_names[cluster], "offers": offers_map[cluster]}
```

* **Reindexing** protects against missing categories.
* **Serialization** via `joblib` keeps load times minimal.

---

## 📈 Results & Insights

* **Segment A (High Spender)**:

  * High income & spending score, frequent purchases.
  * Action: Offer premium bundles and express shipping.

* **Segment B (Value Seeker)**:

  * Moderate income, low spending score.
  * Action: Promote discounts and loyalty rewards.

* **Segment C (Loyal Regular)**:

  * Long membership, steady purchase frequency.
  * Action: Upsell and referral incentives.

---

## 📖 Usage

1. **Backend**

   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
2. **Frontend**

   ```bash
   cd frontend
   npm install
   npm start
   ```
3. **Test**

   * Navigate to `http://localhost:3000`
   * Fill in form and submit — see your segment & offers!

---

## 🌟 Competitive Edge

* **End-to-end reproducibility**: notebooks, model artifacts, API, and UI—all versioned.
* **Robust validation**: Elbow, Silhouette, and PCA checks guard against overfitting.
* **Production-ready**: FastAPI backend with solid input validation and serialization.
* **Extensibility**: Easily swap in new models (e.g., DBSCAN, hierarchical) or add real-time streaming.

---

## 🔭 Future Enhancements

* Integrate real-time user behavior data (clickstream) for dynamic re-segmentation.
* Add a **recommendation engine** (collaborative filtering) for product-level suggestions.
* Deploy using Kubernetes & CI/CD pipelines for auto-scaling.

---

### **Built with passion by**:

- *Desh Deepak Verma* (U22EC028)  
- *Krishna Meghwal* (U22EC042)  
- *Ajay Parmar* (U22EC038)  

— Ready to drive data-driven marketing strategies and elevate customer engagement!
