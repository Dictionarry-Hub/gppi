import sys
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from kneed import KneeLocator
from typing import Dict, Optional, Tuple

def evaluate_k(data: np.ndarray, k: int) -> Tuple[float, float, float]:
    """
    Evaluate a specific k using multiple metrics
    Returns: (silhouette_score, calinski_harabasz_score, inertia)
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Silhouette score: ranges from -1 to 1, higher is better
    sil_score = silhouette_score(data, labels) if k > 1 else -1
    
    # Calinski-Harabasz index: higher is better
    ch_score = calinski_harabasz_score(data, labels)
    
    return sil_score, ch_score, kmeans.inertia_

def optimize_k(data: np.ndarray, max_k: int = 10) -> int:
    """
    Find optimal k using multiple methods:
    1. Silhouette Method
    2. Calinski-Harabasz Index
    3. Elbow Method
    """
    # Ensure max_k isn't larger than the dataset
    max_k = min(max_k, len(data) - 1)
    k_range = range(2, max_k + 1)  # Start from 2 as we need at least 2 clusters
    
    scores = {
        'silhouette': [],
        'calinski_harabasz': [],
        'inertia': []
    }
    
    # Collect scores for each k
    for k in k_range:
        sil, ch, inertia = evaluate_k(data, k)
        scores['silhouette'].append(sil)
        scores['calinski_harabasz'].append(ch)
        scores['inertia'].append(inertia)
    
    # 1. Silhouette Method (highest score)
    k_sil = k_range[np.argmax(scores['silhouette'])]
    
    # 2. Calinski-Harabasz Method (highest score)
    k_ch = k_range[np.argmax(scores['calinski_harabasz'])]
    
    # 3. Elbow Method
    kneedle = KneeLocator(
        list(k_range),
        scores['inertia'],
        S=1.0,
        curve='convex',
        direction='decreasing'
    )
    k_elbow = kneedle.elbow if kneedle.elbow else k_range[0]
    
    # Voting system for final k
    k_votes = [k_sil, k_ch, k_elbow]
    k_optimal = int(np.median(k_votes))
    
    # Print evaluation metrics
    print(f"Evaluation Metrics (stderr):", file=sys.stderr)
    print(f"Silhouette Analysis suggests k={k_sil}", file=sys.stderr)
    print(f"Calinski-Harabasz Index suggests k={k_ch}", file=sys.stderr)
    print(f"Elbow Method suggests k={k_elbow}", file=sys.stderr)
    print(f"Final chosen k={k_optimal}", file=sys.stderr)
    
    return k_optimal

def cluster_dictionary(data: Dict, k: Optional[int] = None) -> Dict:
    """
    Group dictionary values using k-means clustering.
    If k is not provided, finds optimal k using multiple methods.
    """
    # Convert values to numpy array for clustering
    values = np.array(list(data.values())).reshape(-1, 1)
    keys = list(data.keys())
    
    # Scale the data
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)
    
    # Determine k if not provided
    if k is None:
        k = optimize_k(scaled_values)
        
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_values)
    
    # Calculate quality metrics for final clustering
    sil_score, ch_score, inertia = evaluate_k(scaled_values, k)
    
    # Group results
    groups = {}
    for i in range(k):
        # Get items in this cluster
        mask = labels == i
        cluster_values = values[mask].flatten()
        cluster_keys = np.array(keys)[mask]
        
        groups[f"group_{i}"] = {
            "items": dict(zip(cluster_keys, cluster_values)),
            "average": float(np.mean(cluster_values)),
            "size": int(sum(mask))
        }
    
    # Add clustering quality metrics
    metadata = {
        "metrics": {
            "silhouette_score": float(sil_score),
            "calinski_harabasz_score": float(ch_score),
            "inertia": float(inertia),
            "k": k
        }
    }
    
    # Sort groups by average value (highest first)
    sorted_groups = dict(sorted(
        groups.items(),
        key=lambda x: x[1]["average"],
        reverse=True
    ))
    
    return {"metadata": metadata, "groups": sorted_groups}

def main():
    try:
        # Read input dictionary from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Get k from command line if provided
        k = int(sys.argv[1]) if len(sys.argv) > 1 else None
        
        # Validate input
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
        if not all(isinstance(v, (int, float)) for v in input_data.values()):
            raise ValueError("All values must be numeric")
        
        # Perform clustering
        results = cluster_dictionary(input_data, k)
        
        # Output results as JSON
        print(json.dumps(results, indent=2))
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON input", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()