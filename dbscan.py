#!/usr/bin/env python3

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path


def load_gppi_data(input_file):
    """Load GPPI data from a JSON file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: Input JSON file not found: {input_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {input_file}")
        exit(1)


def apply_dbscan(data, eps_factor=0.12, min_samples=1):
    """
    Apply DBSCAN clustering to GPPI data.
    
    Args:
        data: Dictionary of group names to GPPI values
        eps_factor: Factor to multiply by data range to get eps value
        min_samples: Minimum points in a neighborhood to form a core point
        
    Returns:
        DataFrame with groups, GPPI values, and cluster assignments
    """
    groups = list(data.keys())
    gppi_values = list(data.values())

    # Prepare data for DBSCAN
    X = np.array(gppi_values).reshape(-1, 1)

    # Calculate eps as a fraction of the data range
    data_range = max(gppi_values) - min(gppi_values)
    eps_value = data_range * eps_factor

    # Apply DBSCAN
    clustering = DBSCAN(eps=eps_value, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    # Count clusters (excluding noise points with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Create DataFrame with results
    df = pd.DataFrame({
        'group': groups,
        'gppi': gppi_values,
        'cluster': labels
    })

    # Handle any noise points (-1 label) - assign to lowest tier
    if -1 in df['cluster'].values:
        min_non_noise_cluster = df[df['cluster'] != -1]['cluster'].min()
        df.loc[df['cluster'] == -1, 'cluster'] = min_non_noise_cluster - 1

    # Convert to 1-indexed tiers
    df['tier'] = df['cluster'] - df['cluster'].min() + 1

    # Sort by GPPI within each tier
    df = df.sort_values(['tier', 'gppi'], ascending=[True, False])

    return df, n_clusters, eps_value


def optimize_eps(data, min_clusters=3, max_clusters=8):
    """
    Find optimal eps value to get a reasonable number of clusters.
    
    Args:
        data: Dictionary of group names to GPPI values
        min_clusters: Minimum desired number of clusters
        max_clusters: Maximum desired number of clusters
        
    Returns:
        Optimal eps factor and resulting number of clusters
    """
    gppi_values = list(data.values())
    X = np.array(gppi_values).reshape(-1, 1)
    data_range = max(gppi_values) - min(gppi_values)

    # Try different eps factors
    eps_factors = np.linspace(0.05, 0.3,
                              26)  # Try values from 5% to 30% of range

    for eps_factor in eps_factors:
        eps_value = data_range * eps_factor
        clustering = DBSCAN(eps=eps_value, min_samples=1).fit(X)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if min_clusters <= n_clusters <= max_clusters:
            return eps_factor, n_clusters

    # If we couldn't find an optimal value, return a reasonable default
    default_eps = 0.12
    clustering = DBSCAN(eps=data_range * default_eps, min_samples=1).fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return default_eps, n_clusters


def format_output(df, resolution, group_type, eps_value, args):
    """
    Format the clustering results into the required JSON structure.
    
    Args:
        df: DataFrame with clustering results
        resolution: Resolution string (e.g., "1080p")
        group_type: Group type string (e.g., "Quality" or "Efficient")
        eps_value: Epsilon value used for DBSCAN
        args: Command line arguments
        
    Returns:
        Dictionary with formatted results
    """
    # Prepare tiered_groups array
    tiered_groups = []
    for _, row in df.iterrows():
        tiered_groups.append({"name": row['group'], "tier": int(row['tier'])})

    # Create metadata
    metadata = {
        "total_groups": len(df),
        "total_tiers": df['tier'].nunique(),
        "resolution": resolution,
        "type": group_type,
        "algorithm": "DBSCAN",
        "eps_value": eps_value,
        "eps_factor": args.eps,
        "min_samples": args.min_samples
    }

    # Add tier statistics
    tier_stats = {}
    for tier in sorted(df['tier'].unique()):
        tier_data = df[df['tier'] == tier]
        tier_stats[f"tier_{tier}"] = {
            "count": len(tier_data),
            "min_gppi": float(tier_data['gppi'].min()),
            "max_gppi": float(tier_data['gppi'].max()),
            "avg_gppi": float(tier_data['gppi'].mean())
        }

    # Combine everything
    result = {
        "metadata": metadata,
        "tier_statistics": tier_stats,
        "tiered_groups": tiered_groups
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Cluster release groups using DBSCAN based on GPPI values')
    parser.add_argument('input_file', help='Input JSON file with GPPI values')
    parser.add_argument('--resolution',
                        choices=['SD', '720p', '1080p', '2160p'],
                        required=True,
                        help='Resolution for the output')
    parser.add_argument('--type',
                        choices=['Quality', 'Efficient'],
                        required=True,
                        help='Type of release groups')
    parser.add_argument('--output-dir',
                        default='.',
                        help='Directory for output JSON file')
    parser.add_argument(
        '--eps',
        type=float,
        default=None,
        help='Epsilon factor (as proportion of data range) for DBSCAN')
    parser.add_argument('--min-samples',
                        type=int,
                        default=1,
                        help='Minimum samples parameter for DBSCAN')
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Automatically optimize epsilon to get 3-8 clusters')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print detailed information about clusters')

    args = parser.parse_args()

    # Load data
    data = load_gppi_data(args.input_file)

    # Determine the eps value to use
    if args.optimize:
        eps_factor, n_clusters = optimize_eps(data)
        print(
            f"Optimized eps factor: {eps_factor:.4f} (yields {n_clusters} tiers)"
        )
    elif args.eps is not None:
        eps_factor = args.eps
    else:
        eps_factor = 0.12  # Default

    # Apply DBSCAN
    df, n_clusters, eps_value = apply_dbscan(data, eps_factor,
                                             args.min_samples)

    # Prepare output file path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.resolution} {args.type}.json"

    # Format results
    result = format_output(df, args.resolution, args.type, eps_value, args)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Found {n_clusters} natural tiers among {len(data)} release groups")
    print(f"Results saved to: {output_file}")

    # Print verbose output if requested
    if args.verbose:
        print("\nTier details:")
        for tier in sorted(df['tier'].unique()):
            tier_data = df[df['tier'] == tier]
            print(f"\nTier {tier} ({len(tier_data)} groups):")
            print(
                f"  GPPI Range: {tier_data['gppi'].min():.2f} to {tier_data['gppi'].max():.2f}"
            )

            # Format groups into rows of approximately 70 chars each
            groups = tier_data['group'].tolist()
            group_text = ""
            line = "  Groups: "

            for group in groups:
                if len(line + group + ", ") > 80:
                    group_text += line + "\n"
                    line = "          " + group + ", "
                else:
                    line += group + ", "

            if line.endswith(", "):
                line = line[:-2]

            group_text += line
            print(group_text)


if __name__ == "__main__":
    main()
