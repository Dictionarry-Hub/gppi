# GPPI Clustering Tool

This tool uses the DBSCAN algorithm to automatically cluster release groups into tiers based on their Golden Popcorn Performance Index (GPPI) values. It generates a structured JSON output that can be used with Radarr custom format creation scripts.

## Overview

The Golden Popcorn Performance Index (GPPI) measures how likely a release group is to produce high-quality "Golden Popcorn" encodes. This script:

1. Takes a JSON file containing release group GPPI values
2. Uses DBSCAN to find natural clusters/tiers in the data
3. Outputs a JSON file with tiered group assignments

No manual assignment of tiers or predefined number of clusters is required - the algorithm discovers the natural tiers in your data.

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - pandas
  - scikit-learn

Install dependencies with:

```
pip install numpy pandas scikit-learn
```

## Usage

### Basic Usage

```bash
./dbscan.py input_file.json --resolution 1080p --type Quality
```

This will:

1. Read GPPI values from `input_file.json`
2. Cluster the groups using DBSCAN with default parameters
3. Save the results to `1080p Quality.json`

### Input Format

The input file should be a JSON object with release group names as keys and their GPPI values as numeric values:

```json
{
  "EbP": 412.45,
  "DON": 350.55,
  "HiDt": 227.22,
  ...
}
```

### Output Format

The output file contains:

1. Metadata about the clustering
2. Statistics for each tier
3. A `tiered_groups` array with objects containing `name` and `tier` properties

```json
{
  "metadata": {
    "total_groups": 37,
    "total_tiers": 5,
    "resolution": "1080p",
    "type": "Quality",
    "algorithm": "DBSCAN",
    "eps_value": 49.37,
    "eps_factor": 0.12,
    "min_samples": 1
  },
  "tier_statistics": {
    "tier_1": {
      "count": 2,
      "min_gppi": 350.55,
      "max_gppi": 412.45,
      "avg_gppi": 381.5
    },
    ...
  },
  "tiered_groups": [
    {
      "name": "EbP",
      "tier": 1
    },
    {
      "name": "DON",
      "tier": 1
    },
    ...
  ]
}
```

## Command Line Options

```
positional arguments:
  input_file            Input JSON file with GPPI values

required arguments:
  --resolution {SD,720p,1080p,2160p}
                        Resolution for the output
  --type {Quality,Efficient}
                        Type of release groups

optional arguments:
  --output-dir OUTPUT_DIR
                        Directory for output JSON file
  --eps EPS             Epsilon factor (as proportion of data range) for DBSCAN
  --min-samples MIN_SAMPLES
                        Minimum samples parameter for DBSCAN
  --optimize            Automatically optimize epsilon to get 3-8 clusters
  --verbose             Print detailed information about clusters
```

## Advanced Usage

### Automatic Optimization

To automatically find the best epsilon value that gives between 3-8 clusters:

```bash
./dbscan.py input.json --resolution 1080p --type Quality --optimize --verbose
```

### Manual Epsilon Control

You can control the clustering sensitivity with the `--eps` parameter:

```bash
./dbscan.py input.json --resolution 1080p --type Quality --eps 0.15
```

The epsilon value is specified as a proportion of the data range:

- Lower values (0.05-0.10): More tiers, finer granularity
- Higher values (0.15-0.25): Fewer tiers, broader categories

### Verbose Output

Add the `--verbose` flag to see detailed information about the discovered tiers:

```bash
./dbscan.py input.json --resolution 1080p --type Quality --verbose
```

## Tips for Getting Good Clusters

1. Start with `--optimize --verbose` to see what the automatic optimization discovers
2. If you want more tiers, use a smaller epsilon (e.g., `--eps 0.08`)
3. If you want fewer tiers, use a larger epsilon (e.g., `--eps 0.2`)
4. Look at the GPPI distribution to understand natural groupings in your data

## Integration with Custom Format Scripts

The output of this script is designed to work directly with Radarr custom format creation scripts. The JSON structure provides a `tiered_groups` array that can be used to create custom formats for different quality tiers.

## Example Workflow

1. Generate GPPI values for release groups at a specific resolution
2. Run this script to automatically cluster them into tiers
3. Use the resulting JSON with your custom format creation script
4. Import the custom formats into Radarr

This approach ensures that your quality tiers are based on natural groupings in the GPPI data rather than arbitrary assignments.
