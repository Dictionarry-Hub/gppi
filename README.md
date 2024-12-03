# GPPI

Code for Golden Popcorn Performance Index algorithm + K-Means Clustering Optimisation

## Clustering

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python cluster.py < input.json        # auto-optimize k
python cluster.py 3 < input.json      # specify k=3
```

Input format (`input.json`):

```json
{
  "key1": 1,
  "key2": 2,
  "key3": 3
}
```
