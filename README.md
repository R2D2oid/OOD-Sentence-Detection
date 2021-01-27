# Out-of-domain Sentence Detection
Detects out-of-domain (OOD) sentences.

### Python environment
```
virtualenv --system-site-packages -p python3 env_OOD
source env_OOD/bin/activate
pip install -r requirements.txt
```
### running the pipeline
```
python3 main.py --configpath config.cfg
```
experiment configurations are loaded from [`config.cfg`](config.cfg)

### visualizing corpora
```
python3 vis_compare_corpora.py --corpora_dirs=/path/to/corpus_embeddings1,/path/to/corpus_embeddings2 --corpora_names=Corpus1,Corpus2 --cap_size=1000
```

### clustering with locality sensitive hashing (LSH)
```
python3 lsh.py --corpora_dirs=data/04_embeddings/embeddings_NHL_raw_partial,data/04_embeddings/embeddings_MPC_raw_full --corpora_names=NHL,MPC --clustering_dir=data/05_clustering --cap_size=2000 --reduced_dim=50 --num_clusters=10
```
