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

### Experiment Configurations
All configs are loaded from `config.cfg` 

### visualizing corpora
```
python3 vis_compare_corpora.py --corpora_dirs=/path/to/corpus_embeddings1,/path/to/corpus_embeddings2 --corpora_names=Corpus1,Corpus2 --cap_size=1000
```