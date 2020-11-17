# Out-of-domain Sentence Detection
Detects out-of-domain (OOD) sentences.

### Python environment
```
virtualenv --system-site-packages -p python3 env_OOD
source env_OOD/bin/activate
pip install -r requirements.txt
```
### Extract sentences from subtitles and visualize embeddings of the corpus
```
python3 main.py --extract_sentences True
```

