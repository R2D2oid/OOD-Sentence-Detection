# Out-of-domain Sentence Detection
Detects out-of-domain (OOD) sentences.

### Python environment
```
virtualenv --system-site-packages -p python3 env_OOD
source env_OOD/bin/activate
pip install -r requirements.txt
```
### Command to run
```
python3 main.py --configpath config.cfg
```

### Experiment Configurations
All configs are loaded from `config.cfg` 