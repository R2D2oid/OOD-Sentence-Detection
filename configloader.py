import configparser

def configloader (configpath):
	''' loads the config file from the provided path'''
	cp = configparser.RawConfigParser()
	cp.read(configpath)
	config = {}

	section = {}
	tag = 'var'
	section['model_name'] = cp.get(tag, 'model_name')
	section['terms_path'] = cp.get(tag, 'terms_path')
	section['dataset_name'] = cp.get(tag, 'dataset_name')
	section['subtitle_dir'] = cp.get(tag, 'subtitle_dir')
	section['sentence_dir'] = cp.get(tag, 'sentence_dir')
	section['artifact_dir'] = cp.get(tag, 'artifact_dir')
	section['embedding_dir'] = cp.get(tag, 'embedding_dir')
	section['clean_sentence_dir'] = cp.get(tag, 'clean_sentence_dir')
	section['emoticons_path'] = cp.get(tag, 'emoticons_path')
	section['extract_sentences'] = cp.getboolean(tag, 'extract_sentences')
	section['clean_sentences'] = cp.getboolean(tag, 'clean_sentences')
	section['precomputed_embeddings'] = cp.getboolean(tag, 'precomputed_embeddings')
	section['pregrenerated_corpus'] = cp.getboolean(tag, 'pregrenerated_corpus')
	section['use_clean_sentences'] = cp.getboolean(tag, 'use_clean_sentences')
	section['low_memory_mode'] = cp.getboolean(tag, 'low_memory_mode')
	section['pca_dims'] = cp.getint(tag, 'pca_dims')
	
	config[tag] = section

	section = {}
	tag = 'logging'
	section['logfile'] = cp.get(tag, 'logfile')
	section['format'] = cp.get(tag, 'format')
	section['level'] = cp.getint(tag, 'level')
	config[tag] = section

	return config

def get_variables(var):
	'''
	Loads variable values from config dictionary entry and returns them
	'''
	subtitle_dir  = var['subtitle_dir']
	sentence_dir  = var['sentence_dir']
	artifact_dir  = var['artifact_dir']
	embedding_dir = var['embedding_dir']
	clean_sentence_dir  = var['clean_sentence_dir']

	dataset_name = var['dataset_name']
	model_name 	  = var['model_name']

	emoticons_path = var['emoticons_path']
	terms_path	  = var['terms_path']
	
	extract_sentences = var['extract_sentences']
	clean_sentences = var['clean_sentences']
	precomputed_embeddings = var['precomputed_embeddings']
	pregrenerated_corpus = var['pregrenerated_corpus']
	use_clean_sentences = var['use_clean_sentences']
	low_memory_mode = var['low_memory_mode']

	pca_dims = var['pca_dims']

	return dataset_name, \
			 subtitle_dir, \
			 sentence_dir, \
			 artifact_dir, \
			 terms_path, \
			 model_name, \
			 extract_sentences, \
			 clean_sentences, \
			 clean_sentence_dir, \
			 embedding_dir, \
			 emoticons_path, \
			 precomputed_embeddings, \
			 pregrenerated_corpus, \
			 use_clean_sentences, \
			 low_memory_mode, \
			 pca_dims