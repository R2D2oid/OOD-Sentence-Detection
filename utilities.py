import pickle as pkl
from os import listdir, path, makedirs

def get_filenames(dir_):
	return listdir(dir_)

def get_filepaths(dir_):
	return ['{}/{}'.format(dir_, f) for f in get_filenames(dir_)]

def create_dir_if_not_exist(path_):
	if not path.exists(path_):
		makedirs(path_)

def load_picklefile(path_):
	with open(path_, 'rb') as f:
		return pkl.load(f)

def dump_picklefile(object_, path_):
	with open(path_, 'wb') as f:
		pkl.dump(object_, f, protocol = 3)
	return True

def load_textfile(path_):
	with open(path_, 'r') as f:
		lines = f.readlines()
	return [l.strip() for l in lines]

def dump_textfile(data, path_):
	with open(path_, 'w') as f:
		for item in data:
			if item is not None:
				f.write(item+'\n')
	return True

def load_precomputed_embeddings(dir_):
	filepaths = get_filepaths(dir_)
	precomputed_embeddings = []
	for path_ in filepaths:
		print('loading {}'.format(path_))
		sentence_embeddings = load_picklefile(path_)
		precomputed_embeddings.extend(sentence_embeddings)
	return precomputed_embeddings

def sentence_contains_term(sentence, terms):
	for t in terms:
		if t in sentence:
			return 1
	return 0

def replace_words_in_sentence(s, words, replacement_token = 'UNK', emoticons = False):
	if emoticons:
		for w in words:
			s = s.replace(w, replacement_token)
		return s
	else:	
		for w in words:
			s = s.replace(' '+w+' ', ' {} '.format(replacement_token))
			if s.startswith(w+' '):
				idx = len(w) + 1
				s = '{} {}'.format(replacement_token, s[idx:])
		return s	
