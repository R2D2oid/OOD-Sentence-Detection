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
			f.write(item+'\n')
	return True

def sentence_contains_term(sentence, terms):
	for t in terms:
		if t in sentence:
			return 1
	return 0
