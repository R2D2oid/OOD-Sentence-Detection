from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist
import utilities as utils
from nltk import tokenize

class Corpus:
	def __init__(self, textfiles_dir, freqdist_path = None):
		self.corpus = self.get_corpus_from_textfiles(textfiles_dir)
		self.words = self.corpus.words()
		self.sents = self.corpus.sents()
		self.freqdist = FreqDist(self.words) if freqdist_path is None else utils.load_picklefile(freqdist_path)

	def get_word_frequency_distribution(self):
		return self.freqdist

	def get_sorted_word_frequency_distribution(self):
		return sorted(self.freqdist , key = self.freqdist.__getitem__, reverse = True) 

	def dump_word_frequency_distribution(self, path_):
		return utils.dump_picklefile(self.get_word_frequency_distribution(), path_)
		
	def plot_normalized_word_frequency_distribution(self):
		freqdist = self.get_word_frequency_distribution()
		dist = freqdist.copy()
		count = dist.N()
		for w in dist:
			dist[w] = dist[w]/count
		dist.plot()

	def get_least_frequent_words(self, count_less_than = 4):
		least_freq_words = [] 
		for w in self.freqdist:
			if self.freqdist[w] < 4:
				least_freq_words.append(w)
		return least_freq_words

	def get_sentence_encodings(self, model_name = 'universal-sentence-encoder'):
		if model_name == 'universal-sentence-encoder':
			import tensorflow as tf
			import tensorflow_hub as hub

			model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

		elif model_name == 'fasttext':
			import sister

			model = sister.MeanEmbedding(lang="en")
	
		self.sentence_encodings  = [(' '.join(s), model([' '.join(s)])) for s in self.sents]
		
		return self.sentence_encodings 

	@staticmethod
	def get_corpus_from_textfiles(textfiles_dir):
		filenames = utils.get_filenames(textfiles_dir)
		return PlaintextCorpusReader(textfiles_dir, filenames)

	@staticmethod
	def sort_dict(dct):
		return sorted(dct.items(), key=lambda x: x[1], reverse=True)


def convert_subtitles_to_sentences(subtitle_dir, sentence_dir):
	'''
	Reads subtitle file for each game and breaks them into sentences
	Then writes them each separately to a text file
	Input:
		subtitle_dir: directory containing subtitle for each game
		sentence_dir: target directory to write output sentences
	Output
		dumps sentences for each subtitle file to a textfile at sentence_dir
	'''
	filenames = utils.get_filenames(subtitle_dir)
	filepaths = ['{}/{}'.format(subtitle_dir, f) for f in filenames] 

	for name_, path_ in zip(filenames, filepaths):
		print('processing subtitles at ', path_)
		doc = load_subtitles(path_)
		sentences = document_to_sentences(doc)		

		path_ = '{}/{}.txt'.format(sentence_dir, name_.split('.')[0])
		utils.dump_textfile(sentences, path_)

	return True

def load_subtitles(path_):
	subs = utils.load_picklefile(path_)
	doc = ''
	for v in subs.values():
		doc += ' {}'.format(v)
	doc = doc.lower()
	return doc

def document_to_sentences(doc_):
	sentences = tokenize.sent_tokenize(doc_)
	sentences = [s.strip() for s in sentences]
	return sentences