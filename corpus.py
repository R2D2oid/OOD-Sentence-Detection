from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist
import utilities as utils
from nltk import tokenize

class Corpus:
	def __init__(self, textfiles_dir, freqdist_path = None):
		self.corpus = self.get_corpus_from_textfiles(textfiles_dir)
		self.words = self.corpus.words()
		self.sents = self.corpus.sents()
		self.sents_count = len(self.sents)
		self.words_count = len(self.words)
		self.freqdist = FreqDist(self.words) if freqdist_path is None else utils.load_picklefile(freqdist_path)
		self.batch_size = 10000

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

	def get_least_frequent_words(self, count_less_than = 4, min_word_len = 5):
		least_freq_words = [] 
		for w in self.freqdist:
			if self.freqdist[w] < count_less_than and len(w) >= min_word_len:
				least_freq_words.append(w)
		return least_freq_words

	def get_sentence_encodings(self, model_name = 'universal-sentence-encoder', output_dir = None):
		if model_name == 'universal-sentence-encoder':
			import tensorflow as tf
			import tensorflow_hub as hub
			model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
		elif model_name == 'fasttext':
			import sister
			model = sister.MeanEmbedding(lang="en")
	
		## write embeddings to output_dir using smaller files in consideration for memory usage
		if output_dir:				
			for i in range(0,self.sents_count, self.batch_size):
				path_ = '{}/sentence_encoding_batch_{}.pkl'.format(output_dir, int(i/self.batch_size))
				print('extracting embeddings ', path_)
				sentence_encoding_batch = [(' '.join(s), model([' '.join(s)])) for s in self.sents[i:i+self.batch_size]]
				utils.dump_picklefile(sentence_encoding_batch, path_)
			
			## load embeddings into memory
			if self.sents_count > 100*self.batch_size:
				print('The embeddings are too large to be loaded into memory')
				print('A smaller subset is loaded and returned')
				return sentence_encoding_batch
			else:
				return utils.load_precomputed_embeddings(output_dir)

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

def split_sentence_on_punctuations(line):
	puncts = ['.', '?', '!', ',', ';', ':']
	for p in puncts:
		line = line.replace(p, '.')
	sents = line.split('.')
	sents = [s.replace(')', '').replace('(', '').replace('*', '').strip() for s in sents]
	sents = [s for s in sents if len(s)>3]
	return sents

def replace_infrequent_words_with_UNK(sentence_dir, sentence_clean_dir, infreq_words, proper_nouns = None):
	'''
	Reads sentence files for each game and replaces infrequent words and/or proper nouns with UNK token
	Then writes back the clean sentence files
	Input:
		sentence_dir: input directory to of raw sentences
		sentence_clean_dir: output directory of clean sentences 
	Output
		dumps clean sentences to sentence_clean_dir
	'''
	filenames = utils.get_filenames(sentence_dir)
	filepaths = ['{}/{}'.format(sentence_dir, f) for f in filenames] 

	filter_words = set(infreq_words)

	for name_, path_ in zip(filenames, filepaths):
		print('processing senteces from ', path_)
		sentences = utils.load_textfile(path_)
		filtered_sentences = []
		for s in sentences:
			filtered_sentences.append(utils.replace_words_in_sentence(s, filter_words))
		path_ = '{}/{}'.format(sentence_clean_dir, name_)
		utils.dump_textfile(filtered_sentences, path_)

def convert_MPC_corpus_to_sentences(subtitle_dir, sentence_dir, emoticons_path):
	'''
	Reads MPC chat text files and cleans them into sentences
	Then writes them to text files
	Input:
		subtitle_dir: directory containing subtitle for each game
		sentence_dir: target directory to write output sentences
	Output
		dumps sentences for each subtitle file to a textfile at sentence_dir
	'''
	filenames = utils.get_filenames(subtitle_dir)
	filepaths = ['{}/{}'.format(subtitle_dir, f) for f in filenames] 

	emoticons = utils.load_textfile(emoticons_path)

	for name_, path_ in zip(filenames, filepaths):
		print('processing textfiles at ', path_)
		lines = utils.load_textfile(path_)
		sentences = []
		for l in lines:
			ll = l.split('):  ')[-1]
			ll = document_to_sentences(ll)
			ss = []
			for l in ll:
				ss.extend(split_sentence_on_punctuations(l))
			ss = [utils.replace_words_in_sentence(s, emoticons, replacement_token = '', emoticons = True) for s in ss]
			sentences.extend(ss)

		sentences = [s+'.' for s in sentences]
		path_ = '{}/clean_{}'.format(sentence_dir, name_)
		utils.dump_textfile(sentences, path_)

	return True

