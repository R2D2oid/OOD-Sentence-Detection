import argparse
import numpy as np
import utilities as utils
from corpus import Corpus, convert_subtitles_to_sentences, convert_MPC_corpus_to_sentences, replace_infrequent_words_with_UNK
from dim_reduction import pca_dim_reduction, tsne_dim_reduction
from plotting import plot_tsne
from configloader import configloader, get_variables
 
if __name__ == '__main__':

	parser = argparse.ArgumentParser ()
	parser.add_argument('--configpath', dest = 'configpath', default = 'config.cfg', help = 'config file dir')
	args = parser.parse_args()

	config = configloader(args.configpath)
	dataset_name, subtitle_dir, sentence_dir, artifact_dir, terms_path, model_name, extract_sentences, clean_sentences, clean_sentence_dir, embedding_dir, emoticons_path, precomputed_embeddings, pregrenerated_corpus, use_clean_sentences, pca_dims = get_variables(config['var'])

	utils.create_dir_if_not_exist(sentence_dir)
	utils.create_dir_if_not_exist(embedding_dir)
	utils.create_dir_if_not_exist(artifact_dir)

	## extract sentences from NHL subtitles
	if extract_sentences and dataset_name == 'NHL_Corpus': 
		convert_subtitles_to_sentences(subtitle_dir, sentence_dir)

	## extract sentences from MPC chat records
	if extract_sentences and dataset_name == 'MPC_Corpus': 
		convert_MPC_corpus_to_sentences(subtitle_dir, sentence_dir, emoticons_path)

	## clean up sentences by removing the lease frequent words 
	if clean_sentences:
		utils.create_dir_if_not_exist(clean_sentence_dir)
		corpus = Corpus(sentence_dir)
		## get infrequent words from corpus
		infrequent_words = corpus.get_least_frequent_words(count_less_than = 4)

		## replace the words with UNK token
		replace_infrequent_words_with_UNK(sentence_dir, clean_sentence_dir, infrequent_words)

		## if clean_sentences is set to True, set use_clean_sentences True as well
		sentence_dir = clean_sentence_dir

	if use_clean_sentences:
		sentence_dir = clean_sentence_dir

	## load corpus 
	if pregrenerated_corpus and not use_clean_sentences: # ensures unintentional use of unclean sentences are avoided
		## load corpus from pickle file
		corpus = utils.load_picklefile('{}/corpus.pkl'.format(artifact_dir))
	else:	
		## load sentences into corpus
		corpus = Corpus(sentence_dir)
		utils.dump_picklefile(corpus, '{}/corpus.pkl'.format(artifact_dir))

	## encode sentences 
	if precomputed_embeddings:
		sentence_encodings = utils.load_precomputed_embeddings(embedding_dir)
	else:
		if model_name == 'universal-sentence-encoder':
			sentence_encodings = corpus.get_sentence_encodings(model_name = 'universal-sentence-encoder', output_dir = embedding_dir)
		elif model_name == 'fasttext':
			sentence_encodings = corpus.get_sentence_encodings(model_name = 'fasttext', output_dir = embedding_dir)

	## reduce embedding dimensionality using pca and t-sne
	sentences = [e[0] for e in sentence_encodings]
	embeddings = [e[1] for e in sentence_encodings]
	embeddings = np.array(embeddings).squeeze()
	embeddings = pca_dim_reduction(embeddings, pca_dims=pca_dims)
	embeddings = tsne_dim_reduction(embeddings)

	## create sentence labels based on term presence in sentences
	if dataset_name == 'NHL_Corpus':
		## load hockey terms 
		terms = utils.load_textfile(terms_path)

		## create sentence label classes based on presence of hockey terms
		classes = [utils.sentence_contains_term(s,terms) for s in sentences]

	if dataset_name == 'MPC_Corpus':
		## no labels for MPC corpus yet
		classes = [0 for s in sentences]

	## plot T-SNE embeddings
	plot_tsne(embeddings, sentences, classes)
