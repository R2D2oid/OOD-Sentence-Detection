import argparse
import numpy as np
import utilities as utils
from corpus import Corpus, convert_subtitles_to_sentences
from dim_reduction import pca_dim_reduction, tsne_dim_reduction
from plotting import plot_tsne
 
if __name__ == '__main__':

	parser = argparse.ArgumentParser ()
	parser.add_argument('--subtitle_dir', dest = 'subtitle_dir', default = 'data/subtitles_with_timestamp', help = 'subtitles dir')
	parser.add_argument('--sentence_dir', dest = 'sentence_dir', default = 'data/sentences', help = 'sentences dir')
	parser.add_argument('--artifact_dir', dest = 'artifact_dir', default = 'data/artifacts', help = 'artifacts dir')
	parser.add_argument('--terms_path', dest = 'terms_path', default = 'data/hockey_terms.txt', help = 'terms path')
	parser.add_argument('--freqdist_path', dest = 'freqdist_path', default = 'data/artifacts/NHL.corpus.freqdist.pkl', help = 'corpus freqdist path')
	parser.add_argument('--model_name', dest = 'model_name', default = 'universal-sentence-encoder', help = 'model for extracting corpus embeddings: universal-sentence-encoder or fasttext ')
	parser.add_argument('--extract_sentences', dest = 'extract_sentences', default = False, help = 'boolean value indicating whether to extract sentences from subtitles or load existing sentences')
	parser.add_argument('--embedding_path', dest = 'embedding_path', help = 'If embeddings path is provided, embeddings are loaded from the path. skips embeddings extraction.')

	args = parser.parse_args()

	subtitle_dir  = args.subtitle_dir
	sentence_dir  = args.sentence_dir
	artifact_dir  = args.artifact_dir
	terms_path	  = args.terms_path
	freqdist_path = args.freqdist_path	
	model_name 	  = args.model_name
	extract_sentences = args.extract_sentences
	embedding_path = args.embedding_path
	
	utils.create_dir_if_not_exist(sentence_dir)
	utils.create_dir_if_not_exist(artifact_dir)

	if extract_sentences: # default: True. otherwise assumes that sentences are already extracted
		convert_subtitles_to_sentences(subtitle_dir, sentence_dir)

	corpus = Corpus(sentence_dir)
	# corpus = Corpus(sentence_dir, freqdist_path = freqdist_path)

	## get infrequent words from corpus
	# infrequent_words = corpus.get_least_frequent_words()

	## encode sentences 
	if embedding_path: 	# load previously calculated embeddings if the path is provided
		sentence_encodings = utils.load_picklefile(embedding_path)
	else: # if path is not provided extract embeddings
		if model_name == 'universal-sentence-encoder':
			corpus_embeddings_path = '{}/embeddings.corpus.universal.pkl'.format(artifact_dir)
			sentence_encodings = corpus.get_sentence_encodings(model_name = 'universal-sentence-encoder')
		elif model_name == 'fasttext':
			corpus_embeddings_path = '{}/embeddings.corpus.fasttext.pkl'.format(artifact_dir)
			sentence_encodings = corpus.get_sentence_encodings(model_name = 'fasttext')
		utils.dump_picklefile(sentence_encodings, corpus_embeddings_path)

	## reduce embedding dimensionality using pca and t-sne
	sentences = [e[0] for e in sentence_encodings]
	embeddings = [e[1] for e in sentence_encodings]
	embeddings = np.array(embeddings).squeeze()
	embeddings = pca_dim_reduction(embeddings)
	embeddings = tsne_dim_reduction(embeddings)

	## load hockey terms 
	terms = utils.load_textfile(terms_path)

	## create sentence label classes based on presence of hockey terms
	classes = [utils.sentence_contains_term(s,terms) for s in sentences]

	## plot T-SNE embeddings
	plot_tsne(embeddings, sentences, classes)
