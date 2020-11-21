import numpy as np
import utilities as utils
from dim_reduction import pca_dim_reduction, tsne_dim_reduction
from plotting import plot_tsne
 
if __name__ == '__main__':

	## dataset embeddings
	embeddings_corpus1 = '/path/to/embeddings_corpus1'
	embeddings_corpus2 = '/path/to/embeddings_corpus2'

	embedding_sources = []
	embedding_sources.append(embeddings_corpus1)
	embedding_sources.append(embeddings_corpus2)

	## limit the number of data points from each corpus
	cap_size = 20000
	
	sentences = []
	embeddings = []
	lengths = []

	for source in embedding_sources:
		sentences_embeddings = utils.load_precomputed_embeddings(source)
		sents = [e[0] for e in sentences_embeddings[:cap_size]]
		embds = [e[1][0] for e in sentences_embeddings[:cap_size]]

		sentences.extend(sents)
		embeddings.extend(embds)

		lengths.append(len(sentences))

	embeddings = np.array(embeddings).squeeze()
	embeddings = pca_dim_reduction(embeddings, pca_dims=50)
	embeddings = tsne_dim_reduction(embeddings)

	size = len(sentences)

	## create sentence labels based on the dataset where the embeddings come from
	classes = [0 for i in range(size)]
	for i in range(len(lengths)-1):
		for j in range(lengths[i],lengths[i+1]):
			classes[j] = i+1

	## plot t-SNE embeddings
	legend_info = [('Corpus 1','purple'), ('Corpus 2','gold')]
	plot = plot_tsne(embeddings, sentences, classes, legend_info)

	plot.savefig('corpora_comparison.png', dpi=100)
