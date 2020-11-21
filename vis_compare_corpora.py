import numpy as np
import argparse
import utilities as utils
from dim_reduction import pca_dim_reduction, tsne_dim_reduction
from plotting import plot_tsne
 
if __name__ == '__main__':

	parser = argparse.ArgumentParser ()
	parser.add_argument('--corpora_dirs', dest = 'corpora_dirs', default = '/path/to/corpus_embeddings1,/path/to/corpus_embeddings2', help = 'provide a list of corpus embeddings directories separated by comma')
	parser.add_argument('--corpora_names', dest = 'corpora_names', default = 'corpus1,corpus2', help = 'provide a list of corpus names separated by comma')
	parser.add_argument('--colors', dest = 'colors', default='purple,gold,cyan,black', help = 'provide a list of colors separated by comma')
	parser.add_argument('--cap_size', dest = 'cap_size', default = 1000, help = 'provide a list of corpus embeddings directories')

	args = parser.parse_args()

	## dataset embedding dirs
	corpora_dirs = args.corpora_dirs.split(',')
	corpora_names = args.corpora_names.split(',')
	colors = args.colors.split(',')

	## limit the number of data points from each corpus
	cap_size = int(args.cap_size)

	embedding_sources = []
	[embedding_sources.append(c) for c in corpora_dirs]

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
	legend_info = []
	[legend_info.append((corpora_names[i],colors[i])) for i in range(len(corpora_names))]

	plot = plot_tsne(embeddings, sentences, classes, legend_info)
	plot.savefig('corpora_comparison.png', dpi=100)
