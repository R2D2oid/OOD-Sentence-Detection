import numpy as np
import argparse
import utilities as utils
from dim_reduction import pca_dim_reduction, tsne_dim_reduction
from LocalitySensitiveHashing import *

def get_lsh_clusters(data_path, output_path, dim, num_clusters, r = 50, b = 100 ):
	lsh = LocalitySensitiveHashing(
					datafile = data_path,
					dim = dim,	#  data dimensionality
					r = r,		# Number of rows in each band (each row is for one hash func)
					b = b,		# Number of bands self.how_many_hashes =  r * b
					expected_num_of_clusters = num_clusters
					)

	# load data
	lsh.get_data_from_csv()

	# obtain lsh clusters
	lsh.initialize_hash_store()
	lsh.hash_all_data()

	# similarity_groups
	sim_grps = lsh.lsh_basic_for_neighborhood_clusters()
	sim_grps_coalesced = lsh.merge_similarity_groups_with_coalescence(sim_grps)
	sim_grps_merged = lsh.merge_similarity_groups_with_l2norm_sample_based(sim_grps_coalesced)

	# write clusters to the output
	lsh.write_clusters_to_file(sim_grps_merged, output_path)
	
def load_sentence_embeddings(corpora_dirs, corpora_names, cap_size, reduced_dim = None):
	sentences = []
	embeddings = []

	for (source, name) in zip(corpora_dirs, corpora_names):
		sentences_embeddings = utils.load_precomputed_embeddings(source)
		
		print('total len, {}'.format(name))
		print(len(sentences_embeddings))

		sents = [[name, e[0]] for e in sentences_embeddings[:cap_size]]
		embds = [e[1][0] for e in sentences_embeddings[:cap_size]]

		sentences.extend(sents)
		embeddings.extend(embds)

	# reduce dimensionality
	if reduced_dim:
		embeddings = pca_dim_reduction(embeddings, pca_dims = reduced_dim)

	embeddings_with_id = []
	sentences_with_id = []
	count = 0

	for s,e in zip(sentences,embeddings):
		sent_id = '{}_{}'.format(s[0],count)
		
		row = [sent_id]
		row.extend(e)
		embeddings_with_id.append(row)
	
		sentences_with_id.append([sent_id, s[1]])

		count += 1

	return sentences_with_id, embeddings_with_id

# python3 lsh.py --corpora_dirs=data/04_embeddings/embeddings_NHL_raw_partial,data/04_embeddings/embeddings_MPC_raw_full --corpora_names=NHL,MPC --clustering_dir=data/05_clustering --cap_size=2000 --reduced_dim=50 --num_clusters=10
if __name__ == '__main__':
	parser = argparse.ArgumentParser ()
	parser.add_argument('--corpora_dirs', dest = 'corpora_dirs', default = '/path/to/corpus_embeddings1,/path/to/corpus_embeddings2', help = 'provide a list of corpus embeddings directories separated by comma')
	parser.add_argument('--corpora_names', dest = 'corpora_names', default = 'corpus1,corpus2', help = 'provide a list of corpus names separated by comma')
	parser.add_argument('--colors', dest = 'colors', default='purple,gold,cyan,black', help = 'provide a list of colors separated by comma')
	parser.add_argument('--cap_size', dest = 'cap_size', type=int, default = 2000, help = 'provide an integer')
	
	# clustering params
	parser.add_argument('--clustering_dir', dest = 'clustering_dir', default = '/path/to/clusters', help = 'provide a directory path')
	parser.add_argument('--reduced_dim', dest = 'reduced_dim', type=int, default = 50, help = 'provide an integer')
	parser.add_argument('--num_clusters', dest = 'num_clusters', type=int, default = 10, help = 'provide an integer')
	
	args = parser.parse_args()

	## dataset embedding dirs
	corpora_dirs = args.corpora_dirs.split(',')
	corpora_names = args.corpora_names.split(',')
	colors = args.colors.split(',')

	# clustering params
	clustering_dir = args.clustering_dir
	reduced_dim = args.reduced_dim
	num_clusters = args.num_clusters

	## limit the number of data points from each corpus
	cap_size = int(args.cap_size)

	experiment_name = 'experiment_sents_{}({})_{}({})_reduceddim_{}_numclusters_{}'.format(cap_size, corpora_names[0], cap_size, corpora_names[1], reduced_dim, num_clusters)
	experiment_path = '{}/{}'.format(clustering_dir, experiment_name)
	utils.create_dir_if_not_exist(experiment_path)

	# load sentences and their embeddings from previously extracted files
	# add an id to each sentence to prepare them for clustering
	sents_with_id, embs_with_id = load_sentence_embeddings(corpora_dirs, corpora_names, cap_size, reduced_dim = reduced_dim)

	embs_with_id_path =  '{}/embeddings_with_sent_id.csv'.format(experiment_path)
	np.savetxt(embs_with_id_path, np.asarray(embs_with_id), delimiter=',', fmt='%s')

	sents_with_id_path =  '{}/sentences_with_sent_id.csv'.format(experiment_path)
	np.savetxt(sents_with_id_path, np.asarray(sents_with_id), delimiter=',', fmt='%s')

	output_path = '{}/clusters.txt'.format(experiment_path)
	get_lsh_clusters(embs_with_id_path, output_path, dim = reduced_dim, num_clusters = num_clusters, r = 50, b = 100)
