import argparse
import utilities as utils
import numpy as np

def get_relevance_score(group, eps, pi):
	score = 0.0
	for s in group:
		corp = s[:3]
		if corp == 'MPC':
			score += eps
		elif corp == 'NHL':
			score += pi
	return float(score)/len(group)

# python3 relevance_score.py --clusters_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/clusters.txt --output_dir data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/
if __name__ == '__main__':
	parser = argparse.ArgumentParser ()
	parser.add_argument('--clusters_path', dest = 'clusters_path', default = '/path/to/clusters.txt', help = 'provide a path')
	parser.add_argument('--output_dir', dest = 'output_dir', default = '/path/to/', help = 'provide a dir')
	parser.add_argument('--epsilon', dest = 'epsilon', type=float, default = 0.01, help = 'provide probability of p(y=1|C=MPC) = epsilon << 1')
	parser.add_argument('--pi', dest = 'pi', type=float, default = 0.8, help = 'provide probability of p(y=1|C=NHL) = pi')

	args = parser.parse_args()

	clusters_path = args.clusters_path
	output_dir = args.output_dir
	epsilon = args.epsilon
	pi  = args.pi

	# load LSH clusters computed using lsh.py
	lines = utils.load_textfile(clusters_path)
	lines = [l.strip('{} \n\'\"').replace(' ', '').replace('\'', '') for l in lines]

	groups = []
	for l in lines:
		groups.append(l.split(','))


	# obtain relevance score for each sentence using equation (3.5)
	sents = {}
	for group in groups:
		score = get_relevance_score(group, epsilon, pi)
		for s in group:
			if s in sents.keys():
				sents[s].append(score)
			else:
				sents[s] = [score]

	# some sentences are present in more than cluster (or group)
	# average relevance score for sentences that belong to more than one cluster
	NHL_scores = {}
	MPC_scores = {}
	for s,v in sents.items():
		if s[:3] == 'NHL':
			NHL_scores[s] = np.average(np.array(v))
		else:
			MPC_scores[s] = np.average(np.array(v))


	# sort based on relevance score 
	sorted_NHL_scores= dict(sorted(NHL_scores.items(), key=lambda item: item[1])) 

	# create bins for taking samples
	step = 0.1
	bins = [(i/10.) + step for i in range(10)]
	bins_NHL = {}
	for ths in bins:
		bins_NHL[ths] = []
		for k,v in sorted_NHL_scores.items():
			if v < ths and v >= ths-step:
				bins_NHL[ths].append(k)  

	print(bins_NHL.keys())

	# take samples from each bin
	sample_bins = {}
	num_samples_per_bin = 51
	for k,v in bins_NHL.items():
		print(k, len(v))
		sample_bins[k] = v[:num_samples_per_bin]

	# store samples as a pickle file
	selected_NHL_samples_path = '{}/selected_NHL_samples.pkl'.format(output_dir)
	utils.dump_picklefile(sample_bins, selected_NHL_samples_path)

	# store samples as a pickle file
	all_NHL_samples_path = '{}/all_NHL_samples.pkl'.format(output_dir)
	utils.dump_picklefile(sorted_NHL_scores, all_NHL_samples_path)
