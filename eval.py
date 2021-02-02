import argparse
import utilities as utils
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from plotting import plot_precision_by_conf, plot_precision_recall_f1_curve

def get_precision_by_conf(annot):
	'''
		calculates precision by predicted confidence interval
	'''
	precision = annot.groupby('conf_').mean()
	return precision.index, precision['gt_']

def get_pred_state(pred_, gt_):
	if pred_ == gt_:
		return 'tp' if pred_ == 1 else 'tn'
	if pred_ != gt_: 
		return 'fp' if pred_ == 1 else 'fn'

def get_precision_recall_by_threshold(annot):
	thresholds = np.arange(1, 10, 1)
	precisions = []
	recalls = []

	for th in thresholds:
		th_ = 'th{}'.format(th)
		annot['stat'] = [get_pred_state(annot.loc[r, th_], annot.loc[r, 'gt_'])for r in range(len(annot))]
		stat = annot.groupby('stat').count()[th_]

		tp = 0 if 'tp' not in stat.keys() else stat['tp']
		fp = 0 if 'fp' not in stat.keys() else stat['fp']
		tn = 0 if 'tn' not in stat.keys() else stat['tn']
		fn = 0 if 'fn' not in stat.keys() else stat['fn']

		precision = tp/(tp+fp)
		precisions.append(precision)

		recall = tp/(tp+fn)
		recalls.append(recall)

		df.drop('stat', axis=1, inplace=True)
	
	return precisions, recalls, thresholds

def get_f1_scores(precisions, recalls):
	return [round(2*(p*r)/(p+r),2) for (p,r) in zip(precisions, recalls)]
	
# cross-check sample id with sentence to create base annot file to be manually labeled
# python3 eval.py 	--samples_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/selected_NHL_samples.pkl 
# 					--id_to_sent_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/id_to_sent.csv 
# 					--annotation_csv_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/annot.csv
# 					--generate_base_annotation True

# evaluate annotated file
# python3 eval.py 	--annotation_csv_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/annotation_samples.csv
# 					--output_dir data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100
if __name__ == '__main__':
	parser = argparse.ArgumentParser ()

	parser.add_argument('--samples_path', dest = 'samples_path', default = '/path/to/samples.pkl', help = 'provide a path')
	parser.add_argument('--output_dir', dest = 'output_dir', default = '/path/to/', help = 'provide a dir')
	parser.add_argument('--id_to_sent_path', dest = 'id_to_sent_path', default = '/path/to/id_to_sent.txt', help = 'provide a path')
	parser.add_argument('--annotation_csv_path', dest = 'annotation_csv_path', default = '/path/to/id_to_sent.txt', help = 'provide a path')
	parser.add_argument('--generate_base_annotation', dest = 'generate_base_annotation', type=bool, default = False, help = 'are you generating the base annotation file and plan on manually annotating samples next?')

	args = parser.parse_args()

	output_dir = args.output_dir
	samples_path = args.samples_path
	id_to_sent_path = args.id_to_sent_path
	annotation_csv_path = args.annotation_csv_path
	generate_base_annot = args.generate_base_annotation

	# python3 eval.py 	--samples_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/selected_NHL_samples.pkl 
	# 					--id_to_sent_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/id_to_sent.csv 
	# 					--annotation_csv_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/annot.csv
	# 					--generate_base_annotation True
	if generate_base_annot: 
	# generate base annotation file
	# manual step: add ground truth labels as gt_
	# afterwards add add th1, ..., th9 as the pred_ for each confidence threshold
		samples = utils.load_picklefile(samples_path)
		id_to_sent = utils.load_csvfile(id_to_sent_path)
		sent_by_id = utils.csv_to_dict(id_to_sent)
		header = ['conf_', 'id_', 'sent_'] # gt_ has to be manually assigned 
		output = []
		for th_,v in samples.items():
			for id_ in v:
				row = [round(th_,2), id_, sent_by_id[id_][0]]
				output.append(row)
		df = pd.DataFrame(output, columns=header)
		df.to_csv(annotation_csv_path, index=False)
		exit()

	# if annotation file with ground truth and pred_ for each confidence level is available, use command below
	# python3 eval.py 	--annotation_csv_path data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100/annotated_samples.csv
	# 					--output_dir data/05_clustering/experiment_sents_2000\(NHL\)_2000\(MPC\)_reduceddim_50_numclusters_100
	# evaluate annotated csv
	annotations = utils.load_csvfile(annotation_csv_path)

	# csv to dataframe
	df = pd.read_csv(annotation_csv_path) 

	# plot precision for each confidence bin
	bin_, precision_ = get_precision_by_conf(df)
	plot = plot_precision_by_conf(bin_,precision_)
	plot.savefig('{}/precision_by_conf.png'.format(output_dir))

	# # plot precision recall curve 
	precision_, recall_, threshold_ = get_precision_recall_by_threshold(df)
	f1_ = get_f1_scores(precision_, recall_)
	plot = plot_precision_recall_f1_curve(precision_, recall_, threshold_, f1_)
	plot.savefig('{}/precision_recall.png'.format(output_dir))

	# print precision-recall
	for (p,r,f,t) in zip(precision_, recall_, f1_, threshold_):
		print('0.{}: ({} , {}), f1: {}'.format(t, round(p,2),round(r,2), f))


