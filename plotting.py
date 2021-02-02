import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def plot_tsne(embeddings, sentences, classes, legend_info = None):
	'''
	Input:
	embeddings: np.array of shape num_sentences x 2
	sentences: list of sentences of length num_sentences
	classes: list of binary labels of size num_sentences. 1 when the sentence belongs to class 1 and 0 otherwise.
	Output:
	plots embeddings with hover labels
	** The hover annotations are based on https://stackoverflow.com/a/47166787/1434041 **
	'''
	x = embeddings[:, 0]
	y = embeddings[:, 1]
	names = sentences

	fig, ax = plt.subplots()

	if legend_info is None:
		sc = plt.scatter(x, y, c = classes, alpha = 0.5) 
	else:
		class_mpatches = [mpatches.Patch(color=color, label=name) for name,color in legend_info]
		colormap = ListedColormap([c for _,c in legend_info])
		plt.legend(handles = class_mpatches)

		sc = plt.scatter(x, y, c = classes, cmap = colormap, alpha = 0.5)


	annot = ax.annotate('', xy = (0,0), 
						xytext = (-10,10), 
						textcoords = 'offset points',
						bbox = dict(boxstyle = 'round', fc = 'w'),
						arrowprops = dict(arrowstyle = '->'))
	annot.set_visible(False)

	def update_annot(ind):
		pos = sc.get_offsets()[ind['ind'][0]]
		annot.xy = pos
		text = '{}'.format('\n'.join([names[n] for n in ind['ind']]))
		annot.set_text(text)
		annot.get_bbox_patch().set_alpha(0.4)


	def hover(event):
		vis = annot.get_visible()
		if event.inaxes == ax:
			cont, ind = sc.contains(event)
			if cont:
				update_annot(ind)
				annot.set_visible(True)
				fig.canvas.draw_idle()
			else:
				if vis:
					annot.set_visible(False)
					fig.canvas.draw_idle()

	fig.canvas.mpl_connect('motion_notify_event', hover)

	fig1 = plt.gcf()
	plt.draw()
	plt.show()
	return fig1

def plot_precision_recall_f1_curve(precisions, recalls, thresholds, f1_scores):
	'''
	Given arrays of precisions, recalls, thresholds, f1_scores for each confidence threshold 
	Plots precision-recall curve with and f1 score overlay 
	'''
	plt.scatter(recalls, precisions, c = thresholds)
	plt.plot(recalls, precisions, label = 'precision-recall')
	plt.plot(recalls, f1_scores, c = 'r', label = 'f1 score')

	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.ylim(0.0, 1.0)	
	plt.xlim(0.2, 1.0)
	plt.legend()
	fig1 = plt.gcf()
	plt.show()
	return fig1


def plot_precision_by_conf(bins_, precision_by_bin):
	'''
	Given precisions array for each confidence bin
	Plots precision by confidence bin
	'''
	plt.plot(bins_, precision_by_bin)
	plt.scatter(bins_, precision_by_bin)
	plt.ylim(0.0, 1.0)	
	plt.xlabel('Confidence Bins')
	plt.ylabel('Precision')
	plt.title('Confidence Score Calibration')
	fig1 = plt.gcf()
	plt.show()
	return fig1