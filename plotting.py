import matplotlib.pyplot as plt

def plot_tsne(embeddings, sentences, classes):
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
	sc = plt.scatter(x, y, c = classes, alpha = 0.5) 
	# ax.set_axis_off()
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
	plt.show()