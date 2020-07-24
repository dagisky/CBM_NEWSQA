import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


def visualize(weights, dialog=None, name='model'):
	"""
	Visualize attention Weight
		Args:
			weights (Tensor): attention weight
			dialog (list): toxenized list of words
	"""
	weight = torch.cat(weights, dim=0).squeeze().transpose(1,0)	
	weight = weight.cpu().detach().numpy()
	shape = weight.shape
	heads = list()
	for i in range(shape[1]):
		# heads.append('h'+str(i))
		heads.append(str(i))

	print(heads)
	df = pd.DataFrame(weight, columns=heads, index=dialog ) # , columns=heads, index=dialog
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
	fig.colorbar(cax)
	x_tick_spacing = 1
	y_tick_spacing = 5
	ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
	ax.set_xticklabels([''] + list(df.columns))
	ax.set_yticklabels([''] + list(df.index))
	plt.savefig(f'model/{name}.png')
	plt.show()

