import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def roc(Y_true,Y_pred_score):
	# print(np.sum(Y_pred_score))
	# Y_pred_score = np.divide(Y_pred_score,np.sum(Y_pred_score))
	# print(Y_pred_score.tolist())
	# quit(0)
	Y_pred_max, Y_pred_min = np.max(Y_pred_score),np.min(Y_pred_score)
	# print(Y_pred_max,Y_pred_min)
	# quit(0)
	intervals = np.linspace(Y_pred_min-1,Y_pred_max+1,1000)
	num_samples = len(Y_true)
	tpr,fpr = np.zeros(len(intervals)),np.zeros(len(intervals))
	for l in range(len(intervals)):
		temp = 0
		score_temp = intervals[l]
		fp_temp = 0
		tp_temp = 0
		Y_pred_temp = []
		for l1 in Y_pred_score:
			if l1 < score_temp:
				Y_pred_temp.append(0)
			else:
				Y_pred_temp.append(1)
		# print("Num classes",len(set(Y_true)))
		conf_mat = confusion_matrix(Y_true,Y_pred_temp,2)
		tpr[l] = conf_mat[1][1]/np.sum(conf_mat,axis = 1)[1]
		fpr[l] = conf_mat[0][1]/np.sum(conf_mat,axis = 1)[0]

	return tpr,fpr


def plot_roc_multiclass(tpr_multiclass,fpr_multiclass):
	plt.figure()
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"blue","green","red","cyan","magenta","yellow","black","brown"])
	n_classes = len(tpr_multiclass)
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr_multiclass[i], tpr_multiclass[i], color = color, label='ROC curve of class {0}'''.format(i))

	plt.plot([0, 1], [0, 1], 'k--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.show()

def confusion_matrix(Y_true,Y_pred,num_classes):
	# print(Y_true)
	# print(Y_pred)
	conf_mat = np.zeros((num_classes,num_classes))
	for i in range(len(Y_true)):
		conf_mat[Y_true[i]][Y_pred[i]] += 1

	return conf_mat

def plot_confusion_matrix(conf_mat,labels):

	conf_mat = np.array(conf_mat)
	fig = plt.figure()
	plt.title("Confusion Matrix")
	ax = fig.add_subplot(111)
	cax = ax.matshow(conf_mat, interpolation='nearest')
	fig.colorbar(cax)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.set_xticklabels(['']+labels)
	ax.set_yticklabels(['']+labels)

	plt.show()

