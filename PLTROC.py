import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

class PLTROC():
#绘制ROC曲线

	def __init__(self, name, numsOfCurve = 0):
		self.name = name
		self.numsOfCurve = 0
		self.curveNames = []
		self.trueLabelsList = []
		self.predictLabelList = []
		self.colorList = ['cyan', 'darkorange', 'darkgreen']

	def addCurve(self, trueLabels, predictLabel, curveName = "line"):
		self.numsOfCurve += 1
		self.curveNames.append(curveName)
		self.trueLabelsList.append(trueLabels)
		self.predictLabelList.append(predictLabel)

	def delCurve(self):
		pass

	def pltCurve(self):
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		print(self.numsOfCurve)
		for i in range(self.numsOfCurve):
			fpr[i], tpr[i], _ = roc_curve(self.trueLabelsList[i], self.predictLabelList[i])

			roc_auc[i] = auc(fpr[i], tpr[i])
			print(i)


		plt.figure()
		lw = 2
		print(self.numsOfCurve)
		for i in range(self.numsOfCurve):
			plt.plot(fpr[i], tpr[i], color=self.colorList[int((i-i%5)/5)],
			         lw=lw, label=self.curveNames[i] +  '(area = %0.2f)' % roc_auc[i])
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(self.name + ' Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.savefig(self.name + '.jpg')
		plt.show()

		plt.close()
