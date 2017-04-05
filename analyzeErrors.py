import os.path
import re
import matplotlib.pyplot as plt
import numpy as np

def getToErrorPart(data):
	lines = data.readlines()
	predictions = []
	actualValues = []
	errors = []
	for k in range(len(lines)):
		if(lines[k][0:4] == "Step"):
			prediction = (lines[k].split("Prediction:")[1]).split(" Act")[0]
			prediction = prediction.replace(",", ".")
			prediction = float(prediction)
			#prediction += 5.26651
			prediction *= 10
			predictions.append(prediction)
			actual = (lines[k].split(" Act:")[1]).split(" MAE:")[0]
			actual = float(actual)
			actual *= 10	
			actualValues.append(actual)
	sum = 0.
	negativeCount = 0
	positiveCount = 0
	for i in range(len(predictions)):
		error = actualValues[i] - predictions[i]
		errors.append(error)
		sum += actualValues[i] - predictions[i]
		#print actualValues[i] - predictions[i]
	average =  sum/len(errors)
	lagRange = 24
	upperSum = 0
	lowerSum = 0
	for k in range(1, lagRange+1):
		for i in range(len(errors)- k):
			upperSum +=  (errors[i] - average)*(errors[i+k] - average)
		for i in range(len(errors)):
			lowerSum += (errors[i] - average)*(errors[i] - average)
		fraction = upperSum/lowerSum
		print "Lag:" + str(k) + "autocorr:" + str(fraction)
	print sum
	npError = np.histogram(errors, bins = np.arange(10))
	plt.ylabel('Frequency of errors')
	plt.xlabel('Error scale (in Watts)')
	plt.hist(errors, bins = 'auto')
	#plt.title("Error distribution")
	plt.show()
if __name__ == '__main__':
	path = os.path.abspath('parameter choice/BFIntervals/BFInterval36.txt')
	data = open(path)
	getToErrorPart(data)