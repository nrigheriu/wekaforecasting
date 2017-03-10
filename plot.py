import os.path
import matplotlib.pyplot as plt

def plotData(data):
	fig, ax1 = plt.subplots()
	errorList = []
	tempList = [] 
	pointsList = []
	ax1.set_title("Error")
	ax1.set_title("Temp")
	lines = data.readlines()
	i = 0
	while i < len(lines)-1:
		error = float(lines[i])
		temp = float(lines[i+1])
		errorList.append(error)
		tempList.append(temp)
		pointsList.append(20)
		i += 2
	ax1.plot(tempList, errorList, label = "errors", color = 'b')
	plt.show()

if __name__ == '__main__':
	path = os.path.join(os.path.expanduser('~'), 'workspace/wekaforecasting-new-features','errorLogSAw0Start.txt')
	data = open(path)
	plotData(data)
	#errorLogSAw0Start