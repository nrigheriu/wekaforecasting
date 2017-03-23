import arff
import os.path
import re
def createLags(data, lagNumber, output):
	minutes = []
	power_sum = []
	#writing headers
	output.write("@relation 15_min_test\n@attribute local_15min DATE \"yyyy-MM-dd HH:mm\"\n")
	for i in range (lagNumber-1, 0, -1):
		output.write("@attribute lagY-%d numeric\n"%i)
	output.write("@attribute lagY numeric\n")
	output.write("@data\n")	
	lines = data.readlines()
	for i in range(4, len(lines)-1):
		minutes.append(lines[i][:18])
		power_sum.append(lines[i][19:-1])
	minutes.append(lines[i+1][:18])
	power_sum.append(lines[i+1][19:])	
	for i in range (lagNumber, len(power_sum)+1):
		string = ""
		string += minutes[i-1]
		string += ","
		for j in range(i-lagNumber, i):
			if(j < len(power_sum)):
				string += power_sum[j] + ","
		output.write(string)
		output.write("\n")

	output.close()
def createDayOfMonthAttribute(data):
	previousData = []
	daysOfMonth = []
	newData = ""
	lines = data.readlines()
	for k in range(len(lines)):					#copying previous header
		if(lines[k] == "@data\n"):
			break
	for j in range (k):
		newData +=  lines[j]
	newData += "@attribute dayOfMonth {'00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24','25', '26', '27', '28', '29', '30', '31'}\n"
	newData += lines[k]
	for i in range(k+1, len(lines)):
		previousData.append(lines[i][0:-1])
		daysOfMonth.append(lines[i][8:10])
	for i in range(len(previousData)):
		string = ""
		string += previousData[i]
		string += ","
		string += daysOfMonth[i]
		newData += string
		newData += "\n"
	return newData
def createMonthOfYearAttribute(data):
	previousData = []
	daysOfMonth = []
	newData = ""
	lines = data.readlines()
	for k in range(len(lines)):					#copying previous header
		if(lines[k] == "@data\n"):
			break
	for j in range (k):
		newData += lines[j]
	newData += "@attribute monthOfYear {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'}\n"
	newData += lines[k]
	for i in range(k+1, len(lines)):
		previousData.append(lines[i][0:-1])
		daysOfMonth.append(lines[i][5:7])
	for i in range(len(previousData)):
		string = ""
		string += previousData[i]
		string += ","
		string += daysOfMonth[i]
		newData += string
		newData += "\n"
	return newData

def createHolidayAttribute(data):
	previousData = []
	daysOfMonth = []
	holidaysInAYear = ["09-01", "10-13", "11-11",
	 "11-27",  "12-25", "08-27", "09-01", "12-26",
	  "01-01", "01-19", "02-16", "05-25", "06-19", "07-03"]
	holidayList = []
	newData = ""
	lines = data.readlines()
	for k in range(len(lines)):					
		if(lines[k] == "@data\n"):
			break
	for j in range (k):
		newData += lines[j]
	newData += "@attribute Holiday {'True', 'False'}\n"
	newData += lines[k]
	for i in range(k+1, len(lines)):
		previousData.append(lines[i][0:-1])
		if lines[i][5:10] in holidaysInAYear:
			holidayList.append('True')
		else:
			holidayList.append('False')
	for i in range(len(previousData)):
		string = ""
		string += previousData[i]
		string += ","
		string += holidayList[i]
		newData += string
		newData += "\n"
	return newData
def createHourOfDayAttribute(data):
	previousData = []
	hoursOfDay = []
	newData = ""
	lines = data.readlines()
	for k in range(len(lines)):					#copying previous header
		if(lines[k] == "@data\n"):
			break
	for j in range (k):
		newData += lines[j]
	newData += "@attribute hourOfDay {'00', '01', '02', '03', '04', '05', '06','07', '08',"
	newData += " '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'}\n"
	newData += lines[k]
	for i in range(k+1, len(lines)):
		previousData.append(lines[i][0:-1])
		hoursOfDay.append(lines[i][11:13])
	for i in range(len(previousData)):
		string = ""
		string += previousData[i]
		string += ","
		string += hoursOfDay[i]
		newData += string
		newData += "\n"
	return newData

#@param firstDay specifies what day of the week the data starts with (Monday is 0, Tuesday 1 etc.)
def createDayOfWeekAndWeekendAttribute(data, firstDay):				
	newData = ""
	previousData = []
	dayOfWeek = []
	weekendList = []
	lines = data.readlines()
	print "Lines no.:" + str(len(lines))
	for k in range(len(lines)):					#copying previous header
		if(lines[k] == "@data\n"):
			break
	for j in range (k):
		newData += lines[j]
	newData += "@attribute dayOfWeek {'0', '1', '2', '3', '4', '5', '6'}\n"
	newData += "@attribute Weekend {'True', 'False'}\n"
	newData += lines[k]
	temporaryDay = int(lines[k+1][8:10])
	counter = 0
	for i in range(k+1, len(lines)):
		previousData.append(lines[i][0:-1])
		currentDay = int(lines[i][8:10])
		if(temporaryDay != currentDay):				#figure out when the day changed while parsing the data
			temporaryDay = currentDay
			counter += 1
		dayOfWeek.append((counter+firstDay)%7)
		if (((counter+firstDay)%7 == 5) or ((counter+firstDay)%7 == 6)):
			weekendList.append('True')
		else:
			weekendList.append('False')
	for i in range(len(previousData)):
		string = ""
		string += previousData[i]
		string += ","
		string += str(dayOfWeek[i])
		string += "," 
		string += str(weekendList[i])
		newData += string
		newData += "\n"
	return newData
def insertQuarterOfHour(data):
	previousData = []
	quartersOfHour = []
	newData = ""
	lines = data.readlines()
	for k in range(len(lines)):					#copying previous header
		if(lines[k] == "@data\n"):
			break
	for j in range (k):
		newData += lines[j]
	newData += "@attribute quarterOfHour {'00', '15', '30', '45'}\n"
	newData += lines[k]
	for i in range(k+1, len(lines)):
		previousData.append(lines[i][0:-1])
		quartersOfHour.append(lines[i][14:16])
	for i in range(len(previousData)):
		string = ""
		string += previousData[i]
		string += ","
		string += quartersOfHour[i]
		newData += string
		newData += "\n"
	return newData
def multiplyPowerBy100(data):
	newData = ""
	timeStamp = []
	restOfFeatures = []
	power_list = []
	lines = data.readlines()
	print "Lines no. in multiplyby100:" + str(len(lines))
	for k in range(len(lines)):					#copying previous header
		newData += lines[k]
		if(lines[k] == "@data\n"):
			break
	for i in range(k+1, len(lines)):
		timeStamp.append(lines[i][0:25])
		power_list.append(float(lines[i][25:29]) * 100)
		restOfFeatures.append(lines[i][29:-1])
	for i in range(len(timeStamp)):
		string = ""
		string += timeStamp[i]
		string += str(power_list[i])
		string += restOfFeatures[i]
		newData += string
		newData += "\n"
	return newData
def interpolateTemperatures(data):
	newData = ""
	lines = data.readlines()
	for k in range(len(lines)):					#copying previous header
		newData += lines[k]
		if(lines[k] == "@data\n"):
			break
	for i in range (k+1, len(lines)-1):
		newData += lines[i]
		line15MinAfter  = lines[i][0:14] + '15' + lines[i][16:23] +  str(float(lines[i][23:-1]) + (float(lines[i+1][23:-1]) - float(lines[i][23:-1]))/4)[0:5]
		line30MinAfter = lines[i][0:14] + '30' + lines[i][16:23] +  str(float(lines[i][23:-1]) + (float(lines[i+1][23:-1]) - float(lines[i][23:-1]))/2)[0:5]
		line45MinAfter = lines[i][0:14] + '45' + lines[i][16:23] +  str(float(lines[i][23:-1]) + 3*(float(lines[i+1][23:-1]) - float(lines[i][23:-1]))/4)[0:5]
		newData += line15MinAfter + "\n"
		newData += line30MinAfter + "\n"
		newData += line45MinAfter + "\n"
	newData += lines[len(lines)-1]
	newData += "\n"
	return newData
def insertTempToData(data, tempData):
	newData = ""
	dataLines = data.readlines()
	tempLines = tempData.readlines()
	for k in range(len(dataLines)):					#copying previous header
		if(dataLines[k] == "@data\n"):
			break
	for j in range (k):
		newData += dataLines[j]
	newData += "@attribute temperature NUMERIC\n"
	newData += dataLines[k]
	for l in range (len(tempLines)):
		if(tempLines[l] == "@data\n"):
			break
	for i in range(k+1, len(dataLines)):
		newData += dataLines[i][0:-1] + "," + tempLines[l+1][25:-1]
		newData += "\n"
		l += 1
	return newData

def adjustTimeStamp(data):
	newData = ""
	timeStamp = []
	restOfFeatures = []
	lines = data.readlines()
	for k in range(len(lines)):					#copying previous header
		newData += lines[k]
		if(lines[k] == "@data\n"):
			break
	for i in range(k+1, len(lines)):
		timeStamp.append(lines[i][0:10] + "T" + lines[i][11:19] + "-0600")
		restOfFeatures.append(lines[i][19:-1])
	for i in range(len(timeStamp)):
		string = ""
		string += timeStamp[i]
		string += restOfFeatures[i]
		newData += string
		newData += "\n"
	return newData
def createDerivativeAttributes(data):
	newData = ""
	timeStamp = []
	restOfFeatures = []
	power_list = []
	lines = data.readlines()
	print "Lines no. in multiplyby100:" + str(len(lines))
	for k in range(len(lines)):					#copying previous header
		newData += lines[k]
		if(lines[k] == "@data\n"):
			break
	for i in range(k+1, len(lines)):
		timeStamp.append(lines[i][0:25])
		power_list.append(float(lines[i][25:29]) * 100)
		restOfFeatures.append(lines[i][29:-1])
	for i in range(len(timeStamp)):
		string = ""
		string += timeStamp[i]
		string += str(power_list[i])
		string += restOfFeatures[i]
		newData += string
		newData += "\n"
	return newData

if __name__ == '__main__':
	path = os.path.join(os.path.expanduser('~'), 
		'workspace/wekaforecasting-new-features/dataSets','1year3months_1aggregate2_extraFeaturesx.arff')
	pathWrite = os.path.join(os.path.expanduser('~'), 
		'workspace/wekaforecasting-new-features/dataSets','1year3months_1aggregate2_extraFeaturesz.arff')
	firstDayInSet = 4;
	pathTemp = os.path.join(os.path.expanduser('~'),
	 'workspace/wekaforecasting-new-features/dataSets','1year3months_weather_interpolated.arff')
	
	#pathWriteTemp = os.path.join(os.path.expanduser('~'), 'workspace/wekaforecasting-new-features/dataSets','1year3months_weather_interpolated.arff')
	# data = open(pathTemp)
	# newData = interpolateTemperatures(data)
	# output = open(pathWriteTemp, 'w')
	# output.write(newData)
	# output.close()

	# data = open(pathTemp)
	# newData = adjustTimeStamp(data)
	# output = open(pathWrite, 'w')
	# output.write(newData)
	# output.close()

	# tempData = open(pathTemp)
	# data = open(path)
	# newData = insertTempToData(data, tempData)
	# output = open(pathWrite, 'w')
	# output.write(newData)
	# output.close()

	# data = open(path)
	# newData = multiplyPowerBy100(data)
	# output = open(pathWrite, 'w')
	# output.write(newData)
	# output.close()

	data = open(path)
	newData = createMonthOfYearAttribute(data)
	output = open(pathWrite, 'w')
	output.write(newData)
	output.close()

	data = open(pathWrite)
	newData = createDayOfMonthAttribute(data)
	output = open(pathWrite, 'w')
	output.write(newData)
	output.close()

	data = open(pathWrite)
	newData	= createDayOfWeekAndWeekendAttribute(data, firstDayInSet)
	output = open(pathWrite, 'w')	
	output.write(newData)
	output.close()

	data = open(pathWrite)
	newData = createHourOfDayAttribute(data)
	output = open (pathWrite, 'w')
	output.write(newData)
	output.close()

	data = open(pathWrite)
	newData = insertQuarterOfHour(data)
	output = open(pathWrite, 'w')
	output.write(newData)
	output.close()

	data = open(pathWrite)
	newData = createHolidayAttribute(data)
	output = open(pathWrite, 'w')
	output.write(newData)
	output.close()
	
