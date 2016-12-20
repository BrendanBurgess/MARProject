# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy as np
 
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	dataset.pop(0)
	np.random.shuffle(dataset)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/max((2*math.pow(stdev,2)), .0000001)))
	return (1 / max((math.sqrt(2*math.pi) * stdev), .000000001)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	trueNegative = 0
	truePositive = 0
	falseNegative = 0
	falsePositive = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == 0 and predictions[i] == 0:
			trueNegative += 1
		elif testSet[i][-1] == 1 and predictions[i] == 1:
			truePositive +=1
		elif testSet[i][-1] == 0 and predictions[i] == 1:
			falsePositive += 1
		elif testSet[i][-1] == 1 and predictions[i] == 0:
			falseNegative += 1
	#print "The Number of true negatives was %f " % trueNegative
	#print "The Number of true positives was %f " % truePositive
	#print "The Number of false negatives was %f " % falseNegative
	#print "The Number of false positives was %f " % falsePositive
	correct = truePositive + trueNegative
	return (correct/float(len(testSet))) * 100.0
 
def main(splitRatio):
	filename = 'MAR_Data.csv'
	#splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	#print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	return accuracy
	#print('Accuracy: {0}%').format(accuracy)

def rangefinder():
	splitRatios = [0.5, 0.67, 0.75, 0.8]
	for ratio in splitRatios:
		avgSum = 0.0
		for oops in range(0, 100):
			avgSum  = main(ratio)

		print ('Accuracy: {0}%').format(avgSum/100.0)

#print main(0.67)
rangefinder()