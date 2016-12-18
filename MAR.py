import numpy as np
import pcn

"""
DATA LOADING AND PRINT SETTINGS
"""
np.set_printoptions(precision=3)
mar = np.loadtxt('MAR_Data.csv',delimiter=',', skiprows=1)
np.random.shuffle(mar)

print "total"
print len(mar[0])
trainin = mar[::3,:60]
trainin2 = mar[1::3,:60]
traintgt = mar[::3,60:61]
traintgt2 = mar[1::3,60:61]

print "individuals"
print len(trainin)
print len(trainin2)

print "final"
trainin = np.concatenate((trainin, trainin2), axis =0)
traintgt = np.concatenate((traintgt, traintgt2), axis =0)
print len(trainin)

testin = mar[2::3,:60]
testtgt = mar[2::3,60:61]

#print(testin[0].astype(int))
#print(testtgt)




def paramsFinderFull():
	
	maxValue = -1.0;
	maxValueLearnRate = 0;
	maxValueNiterations = 0;
	for N in range(1, 7):
		for R in range (1, 6):
			a = []
			for x in range(0, 1000):
				p1 = pcn.pcn(trainin,traintgt)
				p1.pcntrain(trainin,traintgt,0.05 * R, 50 * N)
				a.append(p1.confmat(testin,testtgt))
			avg = ((np.sum(a))/1000)*100
			if(avg > maxValue):
				maxValue = avg
				maxValueNiterations = 50 * N
				maxValueLearnRate = 0.05 * R

	print "The value was %f which was when the learning rate was %f and the NIterations were %d" %(maxValue, maxValueLearnRate, maxValueNiterations)

def runnerFull (learningRate, NIterations):
	#trainin = mar[::2,:60]
	#testin = mar[1::2,:60]
	#traintgt = mar[::2,60:61]
	#testtgt = mar[1::2,60:61]
	
	print "testin"
	print len(testin)
	print len(testtgt)
	
	a = []
	for x in range(0, 100):
		p1 = pcn.pcn(trainin,traintgt)
		p1.pcntrain(trainin,traintgt,learningRate, NIterations)
		value = p1.confmat(testin,testtgt)
		#print value
		a.append(value)
		
	avg = ((np.sum(a))/100)*100
	print avg

# Perceptron training on the preprocessed dataset
print "raw data output"
#paramsFinderFull()
runnerFull(0.15, 250)