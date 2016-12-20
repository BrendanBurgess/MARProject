import mlp
import numpy as np

def inputs():
	mar = np.loadtxt('SELECT_MAR.csv',delimiter=',', skiprows=1)
	np.random.shuffle(mar)

	#print "total"
	varLength = len(mar[0])
	trainin = mar[::2,:varLength -1]
	otherin = mar[1::2,:varLength -1]
	traintgt = mar[::2,varLength -1:varLength]
	othertgt = mar[1::2,varLength -1:varLength]



	testin = otherin[::2]
	validin = otherin[1::2]
	testtgt = othertgt[::2]
	validtgt = othertgt[1::2]

	print len(trainin)
	print len(testin)
	print len(validtgt)

	print othertgt

	return (trainin, traintgt,validin, validtgt, testin, testtgt)

def main():
	trainin, traintgt,validin, validtgt, testin, testtgt = inputs()
	net = mlp.mlp(trainin,traintgt,5,outtype='logistic')
	net.earlystopping(trainin,traintgt,validin,validtgt,0.1)
	net.confmat(testin,testtgt)

main()