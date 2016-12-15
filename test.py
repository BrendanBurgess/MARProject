import numpy as np
import pcn

mar = np.loadtxt('MAR_Data.csv',delimiter=',', skiprows=1)

np.random.shuffle(mar)

print "total"
print len(mar)
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