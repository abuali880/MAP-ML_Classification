import numpy
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import operator

PriorFlag = True  # If set to True MAP classifier will work, If not ML classifier will work



###################################### Functions Section ######################################

################################## Q2 ##################################
def Mahalanobis(Mean, Cova, X):   # Calculate Mahalanobis distance for a Given X
    a = X - Mean  # difference between X and Mean
    res = a.transpose() * numpy.linalg.inv(Cova) * a  # calculate Mahalanobis
    return res
    pass

################################ Q1 ##########################################
def Discrim(Mean, Cova, X, Prior):  # Calculate Discr. Fn value for a Given X
    Mahal = -0.5 * Mahalanobis(Mean, Cova, X)  # Mahalanobis
    sec = Mahal - (int(Cova.shape[0]) / 2) * math.log(22 / 14)  # d/2 * ln(PI/2)
    sec = sec - 0.5 * math.log(numpy.linalg.det(Cova))  # - 1/2 * ln|Covariance|
    if PriorFlag:
    	sec = sec + math.log(Prior)  # adding Prior
    	pass
    return sec
    pass

################################ Q3 ##########################################
def MeanM(C):    # Calculating Class mean
    return C.mean(1) 
    pass

def CovM(C):    # Calculating Class covariance
    return numpy.cov(C)
    pass



def des(X, ClassW1, ClassW2, ClassW3):  #Given specific point to decide to which class it belongs  
    fir = Discrim(ClassW1[0], ClassW1[1], X, ClassW1[2]) # Discr. fn value for the first class
    sec = Discrim(ClassW2[0], ClassW2[1], X, ClassW2[2]) # Discr. fn value for the second class
    thr = Discrim(ClassW3[0], ClassW3[1], X, ClassW3[2]) # Discr. fn value for the third class
    res = [fir, sec, thr]  # put results in list
    max_index, max_value = max(enumerate(res), key=operator.itemgetter(1)) # Get index & value of Max. element 
    return max_index + 1
    pass


def desVector(X, ClassW1, ClassW2, ClassW3):  # Decide for a vector of points 
    out = []
    for i in range(X.shape[1]):
        out.append(des(numpy.matrix([[X[0, i]], [X[1, i]], [X[2, i]]]), ClassW1, ClassW2, ClassW3))
        pass
    return out
    pass


################################## Classes data ##########################################

w1 = numpy.matrix([[-5.01, -5.43, 1.08, 0.86, -2.67, 4.94, -2.51, -2.25, 5.56, 1.03],
                   [-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33],
                   [-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33]])
w2 = numpy.matrix([[-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50],
                   [-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32],
                   [-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31]])
w3 = numpy.matrix([[5.35, 5.12, -1.34, 4.48, 7.11, 7.17, 5.75, 0.77, 0.90, 3.52],
                   [2.26, 3.22, -5.31, 3.42, 2.39, 4.33, 3.97, 0.27, -0.43, -0.36],
                   [8.13, -2.66, -9.87, 5.19, 9.21, -0.98, 6.65, 2.41, -8.71, 6.43]])



################################ Calculating Mean and Covariance for each class and assign it to the class list ##########################

ClassW1 = [MeanM(w1),CovM(w1),0.8] # Mean , Covariance , Proir for Class 1
ClassW2 = [MeanM(w2),CovM(w2),0.1] # Mean , Covariance , Proir for Class 2
ClassW3 = [MeanM(w3),CovM(w3),0.1] # Mean , Covariance , Proir for Class 3

############################### Printing Q3 & Q5 Result ######################################################
print("For Class w1 : \nMean = ")
print(MeanM(w1))
print("Covariance = ")
print(CovM(w1))
print("For Class w2 : \nMean = ")
print(MeanM(w2))
print("Covariance = ")
print(CovM(w2))
print("For Class w3 : \nMean = ")
print(MeanM(w3))
print("Covariance = ")
print(CovM(w3))
print("\nUsing MAP classifier:")
print("P1 belongs to the class w"+str(des(numpy.matrix([[1], [2], [1]]), ClassW1, ClassW2, ClassW3)))
print("P2 belongs to the class w"+str(des(numpy.matrix([[5], [3], [1]]), ClassW1, ClassW2, ClassW3)))
print("P3 belongs to the class w"+str(des(numpy.matrix([[0], [0], [0]]), ClassW1, ClassW2, ClassW3)))
print("P4 belongs to the class w"+str(des(numpy.matrix([[1], [0], [0]]), ClassW1, ClassW2, ClassW3)))
PriorFlag = False
print("\nUsing ML classifier:")
print("P1 belongs to the class w"+str(des(numpy.matrix([[1], [2], [1]]), ClassW1, ClassW2, ClassW3)))
print("P2 belongs to the class w"+str(des(numpy.matrix([[5], [3], [1]]), ClassW1, ClassW2, ClassW3)))
print("P3 belongs to the class w"+str(des(numpy.matrix([[0], [0], [0]]), ClassW1, ClassW2, ClassW3)))
print("P4 belongs to the class w"+str(des(numpy.matrix([[1], [0], [0]]), ClassW1, ClassW2, ClassW3)))
#PriorFlag = True
'''
x = numpy.matrix([[3], [5], [1]])

print(Discrim(MeanM(w1), CovM(w1), x, 0.8))
print(Discrim(MeanM(w2), CovM(w2), x, 0.1))
print(Discrim(MeanM(w3), CovM(w3), x, 0.1))
print(des(x, ClassW1, ClassW2, ClassW3))
'''
h = 0.2 # step used in meshgrid 

################################# Finding Min & Max for each feature ###################################
x1_min = numpy.array([w1[0, :].min() - 1, w2[0, :].min() - 1, w2[0, :].min() - 1]).min()
x1_max = numpy.array([w1[0, :].max() + 1, w2[0, :].max() + 1, w2[0, :].max() + 1]).max()

x2_min = numpy.array([w1[1, :].min() - 1, w2[1, :].min() - 1, w2[1, :].min() - 1]).min()
x2_max = numpy.array([w1[1, :].max() + 1, w2[1, :].max() + 1, w2[1, :].max() + 1]).max()

x3_min = numpy.array([w1[2, :].min() - 1, w2[2, :].min() - 1, w2[2, :].min() - 1]).min()
x3_max = numpy.array([w1[2, :].max() + 1, w2[2, :].max() + 1, w2[2, :].max() + 1]).max()


levels = [0.9, 1.1, 1.9, 2.1, 2.9, 3.1]
fig, axes = plt.subplots(nrows=2, ncols=4)

##############################Prepare Meshgrid################################################
xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, h), numpy.arange(x2_min, x2_max, h))

############################Give any feature a constant value to Draw in 2D##################3
zz = numpy.empty((xx1.shape[0], xx1.shape[1]))
for i in range(xx1.shape[0]):
    for j in range(xx1.shape[1]):
        zz[i, j] = 1

###########################Take Class dicision for each point and store results in Z################
Z1 = numpy.zeros((xx1.shape[0], xx1.shape[1]))
for i in range(xx1.shape[0]):
    use = numpy.c_[xx1[i, :], xx2[i, :], zz[i, :]]
    use = use.T
    Z1[i, :] = desVector(use,ClassW1, ClassW2, ClassW3) 

##############################Prepare Meshgrid################################################
xx12, xx22 = numpy.meshgrid(numpy.arange(x1_min, x1_max, h), numpy.arange(x3_min, x3_max, h))

############################Give any feature a constant value to Draw in 2D##################
zz = numpy.empty((xx12.shape[0], xx12.shape[1]))
for i in range(xx12.shape[0]):
    for j in range(xx12.shape[1]):
        zz[i, j] = 1

###########################Take Class dicision for each point and store results in Z################
Z12 = numpy.zeros((xx12.shape[0], xx12.shape[1]))
for i in range(xx12.shape[0]):
    use = numpy.c_[xx12[i, :], zz[i, :], xx22[i, :]]
    use = use.T
    Z12[i, :] = desVector(use,ClassW1, ClassW2, ClassW3)



##############################Prepare Meshgrid################################################
xx13, xx23 = numpy.meshgrid(numpy.arange(x2_min, x2_max, h), numpy.arange(x3_min, x3_max, h))

############################Give any feature a constant value to Draw in 2D##################
zz = numpy.empty((xx13.shape[0], xx13.shape[1]))
for i in range(xx13.shape[0]):
    for j in range(xx13.shape[1]):
        zz[i, j] = 1

###########################Take Class dicision for each point and store results in Z################
Z13 = numpy.zeros((xx13.shape[0], xx13.shape[1]))
for i in range(xx13.shape[0]):
    use = numpy.c_[zz[i, :], xx13[i, :], xx23[i, :]]
    use = use.T
    Z13[i, :] = desVector(use,ClassW1, ClassW2, ClassW3)     

axes[0,2].contourf(xx1, xx2, Z1, levels=levels, cmap=plt.cm.Paired)
#plt.colorbar()
#plt.show()
axes[0,3].contourf(xx12, xx22, Z12, levels=levels, cmap=plt.cm.Paired)
#plt.colorbar()
#plt.show()
axes[1,2].contourf(xx13, xx23, Z13, levels=levels, cmap=plt.cm.Paired)
#plt.colorbar()

PriorFlag = True

##############################Prepare Meshgrid################################################
xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, h), numpy.arange(x2_min, x2_max, h))

############################Give any feature a constant value to Draw in 2D##################3
zz = numpy.empty((xx1.shape[0], xx1.shape[1]))
for i in range(xx1.shape[0]):
    for j in range(xx1.shape[1]):
        zz[i, j] = 1

###########################Take Class dicision for each point and store results in Z################
Z1 = numpy.zeros((xx1.shape[0], xx1.shape[1]))
for i in range(xx1.shape[0]):
    use = numpy.c_[xx1[i, :], xx2[i, :], zz[i, :]]
    use = use.T
    Z1[i, :] = desVector(use,ClassW1, ClassW2, ClassW3) 

##############################Prepare Meshgrid################################################
xx12, xx22 = numpy.meshgrid(numpy.arange(x1_min, x1_max, h), numpy.arange(x3_min, x3_max, h))

############################Give any feature a constant value to Draw in 2D##################
zz = numpy.empty((xx12.shape[0], xx12.shape[1]))
for i in range(xx12.shape[0]):
    for j in range(xx12.shape[1]):
        zz[i, j] = 1

###########################Take Class dicision for each point and store results in Z################
Z12 = numpy.zeros((xx12.shape[0], xx12.shape[1]))
for i in range(xx12.shape[0]):
    use = numpy.c_[xx12[i, :], zz[i, :], xx22[i, :]]
    use = use.T
    Z12[i, :] = desVector(use,ClassW1, ClassW2, ClassW3)



##############################Prepare Meshgrid################################################
xx13, xx23 = numpy.meshgrid(numpy.arange(x2_min, x2_max, h), numpy.arange(x3_min, x3_max, h))

############################Give any feature a constant value to Draw in 2D##################
zz = numpy.empty((xx13.shape[0], xx13.shape[1]))
for i in range(xx13.shape[0]):
    for j in range(xx13.shape[1]):
        zz[i, j] = 1

###########################Take Class dicision for each point and store results in Z################
Z13 = numpy.zeros((xx13.shape[0], xx13.shape[1]))
for i in range(xx13.shape[0]):
    use = numpy.c_[zz[i, :], xx13[i, :], xx23[i, :]]
    use = use.T
    Z13[i, :] = desVector(use,ClassW1, ClassW2, ClassW3)     

axes[0,0].contourf(xx1, xx2, Z1, levels=levels, cmap=plt.cm.Paired)
axes[0,0].set_title('Sharing x per column, y per row')
#plt.colorbar()
#plt.show()
axes[0,1].contourf(xx12, xx22, Z12, levels=levels, cmap=plt.cm.Paired)
#plt.colorbar()
#plt.show()
axes[1,0].contourf(xx13, xx23, Z13, levels=levels, cmap=plt.cm.Paired)
#plt.colorbar()
plt.show()




'''
Z1 = numpy.zeros((xx1.shape[0], xx1.shape[1]))
for i in range(xx1.shape[0]):
    use = numpy.c_[xx1[i, :], xx2[i, :], zz[i, :]]
    use = use.T
    Z1[i, :] = desVector(use,ClassW1, ClassW2, ClassW3)  

levels = [0.9, 1.1, 1.9, 2.1, 2.9, 3.1]
plt.contourf(xx1, xx2, Z1, levels=levels, cmap=plt.cm.Paired)
plt.colorbar()
plt.show()
'''