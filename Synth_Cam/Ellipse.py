import numpy as np
from numpy.linalg import solve
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures




def FindEllipseCoeff(points):
    x = points[0]
    y = points[1]
    x2 = np.multiply(x,x)
    y2 = np.multiply(y,y)
    xy = np.multiply(x,y)
    Mat = np.ones((6, x.shape[0]))
    Mat[0,:] = x2
    Mat[1,:] = xy
    Mat[2,:] = y2
    Mat[3,:] = x
    Mat[4,:] = y

    id = np.arange(0,x.shape[0],int(x.shape[0]/5))
    id = id[0:5]
    M = Mat[:,id]
    Coeff_temp = solve(M[1:6,:].transpose(), -M[0,:].transpose())
    Coeff = np.zeros((6,))
    Coeff[0] = 1
    Coeff[1::] = Coeff_temp

    # Coeff[5,:] = Coeff[5,:] - 1

    test = np.matmul(Mat.transpose(),Coeff)

    # # create linear regression object
    # reg = linear_model.LinearRegression()
    # # boston = datasets.load_boston(return_X_y=False)
    # # train the model using the training sets
    # reg.fit(Mat[1:6,:].transpose(), -Mat[0,:].transpose())
    # Coeff = np.zeros((6,))
    # Coeff[0] = 1
    # Coeff[1::] = reg.coef_
    # test = np.matmul(Mat.transpose(), Coeff)

    # poly = PolynomialFeatures(degree=2)
    # poly.fit(Mat[3:5,:], np.zeros((x.shape[0],1)))
    # # X_poly = poly.fit_transform(x)
    return Coeff
    # a = 5