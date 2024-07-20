import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def testCone(RotatedCone, Transobject):
    test = np.ones((len(RotatedCone), Transobject.shape[1]))
    test[0, :] = np.power(Transobject[0], 2)
    test[1, :] = np.power(Transobject[1], 2)
    test[2, :] = np.power(Transobject[2], 2)
    test[3, :] = np.multiply(Transobject[0], Transobject[1])
    test[4, :] = np.multiply(Transobject[1], Transobject[2])
    test[5, :] = np.multiply(Transobject[0], Transobject[2])
    res = np.matmul(RotatedCone,test)
    fgfbb=5

def plotSutface(Eqn):
    X = np.arange(-10,10,0.025)
    Y = np.arange(-10,10,0.025)
    X, Y = np.meshgrid(X, Y)
    Z_coeff = [Eqn[2], Eqn[4]*Y + Eqn[5]*X + Eqn[8],
               Eqn[0]*np.multiply(X,X) + Eqn[1]*np.multiply(Y,Y) + Eqn[3]*np.multiply(X,Y) + Eqn[6]*X + Eqn[7]*Y + Eqn[9]]
    Z = (-Z_coeff[1] + np.sqrt(np.power(Z_coeff[1],2) - 4*Z_coeff[0]*Z_coeff[2]))/(2*Z_coeff[0])
    Z_ = (-Z_coeff[1] - np.sqrt(np.power(Z_coeff[1],2) - 4*Z_coeff[0]*Z_coeff[2]))/(2*Z_coeff[0])
    np.append(X,X)
    np.append(Y,Y)
    np.append(Z,Z_)
    # Z = np.transpose(np.array([Z]))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xcolors = X - min(X.flat)
    xcolors = xcolors / max(xcolors.flat)
    surf = ax.plot_surface(X.transpose(), Y.transpose(), Z.transpose(), rstride=1, cstride=1, facecolors=cm.hot(xcolors),
                           linewidth=1)
    plt.show()

    fgssdf=5

def ProjectEllipse2Cone(EllipseCoeff, CamPlane, R, object_mat_wrt_Cam):

    # Ellipse Eqn: A X2 + B XY + C Y2 + D X + E Y + F = 0
    # Projected Cone Eqn: Af2 X2 + Cf2 Y2 + f Z2 + Bf2 XY + Ef YZ + Df XZ  = 0

    ############### This cone works with the inverse transformed and shifted part ###############################3
    coeff_cone_temp_1 = [EllipseCoeff[0] * CamPlane.f ** 2,
                         EllipseCoeff[2] * CamPlane.f ** 2,
                         EllipseCoeff[5],
                         EllipseCoeff[1] * CamPlane.f ** 2,
                         EllipseCoeff[4] * CamPlane.f,
                         EllipseCoeff[3] * CamPlane.f]
    coeff_cone_temp = coeff_cone_temp_1
    coeff_cone_temp.append(0); coeff_cone_temp.append(0); coeff_cone_temp.append(0); coeff_cone_temp.append(0)

    testCone(coeff_cone_temp_1, object_mat_wrt_Cam)

    ######################### Cone Shifting first followed by rotation #############################3

    A = EllipseCoeff[0]; B = EllipseCoeff[1]; C = EllipseCoeff[2]
    D = EllipseCoeff[3]; E = EllipseCoeff[4]; F = EllipseCoeff[5]
    f = CamPlane.f
    a = CamPlane.point[0]; b = CamPlane.point[1]; c = CamPlane.point[2]

    CoeffConeTrans = [A * f ** 2,
                         C * f ** 2,
                         F,
                         B * f ** 2,
                         E * f,
                         D * f,
                      -(2*a*A*f**2 + b*B*f**2 + c*D*f),
                      -(a*B*f**2 + 2*b*C*f**2 + c*E*f),
                      -(a*D*f + b*E*f + 2*c*F),
                      (A*f**2)*a**2 + b*a*B*f**2 + (C*f**2)*b**2 + D*f*a*c + E*f*b*c + F*C**2]

    # plotSutface(coeff_cone_temp)

    a = CoeffConeTrans[0]; b = CoeffConeTrans[1]; c = CoeffConeTrans[2]
    d = CoeffConeTrans[3]; e = CoeffConeTrans[4]; f = CoeffConeTrans[5]
    g = CoeffConeTrans[6]; h = CoeffConeTrans[7]; i = CoeffConeTrans[8]
    j = CoeffConeTrans[9]

    A = R[0, 0]; B = R[0, 1]; C = R[0, 2]
    D = R[1, 0]; E = R[1, 1]; F = R[1, 2]
    G = R[2, 0]; H = R[2, 1]; I = R[2, 2]



    ################################### Cone rotating first followed by shifting #################################3

    a = coeff_cone_temp_1[0]
    b = coeff_cone_temp_1[1]
    c = coeff_cone_temp_1[2]
    d = coeff_cone_temp_1[3]
    e = coeff_cone_temp_1[4]
    f = coeff_cone_temp_1[5]

    A = R[0,0]; B = R[0,1]; C = R[0,2]
    D = R[1,0]; E = R[1,1]; F = R[1,2]
    G = R[2,0]; H = R[2,1]; I = R[2,2]

    Rotated_cone = [a*A**2 + b*D**2 + c*G**2 + d*A*D + e*D*G + f*A*G,
                    a*B**2 + b*E**2 + c*H**2 + d*B*E + e*E*H + f*B*H,
                    a*C**2 + b*F**2 + c*I**2 + d*C*F + e*F*I + f*C*I,
                    2*a*A*B + 2*b*D*E + 2*c*G*H + d*(A*E+D*B) + e*(D*H+G*E) + f*(A*H+G*B),
                    2*a*B*C + 2*b*E*F + 2*c*H*I + d*(B*F+E*C) + e*(E*H+F*H) + f*(B*I+H*C),
                    2*a*A*C + 2*b*D*F + 2*c*G*I + d*(A*F+D*C) + e*(D*I+G*F) + f*(A*I+G*C)]

    # testCone(coeff_cone_temp_1, object)
    #
    TransPt = CamPlane.point
    #
    i = TransPt[0]; j = TransPt[1]; k = TransPt[2]
    A_ = Rotated_cone[0]; B_ = Rotated_cone[1]; C_ = Rotated_cone[2]
    D_ = Rotated_cone[3]; E_ = Rotated_cone[4]; F_ = Rotated_cone[5]

    point = CamPlane.point
    G = A_*point[0]**2 + B_*point[1]**2 + C_*point[2]**2 \
        + D_*point[0]*point[1] + E_*point[1]*point[2] + F_*point[0]*point[2]

    # Rotated_cone.append(-G)
    # Final_Cone = Rotated_cone

    Final_Cone = np.array([A_, B_, C_, D_, E_, F_,
                  -(2*A_*i + D_*j + F_*k),
                  -(2*B_*j + D_*i + E_*k),
                  -(2*C_*k + E_*j + F_*i),
                  A_*i**2 + B_*j**2 + C_*k**2 + D_*i*j + E_*j*k + F_*i*k])

    # Final_Cone = CoeffConeTrans

    test = [i**2, j**2, k**2, i*j, j*k, k*i, i, j, k, 1]
    res = np.multiply(Final_Cone, test)
    res = sum(res)
    return np.array(Final_Cone)



def CheckCircle(Cone, Pt):
    test = np.ones((Cone.shape[0], Pt.shape[1]))
    test[0, :] = np.power(Pt[0], 2)
    test[1, :] = np.power(Pt[1], 2)
    test[2, :] = np.power(Pt[2], 2)
    test[3, :] = np.multiply(Pt[0], Pt[1])
    test[4, :] = np.multiply(Pt[1], Pt[2])
    test[5, :] = np.multiply(Pt[0], Pt[2])
    test[6, :] = Pt[0]
    test[7, :] = Pt[1]
    test[8, :] = Pt[2]
    res = np.matmul(Cone,test)
    fkjhdkj = 1



    # Rotated_cone = 1
    #hi i am ankit
    #hi



