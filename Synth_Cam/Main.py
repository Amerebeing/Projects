# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from mpl_toolkits import mplot3d
import numpy as np
from numpy import pi
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

class CamPlane:
    def __init__(self,point, vector, f):
        self.point = point
        self.vector = np.divide(vector, np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2))
        self.cam_plane_point = self.point - f*self.vector
        self.f = f

def CamSim_(cam_plane, object):
    # a = cam_plane.point[0]
    x_ = np.divide(object[0]-cam_plane.point[0],object[2]-cam_plane.point[2])
    y_ = np.divide(object[1]-cam_plane.point[1],object[2]-cam_plane.point[2])
    fig = plt.figure()
    # ax = plt.axes(projection='2d')
    plt.plot(x_, y_, color='green')
    fig.show()
    return x_, y_

def testCamSim(origPts, CamPlane, TransPoints):
    origPts[0, :] = origPts[0, :] - CamPlane.point[0]
    origPts[1, :] = origPts[1, :] - CamPlane.point[1]
    origPts[2, :] = origPts[2, :] - CamPlane.point[2]
    dist = np.power(TransPoints,2)#np.sum(np.power(origPts[0],2), np.power(origPts[1],2))
    dist = np.sum(dist,axis=0)

    # sub = TransPoints.copy()

    dist_ = np.power(TransPoints,2)
    dist_ = np.sum(dist_,axis=0)
    diff = dist-dist_
    a=5

def CamSim(cam_plane, object):
    start_vec = np.array([0,0,1])
    z_final = cam_plane.vector
    x_final = np.cross(z_final,[0,0,1])

    # x_final = np.cross([0,0,1],z_final)
    x_final = x_final/norm(x_final)
    y_final = np.cross(z_final,x_final)
    y_final = y_final/norm(y_final)

    n_orig = np.eye(3)
    R = np.array([x_final,y_final,z_final]).transpose()
    R_1 = inv(R)
    object_mat = np.array(object)
    object_mat_wrt_Cam = np.matmul(R_1,object_mat)
    object_mat_wrt_Cam[0,:] = object_mat_wrt_Cam[0,:] - cam_plane.point[0]
    object_mat_wrt_Cam[1, :] = object_mat_wrt_Cam[1, :] - cam_plane.point[1]
    object_mat_wrt_Cam[2, :] = object_mat_wrt_Cam[2, :] - cam_plane.point[2]

    testCamSim(object_mat, cam_plane, object_mat_wrt_Cam)

    x_ = cam_plane.f*np.divide(object_mat_wrt_Cam[0,:],object_mat_wrt_Cam[2,:])
    y_ = cam_plane.f*np.divide(object_mat_wrt_Cam[1,:],object_mat_wrt_Cam[2,:])

    fig = plt.figure()
    # ax = plt.axes(projection='2d')
    plt.plot(x_, y_, color='green')
    fig.show()

    return x_,y_, R, object_mat_wrt_Cam
    # return np.transpose(x_), np.transpose(y_)

    # fig = plt.figure()
    # # ax = plt.axes(projection='2d')
    # plt.plot(x_, y_, color='green')
    # fig.show()
    # fig

    # test = np.matmul(R_1,R)

    # ang_x = np.arcsin(np.dot(cam_plane.vector, [1, 0, 0]) / 1)
    # Rx = np.eye(3)
    # Rx[1,1] = np.cos(np.pi/2)
    # Rx[1,2] = -np.sin(np.pi/2)
    # Rx[2,1] = np.sin(np.pi/2)
    # Rx[2,2] = np.cos(np.pi/2)
    # # Rx[1, 1] = np.cos(ang_x)
    # # Rx[1, 2] = -np.sin(ang_x)
    # # Rx[2, 1] = np.sin(ang_x)
    # # Rx[2, 2] = np.cos(ang_x)
    # x_temp = np.matmul(Rx, np.transpose([1,0,0]))
    # y_temp = np.matmul(Rx, np.transpose([0,1,0]))
    # z_temp = np.matmul(Rx, np.transpose([0,0,1]))
    # test = np.matmul(start_vec,Rx)
    # ang_x = np.arccos(np.dot(cam_plane.vector, x_temp) / 1)
    # ang_y = np.arccos(np.dot(cam_plane.vector, y_temp) / 1)#*180/np.pi
    # ang_z = np.arccos(np.dot(cam_plane.vector, z_temp) / 1)
    # proj_vec_yz = [0, cam_plane.vector[1], cam_plane.vector[2]]
    # ang1 = np.arccos(np.dot(proj_vec_yz,z_temp)/np.linalg.norm(proj_vec_yz))
    # Rx1 = np.eye(3)
    # Rx1[1, 1] = np.cos(ang1)
    # Rx1[1, 2] = -np.sin(ang1)
    # Rx1[2, 1] = np.sin(ang1)
    # Rx1[2, 2] = np.cos(ang1)
    # x_temp1 = np.matmul(Rx1, x_temp)
    # y_temp1 = np.matmul(Rx1, y_temp)
    # z_temp1 = np.matmul(Rx1, z_temp)
    #
    # # proj_vec_xy = [cam_plane.vector[0], cam_plane.vector[1], 0]
    # ang2 = np.arccos(np.dot(cam_plane.vector,z_temp1)/1)
    # Ry = np.eye(3)
    # Ry[0, 0] = np.cos(ang2)
    # Ry[0, 2] = np.sin(ang2)
    # Ry[2, 0] = -np.sin(ang2)
    # Ry[2, 2] = np.cos(ang2)
    # x_temp2 = np.matmul(Ry, x_temp1)
    # y_temp2 = np.matmul(Ry, y_temp1)
    # z_temp2 = np.matmul(Ry, z_temp1)
    # n = np.linalg.norm(z_temp2)
    #
    # Ry = np.eye(3)
    # Ry[0,0] = np.cos(ang_y)
    # Ry[0,2] = np.sin(ang_y)
    # Ry[2,0] = -np.sin(ang_y)
    # Ry[2,2] = np.cos(ang_y)
    #
    # ang_z = np.arcsin(np.dot(cam_plane.vector,[0,0,1]))
    # Rz = np.eye(3)
    # Rz[0,0] = np.cos(ang_z)
    # Rz[0,1] = -np.sin(ang_z)
    # Rz[1,0] = np.sin(ang_z)
    # Rz[1,1] = np.cos(ang_z)
    #
    # test1 = np.matmul(test,Ry)
    # test2 = np.matmul(test1,Rz)
    # R = np.matmul(np.matmul(Rz,Ry),Rx)
    # # R = np.matmul(Rz,Ry)
    # test = np.matmul(R,np.transpose(start_vec))
    # a=5



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


    zdata = 3#15 * np.random.random(100)
    theta = np.linspace(-pi,pi,500)
    theta = np.append(theta, theta[::-1])
    xdata = 5*np.cos(theta)
    ydata = 5*np.sin(theta)
    zdata = 0*np.ones(ydata.shape)
    cam1_plane = CamPlane([5,10,20], [-5,-10,-20], 1)
    # cam1_plane = CamPlane([0, 10, 20], [0, -10, -20], 0.05)
    # cam1_plane = CamPlane([1, 0, 10], [1, 0, 10], 0.05)
    x1, y1, R1, object_mat_wrt_Cam = CamSim(cam1_plane,[xdata,ydata,zdata])
    import Ellipse
    import Cone
    coeff_ellipse_1 = Ellipse.FindEllipseCoeff([x1,y1])

    Cone1 = Cone.ProjectEllipse2Cone(coeff_ellipse_1, cam1_plane, R1, object_mat_wrt_Cam)

    Cone.CheckCircle(Cone1, np.array([xdata,ydata,zdata]))

    # Cone: Af2 X2 + Cf2 Y2 + f Z2 + Bf2 XY + Ef YZ + Df XZ  = 0
    coeff_cone_temp_1 = [coeff_ellipse_1[0] * cam1_plane.f ** 2,
                    coeff_ellipse_1[2] * cam1_plane.f ** 2,
                    coeff_ellipse_1[5],
                    coeff_ellipse_1[1] * cam1_plane.f ** 2,
                    coeff_ellipse_1[4] * cam1_plane.f,
                    coeff_ellipse_1[3] * cam1_plane.f]

    # coeff_cone_1 =

    # xdata = np.append(xdata,xdata[-2::])
    # np.sin(zdata) + 0.1 * np.random.randn(100)
    # ydata = np.sqrt(25-np.square(xdata))#np.cos(zdata) + 0.1 * np.random.randn(100)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, color='green')
    plt.savefig(r'D:\renishaw\syn_cam\3d_scatter_plot.png')
    fig.show()
    fig

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
