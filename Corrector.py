import numpy as np
import sys
import math
from numpy.linalg import inv



class solver(object):


    def __init__(self):

        data_file = "data_file_3D_Corrector.txt"
        self.data_file_3D = open(data_file, "w")

        # hello
        self.NN=5.0
        self.gamma=2.0
        self.H=3.2
        self.beta=0.731
        self.Q1=0.648
        self.Q2=0.441
        self.XA=0.52
        self.YA=0.62
        self.XC=0.12
        self.YC=0.303
        self.YH=0.219
        self.gammadot=0.0005
        self.OmegaB=-1.0
        self.theta=np.pi/6 #initial conditions
        self.phi=np.pi/2
        self.psi=0
        self.Bx=0.0
        self.By=30.0
        self.Bz=0.0
        self.dt=0.1
        self.Tmax=950.0
        self.numstep=int(self.Tmax/self.dt)
        self.theta_old=self.theta
        self.phi_old=self.phi
        self.psi_old=self.psi
        self.Bx_old=self.Bx
        self.By_old=self.By
        self.Bz_old=self.Bz
        self.trajectory_old=np.array([self.Bx_old,self.By_old,self.Bz_old])
        self.angles_old= np.array([self.theta_old,self.phi_old,self.psi_old])
        self.data_file_3D.write("0.0" + " " +str(self.theta) + " " + str(self.phi) + " " + str(self.psi) + " " + str(self.Bx) + " " + str(self.By) + " " + str(self.Bz)+ "\n")

        #self.A11 = np.cos(self.theta)*np.cos(self.phi)*np.cos(self.psi)-np.sin(self.phi)*np.sin(self.psi)
        #self.A22 = -np.cos(self.theta)*np.cos(self.phi)*np.sin(self.psi)-np.sin(self.phi)*np.cos(self.psi)
        #self.A33 = np.cos(self.phi)*np.sin(self.theta)
        #self.C11 = np.cos(self.theta)*np.cos(self.psi)*np.sin(self.phi)+np.cos(self.theta)*np.sin(self.psi)
        #self.C22 = np.cos(self.phi)*np.cos(self.psi)-np.cos(self.theta)*np.sin(self.phi)*np.sin(self.psi)
        #self.C33 = np.sin(self.phi)*np.sin(self.theta)
        #self.A1=0.1
        #self.A2=-0.2
        #self.A3=0.4
        #self.C1=0.5
        #self.C2=-0.7
        #self.C3=0.6
        self.angles = np.array([self.theta,self.phi,self.psi])
        self.trajectory = np.array([self.Bx,self.By,self.Bz])
        self.A1 = np.cos(self.angles[0])*np.cos(self.angles[1])*np.cos(self.angles[2])-np.sin(self.angles[1])*np.sin(self.angles[2])
        self.A2 = -np.cos(self.angles[0])*np.cos(self.angles[1])*np.sin(self.angles[2])-np.sin(self.angles[1])*np.cos(self.angles[2])
        self.A3 = np.cos(self.angles[1])*np.sin(self.angles[0])
        self.C1 = np.cos(self.angles[0])*np.cos(self.angles[2])*np.sin(self.angles[1])+np.cos(self.angles[1])*np.sin(self.angles[2])
        self.C2 = np.cos(self.angles[1])*np.cos(self.angles[2])-np.cos(self.angles[0])*np.sin(self.angles[1])*np.sin(self.angles[2])
        self.C3 = np.sin(self.angles[1])*np.sin(self.angles[0])

        #matrix with constants
        self.Matrix=np.array([[-self.Q1*self.YA-self.gamma+0.5*(self.gamma-1)*(np.sin(self.beta))**2, 0, 0, -3*(self.gamma-1) *(np.sin(self.beta))**2/(8*np.pi), -self.H*self.Q1*self.YA,0],
       [0,-self.Q1*self.YA-self.gamma+0.5*(self.gamma-1)*(np.sin(self.beta))**2,0, self.H*self.Q1*self.YA,-(self.gamma-1)*((np.sin(self.beta))**2)/(8*np.pi),0],
       [0,0,-self.Q1*self.XA+0.5*(-self.gamma-1)+0.5*(self.gamma-1)*np.cos(2*self.beta),0,-(self.gamma-1)*(-1)**(self.NN+1) * np.sin(2*self.beta)/(4*np.pi),(self.gamma-1)/(4*np.pi)*(1-np.cos(2*self.beta))],
       [-3*(self.gamma-1)/(8*np.pi)*(np.sin(self.beta))**2,self.H*self.Q1*self.YA,0,-(self.H**2)*self.Q1*self.YA -self.Q2*self.YC-(self.NN**2 * self.gamma/12)+(self.NN**2)*(self.gamma-1)*(np.sin(self.beta))**2/24+(5*(self.gamma-1)*(np.sin(self.beta))**2)/(16*(np.pi)**2)-(self.gamma*(np.tan(self.beta))**2)/(8*(np.pi)**2),0,0],
       [-self.H*self.Q1*self.YA,-(self.gamma-1)*(np.sin(self.beta))**2/(8*np.pi),-(self.gamma-1)*(-1)**(1+self.NN)*np.sin(2*self.beta)/(4*np.pi),0,-(self.H**2)*self.Q1*self.YA-self.Q2*self.YC-self.NN**2*self.gamma/12-(self.gamma-1)*(np.sin(self.beta))**2/(16*(np.pi)**2)+self.NN**2*(self.gamma-1)*(np.sin(self.beta))**2/(24)-self.gamma*(np.tan(self.beta))**2/(8*np.pi**2),-self.gamma*(-1)**(self.NN) * np.tan(self.beta)/(4*np.pi**2)+(self.gamma-1)*(-1)**(self.NN)*(np.sin(self.beta))**2*np.tan(self.beta)/(4*np.pi**2)],
       [0,0,(self.gamma-1)*np.sin(2*self.beta)*np.tan(self.beta)/(4*np.pi),0,(self.gamma+1)*np.tan(self.beta)*(-1)**(self.NN+1)/(8*np.pi**2)-(self.gamma-1)*np.tan(self.beta)*(-1)**(self.NN)*np.cos(2*self.beta)/(8*np.pi**2),-self.Q2*self.XC-(self.gamma-1)*np.tan(self.beta)*np.sin(2*self.beta)/(8*np.pi**2)-(np.tan(self.beta))**2/(4*np.pi**2)]])


        #rhs check for the minus sign
        self.rhs = -np.array([self.A1 *self.C3*self.H*self.Q1*self.YA*self.gammadot+self.A3*self.C2*(self.gamma-1)* self.gammadot*np.sin(self.beta)**2/(4*np.pi) - self.A2*self.C3*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/(8*np.pi)-self.A3*self.C3*(self.gamma-1)*(-1)**self.NN*self.gammadot*np.sin(2*self.beta)/(4*np.pi)+self.A1*self.Q1*self.YA*self.gammadot*self.trajectory[1]+self.A1*self.gamma*self.gammadot*self.trajectory[1]-0.5*self.A1*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1], self.A2 *self.C3*self.H*self.Q1*self.YA*self.gammadot
                -self.A3*self.C1*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/(4*np.pi)- self.A1*self.C3*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/(8*np.pi)+self.A2*self.Q1*self.YA*self.gammadot*self.trajectory[1]+self.A2*self.gamma*self.gammadot*self.trajectory[1]-0.5*self.A2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1],self.A3 *self.C3*self.H*self.Q1*self.XA*self.gammadot
                 -self.A2*self.C1*(self.gamma-1)*self.gammadot/(8*np.pi)+self.A1*self.C2*(self.gamma-1)*self.gammadot/(8*np.pi)+self.A2*self.C1*(self.gamma-1)*self.gammadot*np.cos(2*self.beta)/(8*np.pi)-self.A1*self.C2*(self.gamma-1)*self.gammadot*np.cos(2*self.beta)/(8*np.pi)+self.A1*self.C3*(-1)**(self.NN+1)*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)/(4*np.pi)+self.A3*self.Q1*self.XA*self.gammadot*self.trajectory[1]-0.5*self.A3*(-1-self.gamma)*self.gammadot*self.trajectory[1]-0.5*self.A3*(-1+self.gamma)*self.gammadot*np.cos(2*self.beta)*self.trajectory[1],
                 -self.A2*self.H*self.Q1*self.YA*self.gammadot*(self.C3*self.H+self.trajectory[1])+0.5*(self.A3*self.C2-self.A2*self.C3)*self.Q2*self.YC*self.gammadot-0.5*(self.A3*self.C2+self.A2*self.C3)*self.Q2*self.YH*self.gammadot-self.A2*self.C3*self.NN**2*self.gamma*self.gammadot/12+self.A2*self.C3*self.NN**2*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/24-3*self.A3*self.C2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(16*np.pi**2)+self.A2*self.C3*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(8*np.pi**2)+(self.A1*self.C1-self.A2*self.C2)*(-1)**self.NN*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*np.tan(self.beta)/(12*np.pi**2)+self.A3*self.C2*self.gamma*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)+(self.gammadot*(-1)**self.NN*(3*self.A3*self.C3*(self.gamma-1)*np.sin(2*self.beta)+2*(self.A2*self.C2-self.A3*self.C3)*self.gamma*np.tan(self.beta)))/(8*np.pi**2)+3*self.A1*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1]/(8*np.pi),
                 self.A1*self.H*self.Q1*self.YA*self.gammadot*(self.C3*self.H+self.trajectory[1])+0.5*(-self.A3*self.C1+self.A1*self.C3)*self.Q2*self.YC*self.gammadot+0.5*(self.A3*self.C1+self.A1*self.C3)*self.Q2*self.YH*self.gammadot+self.A1*self.C3*self.NN**2*self.gamma*self.gammadot/12+self.A3*self.C1*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(16*np.pi**2)+self.A1*self.C3*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(8*np.pi**2)-self.A1*self.C3*self.NN**2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(24)-self.A1*self.C2*(-1)**self.NN*self.gamma*self.gammadot*np.tan(self.beta)/(4*np.pi**2)-self.A2*self.C1*(-1)**self.NN*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*np.tan(self.beta)/(12*np.pi**2)+self.A1*self.C2*(-1)**self.NN*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*np.tan(self.beta)/(6*np.pi**2)-self.A3*self.C1*self.gamma*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)+self.A2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1]/(8*np.pi)+self.A3*(-1)**(self.NN+1)*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)*self.trajectory[1]/(4*np.pi),
                 0.5*(self.A2*self.C1-self.A1*self.C2)*self.Q2*self.XC*self.gammadot-self.Q2*self.XC*self.OmegaB+self.A1*self.C3*(-1)**self.NN*(1+self.gamma)*self.gammadot*np.tan(self.beta)/(8*np.pi**2)+self.A1*self.C3*(-1)**self.NN*(-1+self.gamma)*self.gammadot*np.cos(2*self.beta)*np.tan(self.beta)/(8*np.pi**2)+self.A2*self.C1*(-1+self.gamma)*self.gammadot*np.sin(2*self.beta)*np.tan(self.beta)/(16*np.pi**2)-self.A1*self.C2*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)*np.tan(self.beta)/(16*np.pi**2)+self.A2*self.C1*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)-self.A1*self.C2*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)-self.A3*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)*np.tan(self.beta)*self.trajectory[1]/(4*np.pi)])



        self.w=np.dot(inv(self.Matrix),self.rhs)


        #DTranspose
        self.DTranspose=np.array([[np.cos(self.angles[0])*np.cos(self.angles[1])*np.cos(self.angles[2])-np.sin(self.angles[1])*np.sin(self.angles[2]), -np.cos(self.angles[2])*np.sin(self.angles[1])-np.cos(self.angles[0])*np.cos(self.angles[1])*np.sin(self.angles[2]), np.cos(self.angles[1])*np.sin(self.angles[0])],
                                 [np.cos(self.angles[0])*np.cos(self.angles[2])*np.sin(self.angles[1])+np.cos(self.angles[1])*np.sin(self.angles[2]), np.cos(self.angles[1])*np.cos(self.angles[2])-np.cos(self.angles[0])*np.sin(self.angles[1])*np.sin(self.angles[2]), np.sin(self.angles[0])*np.sin(self.angles[1])],
                                 [-np.cos(self.angles[2])*np.sin(self.angles[0]),np.sin(self.angles[0])*np.sin(self.angles[2]),np.cos(self.angles[0])]])



        #for the equations of theta,phi,psi

        #self.angles_matrix2 = np.array([[np.sin(self.psi),np.cos(self.psi),0],
                                           #[-np.cos(self.psi)/np.sin(self.theta),np.sin(self.psi)/np.sin(self.theta),0],
                                           #[np.cos(self.psi)*np.cos(self.theta)/np.sin(self.theta),-np.sin(self.psi)*np.cos(self.theta)/np.sin(self.theta),1]])

        self.angles_matrix = np.array([[np.sin(self.angles[2]),np.cos(self.angles[2]),0],
                                           [-np.cos(self.angles[2])/np.sin(self.angles[0]),np.sin(self.angles[2])/np.sin(self.angles[0]),0],
                                           [np.cos(self.angles[2])*np.cos(self.angles[0])/np.sin(self.angles[0]),-np.sin(self.angles[2])*np.cos(self.angles[0])/np.sin(self.angles[0]),1]])

        #star
        self.star()
        self.A_C_update()

    def A_C_update(self):
        self.A1 = np.cos(self.angles[0])*np.cos(self.angles[1])*np.cos(self.angles[2])-np.sin(self.angles[1])*np.sin(self.angles[2])
        self.A2 = -np.cos(self.angles[0])*np.cos(self.angles[1])*np.sin(self.angles[2])-np.sin(self.angles[1])*np.cos(self.angles[2])
        self.A3 = np.cos(self.angles[1])*np.sin(self.angles[0])
        self.C1 = np.cos(self.angles[0])*np.cos(self.angles[2])*np.sin(self.angles[1])+np.cos(self.angles[1])*np.sin(self.angles[2])
        self.C2 = np.cos(self.angles[1])*np.cos(self.angles[2])-np.cos(self.angles[0])*np.sin(self.angles[1])*np.sin(self.angles[2])
        self.C3 = np.sin(self.angles[1])*np.sin(self.angles[0])
        self.rhs = -np.array([self.A1 *self.C3*self.H*self.Q1*self.YA*self.gammadot+self.A3*self.C2*(self.gamma-1)* self.gammadot*np.sin(self.beta)**2/(4*np.pi) - self.A2*self.C3*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/(8*np.pi)-self.A3*self.C3*(self.gamma-1)*(-1)**self.NN*self.gammadot*np.sin(2*self.beta)/(4*np.pi)+self.A1*self.Q1*self.YA*self.gammadot*self.trajectory[1]+self.A1*self.gamma*self.gammadot*self.trajectory[1]-0.5*self.A1*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1], self.A2 *self.C3*self.H*self.Q1*self.YA*self.gammadot
                        -self.A3*self.C1*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/(4*np.pi)- self.A1*self.C3*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/(8*np.pi)+self.A2*self.Q1*self.YA*self.gammadot*self.trajectory[1]+self.A2*self.gamma*self.gammadot*self.trajectory[1]-0.5*self.A2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1],self.A3 *self.C3*self.H*self.Q1*self.XA*self.gammadot
                         -self.A2*self.C1*(self.gamma-1)*self.gammadot/(8*np.pi)+self.A1*self.C2*(self.gamma-1)*self.gammadot/(8*np.pi)+self.A2*self.C1*(self.gamma-1)*self.gammadot*np.cos(2*self.beta)/(8*np.pi)-self.A1*self.C2*(self.gamma-1)*self.gammadot*np.cos(2*self.beta)/(8*np.pi)+self.A1*self.C3*(-1)**(self.NN+1)*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)/(4*np.pi)+self.A3*self.Q1*self.XA*self.gammadot*self.trajectory[1]-0.5*self.A3*(-1-self.gamma)*self.gammadot*self.trajectory[1]-0.5*self.A3*(-1+self.gamma)*self.gammadot*np.cos(2*self.beta)*self.trajectory[1],
                         -self.A2*self.H*self.Q1*self.YA*self.gammadot*(self.C3*self.H+self.trajectory[1])+0.5*(self.A3*self.C2-self.A2*self.C3)*self.Q2*self.YC*self.gammadot-0.5*(self.A3*self.C2+self.A2*self.C3)*self.Q2*self.YH*self.gammadot-self.A2*self.C3*self.NN**2*self.gamma*self.gammadot/12+self.A2*self.C3*self.NN**2*(self.gamma-1)*self.gammadot*(np.sin(self.beta))**2/24-3*self.A3*self.C2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(16*np.pi**2)+self.A2*self.C3*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(8*np.pi**2)+(self.A1*self.C1-self.A2*self.C2)*(-1)**self.NN*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*np.tan(self.beta)/(12*np.pi**2)+self.A3*self.C2*self.gamma*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)+(self.gammadot*(-1)**self.NN*(3*self.A3*self.C3*(self.gamma-1)*np.sin(2*self.beta)+2*(self.A2*self.C2-self.A3*self.C3)*self.gamma*np.tan(self.beta)))/(8*np.pi**2)+3*self.A1*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1]/(8*np.pi),
                         self.A1*self.H*self.Q1*self.YA*self.gammadot*(self.C3*self.H+self.trajectory[1])+0.5*(-self.A3*self.C1+self.A1*self.C3)*self.Q2*self.YC*self.gammadot+0.5*(self.A3*self.C1+self.A1*self.C3)*self.Q2*self.YH*self.gammadot+self.A1*self.C3*self.NN**2*self.gamma*self.gammadot/12+self.A3*self.C1*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(16*np.pi**2)+self.A1*self.C3*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(8*np.pi**2)-self.A1*self.C3*self.NN**2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2/(24)-self.A1*self.C2*(-1)**self.NN*self.gamma*self.gammadot*np.tan(self.beta)/(4*np.pi**2)-self.A2*self.C1*(-1)**self.NN*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*np.tan(self.beta)/(12*np.pi**2)+self.A1*self.C2*(-1)**self.NN*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*np.tan(self.beta)/(6*np.pi**2)-self.A3*self.C1*self.gamma*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)+self.A2*(self.gamma-1)*self.gammadot*np.sin(self.beta)**2*self.trajectory[1]/(8*np.pi)+self.A3*(-1)**(self.NN+1)*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)*self.trajectory[1]/(4*np.pi),
                         0.5*(self.A2*self.C1-self.A1*self.C2)*self.Q2*self.XC*self.gammadot-self.Q2*self.XC*self.OmegaB+self.A1*self.C3*(-1)**self.NN*(1+self.gamma)*self.gammadot*np.tan(self.beta)/(8*np.pi**2)+self.A1*self.C3*(-1)**self.NN*(-1+self.gamma)*self.gammadot*np.cos(2*self.beta)*np.tan(self.beta)/(8*np.pi**2)+self.A2*self.C1*(-1+self.gamma)*self.gammadot*np.sin(2*self.beta)*np.tan(self.beta)/(16*np.pi**2)-self.A1*self.C2*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)*np.tan(self.beta)/(16*np.pi**2)+self.A2*self.C1*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)-self.A1*self.C2*self.gammadot*np.tan(self.beta)**2/(8*np.pi**2)-self.A3*(self.gamma-1)*self.gammadot*np.sin(2*self.beta)*np.tan(self.beta)*self.trajectory[1]/(4*np.pi)])
        self.DTranspose=np.array([[np.cos(self.angles[0])*np.cos(self.angles[1])*np.cos(self.angles[2])-np.sin(self.angles[1])*np.sin(self.angles[2]), -np.cos(self.angles[2])*np.sin(self.angles[1])-np.cos(self.angles[0])*np.cos(self.angles[1])*np.sin(self.angles[2]), np.cos(self.angles[1])*np.sin(self.angles[0])],
                                 [np.cos(self.angles[0])*np.cos(self.angles[2])*np.sin(self.angles[1])+np.cos(self.angles[1])*np.sin(self.angles[2]), np.cos(self.angles[1])*np.cos(self.angles[2])-np.cos(self.angles[0])*np.sin(self.angles[1])*np.sin(self.angles[2]), np.sin(self.angles[0])*np.sin(self.angles[1])],
                                 [-np.cos(self.angles[2])*np.sin(self.angles[0]),np.sin(self.angles[0])*np.sin(self.angles[2]),np.cos(self.angles[0])]])
        self.angles_matrix = np.array([[np.sin(self.angles[2]),np.cos(self.angles[2]),0],
                                           [-np.cos(self.angles[2])/np.sin(self.angles[0]),np.sin(self.angles[2])/np.sin(self.angles[0]),0],
                                           [np.cos(self.angles[2])*np.cos(self.angles[0])/np.sin(self.angles[0]),-np.sin(self.angles[2])*np.cos(self.angles[0])/np.sin(self.angles[0]),1]])
        self.w=np.dot(inv(self.Matrix),self.rhs)


    def update(self):
        for i in range(1, self.numstep):
            self.angles = self.angles_old + self.dt * np.dot( self.angles_matrix , self.w[3:6])
            self.trajectory = self.trajectory_old + self.dt * np.dot(self.DTranspose, self.w[0:3])
            self.angles_old=self.angles
            self.trajectory_old=self.trajectory
            self.star()
            self.A_C_update()
            if i % 10  == 0:
                self.write_data(i)
        print(self.trajectory)
        self.data_file_3D.close()

    def star(self):
        self.angles = self.angles_old + self.dt * 0.5 * np.dot(self.angles_matrix, self.w[3:6])
        self.trajectory = self.trajectory_old + self.dt * 0.5 * np.dot(self.DTranspose, self.w[0:3])


    def write_data(self,i):
        self.data_file_3D.write(str(i*self.dt) + " ")
        self.data_file_3D.write('%.7g' % self.angles[0] + " ")
        self.data_file_3D.write('%.7g' % self.angles[1] + " ")
        self.data_file_3D.write('%.7g' % self.angles[2] + " ")
        self.data_file_3D.write('%.7g' % self.trajectory[0] + " ")
        self.data_file_3D.write('%.7g' % self.trajectory[1] + " ")
        self.data_file_3D.write('%.7g' % self.trajectory[2] + ("\n"))
        self.data_file_3D.flush()
#def main():
    #system = solver()
    #print(system.A1)
    #print(system.A2)
    #print(system.A3)
    #print(system.C1)
    #print(system.rhs1)
    #print(system.rhs2)
#main()
