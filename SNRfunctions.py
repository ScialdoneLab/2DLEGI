#Compute the SNR for a generic cluster of cells given A CONNECTIVITY MATRIX, their coordinates, 
# a vector of parameters and their number
from numpy.linalg import inv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Take the extent of cell-cell contacts (which also has info on neighbour cells, so we don't need to pass
#the contact matrix)
def Matrix(ext):  #Connectivity matrix
    N = ext.shape[0]
    Conn=np.array([[0.0 for i in range(N)] for j in range(N)])
    for i in range(N):
        for j in range(N):
            if j!=i:
                if ext[i,j]!=0:
                    rho=ext[i,j] #Each pair of cells has a different value of rho
                else:
                    rho=0
                Conn[i,j]=-rho
                Conn[i,i]+=rho 
        Conn[i,i] += 1
    return Conn 
    
    
def ComputeLEGIMatrix(gammat,mu):
    # Copy the matrix of the exchange rates
    resc_gammat=gammat.copy()
    
    # Fill the diagonal with zeros
    np.fill_diagonal(resc_gammat,0)
    
    # Define the communication matrix (rho=gamma/mu)
    resc_gammat/=mu
    
    # Compute the connectivity matrix for the LEGI model
    LEGImatrix=Matrix(resc_gammat)
    
    return LEGImatrix;   

def LEGISNR(gammat,s,beta,mu,g,a,cbar,mode,num):
	
	ConnMatrix=ComputeLEGIMatrix(gammat,mu)
	
	#Get cell coordinates
	x_coords=np.array(s.face_df['x'])
	y_coords=np.array(s.face_df['y'])
	
	G=beta/mu
	
	xy_coords=np.column_stack((x_coords,y_coords))
	
	#Find the centroid of the configuration
	kmeans_s = KMeans(n_clusters=1, random_state=0).fit(xy_coords)
	centre_s = kmeans_s.cluster_centers_[0]
	
	P=centre_s[0]
	Q=centre_s[1]    
	
	# Define the array of the angles
	theta_arr=np.linspace(0.0,2.0*math.pi,endpoint=True,num=num)
	
	cos_theta = np.cos(theta_arr)
	sin_theta = np.sin(theta_arr)
	
	#Apply the rotation matrix to the initial coordinates
	new_cell_coords=np.array([[[0.0 for i in range(2)] for j in range(s.Nf)] for k in range(len(theta_arr))])
	
	new_cell_coords[:,:,0]=(xy_coords[np.newaxis,:,0]-P)*cos_theta[:,np.newaxis] + (xy_coords[np.newaxis,:,1]-Q)*sin_theta[:,np.newaxis]+P
	new_cell_coords[:,:,1]=-(xy_coords[np.newaxis,:,0]-P)*sin_theta[:,np.newaxis] + (xy_coords[np.newaxis,:,1]-Q)*cos_theta[:,np.newaxis]+Q
	
	#Define the vector of concentrations sensed by the cells
	cn=cbar+g*new_cell_coords[:,:,0]
	
	#Define the communication kernel for all the cells (inverse of the connectivity matrix)
	Kn=inv(ConnMatrix)
	
	#Mean and fluctuations of the local reporter
	xN=G*(a**3)*cn
	dxN=G*G*(a**3)*cn
	
	#Mean and fluctuations of the global reporter
	yN=G*(a**3)*np.sum(Kn[np.newaxis,:,:]*cn[:,np.newaxis,:],axis=2)
	dyN=G*G*(a**3)*np.sum(Kn[np.newaxis,:,:]*Kn[np.newaxis,:,:]*cn[:,np.newaxis,:],axis=2)
	
	#Mean readout and fluctuations of the readout
	delta=xN-yN
	ddelta=xN+dxN+yN+dyN-(2.0*G*G*(a**3)*np.diag(Kn)[np.newaxis,:]*cn)
	
	#Compute the SNR
	SNR=np.sqrt((delta*delta)/ddelta)
	
	#For each angle, select the cell with largest coordinate in direction of the gradient
	max_cell_indices=np.argmax(new_cell_coords[:,:,0],axis=1)
	
	SNR_edge=SNR[np.arange(SNR.shape[0]), max_cell_indices]
	delta_edge=delta[np.arange(SNR.shape[0]), max_cell_indices]
	ddelta_edge=ddelta[np.arange(SNR.shape[0]), max_cell_indices]
	
	if mode=='mean_edge':
		return SNR_edge,delta_edge,ddelta_edge;
	elif mode=='edge':
		return SNR[0,max_cell_indices[0]];
	elif mode=='all':
		# Return the mean SNR over the angle theta for each cell in the configuration
		return np.mean(SNR,axis=0)
