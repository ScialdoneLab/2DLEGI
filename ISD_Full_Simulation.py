#!/usr/bin/env python
# coding: utf-8

# # Simulating the exchange of the LEGI global reporter in the Lateral Intercellular Space (LIS)

# Jonathan Fiorentino, 30/10/2020, Munich

import pickle
import numpy as np
import pandas as pd
import ipyvolume as ipv
import matplotlib
import matplotlib.pylab as plt
import math
import seaborn as sns
from tqdm import tqdm
import time
from tyssue import config
from tyssue import Sheet
from tyssue.io import hdf5

# Python module with all the functions needed for the simulations
from sim_functions import *
import os

# Total time.
T = 50.0
# Number of steps.
N = 1500
# Time step size
dt = T/N

#Array of the times
sim_time=np.array([0]+list(np.arange(dt,T+dt,dt)))

#Number of trajectories
N_traj=5000

# Number of cores to be used
n_cores=15

# Identifier for different number of cells (hexa1 = 7 cells etc )
stages=['hexa1_','hexa2_','hexa3_','hexa4_','hexa5_','hexa6_']

# Values of D (diffusion coefficient of the LEGI global reporter) and lambda (internalization rate)
params=[[1000.0,1.0],[10.0,1.0]]

# 1 set of configurations includes configurations with 7,19,37,61,91,127 cells and mean polygon number 5,5.25,5.5,5.75,6
# We use 10 sets to ensure robustness
sets=['set1','set2','set3','set4','set5','set6','set7','set8','set9','set10']

for s in sets:
	for p in params:
		D=p[0]
		# The Wiener process parameter.
		delta = np.sqrt(2.0*D)
		lamb=p[1]
		# CHANGE WITH YOUR FOLDER
		res_folder="/nas_storage/jonathan.fiorentino/SIMUL_EPITHELIUM/HEXA_SHEETS/"+s+"_Results_D"+str(D)+"_l"+str(lamb)+"/"
		for i in stages:
			for j in range(5):
				dsets = hdf5.load_datasets(s+'_sheet_'+i+str(j)+'.hf5',data_names=['vert', 'edge', 'face'])
				sheet = Sheet(i+str(j), dsets)
				print(i,j)
				absorption_times=np.random.exponential(1.0/lamb,N_traj)
				absorption_indices=np.array(absorption_times/dt,dtype=int)
				filt_edge_df=sheet.edge_df[['face','length','trgt','srce','sx','sy','tx','ty']]
				vert_df=sheet.vert_df
				edges_to_keep=[]
				vert_edges=[]
				# Filter out repeated edges
				for vi in vert_df.index:
				    tmp_df=filt_edge_df.loc[(filt_edge_df['trgt']==vi) | (filt_edge_df['srce']==vi)][['trgt','srce']]
				    tmp_df['check_string'] = tmp_df.apply(lambda row: ''.join(sorted([str(row['srce']),str(row['trgt'])])), axis=1)
				    tmp_df=tmp_df.drop_duplicates('check_string')
				    edges_to_keep.append(list(tmp_df.index))
				    vert_edges.append((filt_edge_df.loc[(filt_edge_df['trgt']==vi) | (filt_edge_df['srce']==vi)]).index)
				
				lengths=[len(ve) for ve in vert_edges]
				vert_edges_arr=np.full((len(vert_edges),np.amax(lengths)),-1).astype(np.int32)
				
				for b in range(len(vert_edges_arr)):
				    vert_edges_arr[b,:len(list(vert_edges[b]))]=list(vert_edges[b])
				    
				edges_to_keep=[item for sublist in edges_to_keep for item in sublist]
				filt_edge_df=filt_edge_df.loc[list(set(edges_to_keep))]
				
				ext_points_lst=np.zeros((len(sheet.edge_df.index),4))
				verts_lst=np.zeros((len(sheet.edge_df.index),2),dtype=np.int32)
				midpoint_lst=np.zeros((len(sheet.edge_df.index),2))
				theta0_lst=np.zeros((len(sheet.edge_df.index),2))
				
				r=0
				for e in sheet.edge_df.index:
				    ext_points=np.array([list(sheet.edge_df.loc[e][['sx','sy','srce']]),list(sheet.edge_df.loc[e][['tx','ty','trgt']])])
				    #Sort according to x
				    ext_points=ext_points[ext_points[:,0].argsort()]
				    # Save the re-ordered source and target vertices
				    verts=ext_points[:,2].astype(np.int32)
				    ext_points=ext_points[:,:2]
				    theta0=ComputeTheta0(ext_points)
				    midpoint=0.5*np.sum(ext_points,axis=0)
				    verts_lst[r]=verts
				    ext_points_lst[r]=[ext_points[0,0],ext_points[0,1],ext_points[1,0],ext_points[1,1]]
				    midpoint_lst[r]=midpoint
				    theta0_lst[r]=[np.cos(theta0),np.sin(theta0)]
				    r+=1
				
				start=time.time()
				ResultTot,res_matrices=FullSimulation(sheet,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,delta,dt,lamb,N_traj,n_cores=n_cores)
				end=time.time()
				print(end-start)
				
				filehandler = open(res_folder+"Matrices_simulation_"+i+str(j)+'.ISD', 'wb') 
				pickle.dump(res_matrices, filehandler)				
