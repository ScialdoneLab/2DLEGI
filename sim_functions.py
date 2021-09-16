from scipy.stats import norm
from math import sqrt
import random
from collections import Counter
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import math
from ISDBrownian import BoundedBrownian


class ISDresult(object):
    def __init__(self,results,absorption_times):
        #self.trajectories = [x[0] for x in results]
        self.abs_cell = [x[0] for x in results]
        self.abs_times=absorption_times
        self.abs_indices=[x[1] for x in results]

def ComputeTheta0(extremes):
    return np.arctan2(extremes[1,1]-extremes[0,1],extremes[1,0]-extremes[0,0])

def Parallelise(edge_df,vert_df,s_edges,s_verts,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,absorption_indices,gauss_incr,u_arr,new_seg_arr,n_cores=None):
    X = [(edge_df,vert_df,s_edges,s_verts,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,abs_ind,n,u,new_seg) for (abs_ind,n,u,new_seg) in zip(absorption_indices,gauss_incr,u_arr,new_seg_arr)]
    
    results = []
    
    if n_cores==None or n_cores > mp.cpu_count():
        pool = mp.Pool(mp.cpu_count())
    else:
        pool = mp.Pool(n_cores) # If the user wants to use a specified number of cores
    
    results = pool.starmap_async(BoundedBrownian, X)
    results = results.get()
    pool.close()    

    return results;

# Note: this function should return an object of class ISDresult with attributes (note that some of these 
# are not used, can be eliminated):
# 1) absorption: index of the absorbing cell for each trajectory
# 2) absorption_times: absorption time for each trajcetory
# 3) absorption_indices: absorption indices for each trajcetory
def OneCellSimul(s,edge_df,vert_df,s_edges,s_verts,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,delta,dt,sci,lamb,N_traj,n_cores=None):
    
    #Extract N_traj absorption time from an exponential distribution (Poisson)
    absorption_times=np.random.exponential(1.0/lamb,N_traj)
    absorption_indices=np.array(absorption_times/dt,dtype=int)
    
    # Extract N*N_traj Gaussian increments
    gauss_incr=norm.rvs(size=(N_traj,N), scale=delta*sqrt(dt))
    
    #Extract uniform random numbers
    u_arr=np.random.uniform(0,1,(N_traj,N))
    
    #Given the index of the starting cell, randomly choose the starting segment from those belonging to that cell
    #The weights are the lengths of the segments corresponding to that cell
    av_edges=s.edge_df[s.edge_df['face']==sci].index
    w_edges=s.edge_df['length'].loc[av_edges]

    new_seg_arr=random.choices(av_edges,weights=w_edges, k=N_traj)
    results=Parallelise(edge_df,vert_df,s_edges,s_verts,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,absorption_indices,gauss_incr,u_arr,new_seg_arr,n_cores)
    
    res=ISDresult(results,absorption_times)
    
    return res;

def OneCellResults(res,res_matrices,sci): 
    
    # Count how many trajectories are absorbed by each segment (or not absorbed)
    # Initialize the counter with zeros for the case in which there are cells that do not absorb any molecules
    Nc=res_matrices[0].shape[0]
    # Count how many trajectories are absorbed by each segment (or not absorbed)
    # Initialize the counter with zeros for the case in which there are cells that do not absorb any molecules
    c = Counter()
    c.update({x:0 for x in list(np.arange(0,Nc))+[np.nan]})
    
    # Convert the -1 in nan
    my_res=[float('nan') if x== -1 else x for x in res.abs_cell]
    
    c.update(my_res)
    
    # Compute the fraction
    total = sum(c.values(), 0.0)
    for key in c:
        c[key] /= total
    
    absorption_probabilities_df=pd.DataFrame.from_dict(c, orient='index').reset_index()
    absorption_probabilities_df_2 = absorption_probabilities_df.reset_index().dropna().set_index('index')
    absorption_probabilities_df_2=absorption_probabilities_df_2.set_index('level_0')
    
    data={'T_abs': res.abs_times,'C_abs':res.abs_cell}
    absorption_df=pd.DataFrame(data=data)
    absorption_df=absorption_df.groupby('C_abs').mean()
    
    for i in list(absorption_probabilities_df_2.index):
        if i not in list(absorption_df.index):
            df2 = pd.DataFrame(columns=['T_abs'])
            df2.loc[0]=0.0
            df2.index=pd.Int64Index([i])
            absorption_df=absorption_df.append(df2,ignore_index=False)
    
    absorption_df=absorption_df.reindex(absorption_probabilities_df_2.index)
    absorption_df['P_abs']=list(absorption_probabilities_df_2[0])
    absorption_df=absorption_df.fillna(0)
    absorption_df.columns=['Mean_T_abs','P_abs']
    
    # Fill the matrices
    res_matrices[0][sci,:]=list(absorption_df['Mean_T_abs'])
    res_matrices[1][sci,:]=list(absorption_df['P_abs'])
    
def FullSimulation(s,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,delta,dt,l,N_traj,n_cores=None):
    
    # Use Numpy arrays instead of Pandas dataframes
    edge_df=np.array(s.edge_df[['face','length']])
    vert_df=np.array(s.vert_df[['x','y']])
    
    s_edges=np.array(s.edge_df.index).astype(np.int32)
    s_verts=np.array(s.vert_df.index).astype(np.int32)

    # Define 3 matrices Ncells x Ncells:
    # 1. Matrix of the average absorption time by each cell
    # 2. Matrix of the probability of absorption (#traj absorbed by given cell/N_traj)
    # 3. Matrix of gammas (2./1.)
    Nc=len(s.face_df)
    abs_time_matrix=np.empty((Nc,Nc))
    abs_prob_matrix=np.empty((Nc,Nc))
    
    res_matrices=[abs_time_matrix,abs_prob_matrix]
    
    ResultTot=[]
    #For each cell in the configuration, generate N_traj trajectories starting from that cell
    start_cell_index_arr=np.arange(0,Nc)
    for start_cell_index in tqdm(range(Nc)):

        res=OneCellSimul(s,edge_df,vert_df,s_edges,s_verts,vert_edges_arr,verts_lst,ext_points_lst,theta0_lst,midpoint_lst,N,delta,dt,start_cell_index,l,N_traj,n_cores)
        
        # This is a list of ISDresult objects containing the results of the simulations for each releasing cell
        # in the epithelial sheet
        ResultTot.append(res)
        OneCellResults(res,res_matrices,start_cell_index)
    return ResultTot,res_matrices;

# Animate a trajectory given the epithelial sheet, the results of the simulation and the index of the releasing cell
def AnimateTraj(s,result_list,N_traj,sci=0):
    
    # Select the object for the given releasing cell
    results=result_list[sci]
    trajectories=results.trajectories
    absorption_indices=results.abs_indices
    absorption=results.abs_cell
    
    # Select a random trajectory 
    my_index=np.random.randint(0,N_traj)
    #my_index=1
    x=trajectories[my_index][0][:absorption_indices[my_index]+1]
    y=trajectories[my_index][1][:absorption_indices[my_index]+1]
        
    # Initialise the animator
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    # Plot the background (the epithelial sheet)
    fig, ax = sheet_view(s, mode="quick", edge={"alpha": 0.8})
    ax.set_xlabel(r'$x (\mu m)$')
    ax.set_ylabel(r'$y (\mu m)$')
    if np.isnan(absorption[my_index]):
        ax.set_title('AVR: %.2f; Releasing cell: %d; Absorbing cell: %s' % (s.vert_df['rank'].mean(),
                                                                 sci,r'$T_{abs} > T$'))
    else:
        ax.set_title('AVR: %.2f; Releasing cell: %d; Absorbing cell: %d' % (s.vert_df['rank'].mean(),
                                                                 sci,absorption[my_index]))
    
#     ax.scatter(trajectories[my_index][0][absorption_indices[my_index]],
#               trajectories[my_index][1][absorption_indices[my_index]],color='red')
#     ax.scatter(trajectories[my_index][0][absorption_indices[my_index]+1],
#               trajectories[my_index][1][absorption_indices[my_index]+1],color='black')
    
    for i, txt in enumerate(s.face_df.index):
        ax.annotate(txt, (s.face_df['x'].loc[i], s.face_df['y'].loc[i]))
    
    # First set up the figure, the axis, and the plot element we want to animate
    line, = ax.plot([], [],'o')
    # Print simulation time
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    fig.tight_layout()
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return (line,time_text)
    
    # animation function. This is called sequentially
    def animate(i):
        line.set_data(x[i], y[i])
        time_text.set_text('time = %.1f s' % sim_time[i])
        return (line,time_text)
    
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(x), interval=50, blit=True)
    
    HTML(anim.to_html5_video())
    anim.save('traj_'+str(sci)+'_'+s.identifier+'.mp4', writer=writer)
    
# Plot the matrices obtained from the simulation:
# - Matrix of the average absorption times
# - Matrix of the absoprtion probabilities
# - Matrix of the exchange rates between cells
def PlotResults(res_matrices,annot=False,x_axis_labels=None,y_axis_labels=None):
	
	if x_axis_labels==None:
		x_axis_labels=np.arange(res_matrices[0].shape[0])
	if y_axis_labels==None:
		y_axis_labels=np.arange(res_matrices[0].shape[0])
	
	fig, ax = plt.subplots(1,2,figsize=(10,4))
	plt.figure()
	ax[0].set_title('Matrix of the average absorption times')
	sns.heatmap(res_matrices[0], cmap="YlGnBu",annot=annot,ax=ax[0],
	cbar_kws={'label': 'Avg absorption time (s)'},xticklabels=x_axis_labels, 
	yticklabels=y_axis_labels)
	
	ax[0].set_xlabel('Absorbing cell')
	ax[0].set_ylabel('Releasing cell')
	
	ax[1].set_title('Matrix of the absorption probabilities')
	
	sns.heatmap(res_matrices[1], cmap="YlGnBu",annot=annot,ax=ax[1],
	cbar_kws={'label': 'Absorption probability'},xticklabels=x_axis_labels, 
	yticklabels=y_axis_labels)
	
	ax[1].set_xlabel('Absorbing cell')
	ax[1].set_ylabel('Releasing cell')
	
	fig.tight_layout()
	plt.show(),plt.close()
