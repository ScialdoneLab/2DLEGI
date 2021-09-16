cimport numpy as np
import numpy as np
from libc.stdint cimport int32_t, int64_t
cimport cython

# Generate one 1d Brownian trajectory on a segment with Poisson absorption
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def BoundedBrownian(double[:,:] edge_df,double[:,:] vert_df,int[:] s_edges,int[:] s_verts,
                    int[:,:] vert_edges_arr,
                    int[:,:] verts_lst,double[:,:] ext_points_lst,double[:,:] theta0_lst,
                    double[:,:] midpoint_lst,
                    int N,int abs_ind,
                    double[:] n,double[:] u_arr,int new_seg):
    
    cdef int32_t i,j,k,new_seg_idx
    cdef int32_t abs_ind_new
    cdef double xold,yold,candx,candy
    cdef double[4] ext_points
    cdef double[4] tmp_extremes
    cdef double[4] tmp_ext_points
    cdef double[2] start
    cdef double[2] midpoint
    cdef double[2] p0
    cdef int32_t count
    cdef int32_t[2] verts
    cdef int32_t[2] length_idx
    cdef double[2] abs_cells_list
    cdef double cos_theta0,sin_theta0,new_seg_length
    cdef double[2] vcoords
    cdef int32_t vi
    cdef double abs_cell
    cdef int32_t[2] results
    ######################################################
    cdef int32_t isBet;
    cdef double epsilon=0.0000001
    cdef double ax,ay,dx,dy;
    cdef double crossproduct,dotproduct,squaredlengthba
    #######################################################
    
    ######################################################
    cdef int32_t isBet2;
    cdef double ax2,ay2,dx2,dy2;
    cdef double crossproduct2,dotproduct2,squaredlengthba2
    #######################################################
    
    # Extreme points of the chosen edge and source and target vertices
    ext_points[0]=ext_points_lst[new_seg,0]
    ext_points[1]=ext_points_lst[new_seg,1]
    ext_points[2]=ext_points_lst[new_seg,2]
    ext_points[3]=ext_points_lst[new_seg,3]
        
    verts[0]=verts_lst[new_seg,0]
    verts[1]=verts_lst[new_seg,1]
    
    cos_theta0=theta0_lst[new_seg,0]
    sin_theta0=theta0_lst[new_seg,1]

    # Initial condition
    p0[0]=(1.0-u_arr[0])*ext_points[0] + u_arr[0]*ext_points[2]
    p0[1]=(1.0-u_arr[0])*ext_points[1] + u_arr[0]*ext_points[3]
    xold=p0[0]
    yold=p0[1]
    
    #If the Poisson absorption time is larger than the simulation length, assign nan to the absorption cell
    if abs_ind>=N:
        abs_cell= -1
        abs_ind_new=abs_ind

    for i in range(N):
        
        candx=xold+n[i]*cos_theta0
        candy=yold+n[i]*sin_theta0

        if i>= abs_ind:
            abs_ind_new=i
            for j in range(len(s_edges)):
                if(s_edges[j] == new_seg):
                    new_seg_idx=j

            new_seg_length=edge_df[new_seg_idx,1]

            # Indexes of the edges with that length
            count=0
            for k in range(len(edge_df[:,1])):
                if(edge_df[k,1] == new_seg_length):
                    length_idx[count]=k
                    abs_cells_list[count]=edge_df[length_idx[count],0]
                    count=count+1
            
            if count==1:
                abs_cell=abs_cells_list[0]
            else:
                # This could be implemented with cython and GSL
                abs_cell=np.random.choice(abs_cells_list,1)
            break;

        #################
        isBet= 1
        
        ax=ext_points[0]
        ay=ext_points[1]
        dx=ext_points[2]
        dy=ext_points[3]
        
        crossproduct = (candy - ay) * (dx - ax) - (candx - ax) * (dy - ay)
        if abs(crossproduct) > epsilon:
            isBet= 0;
        dotproduct = (candx - ax) * (dx - ax) + (candy - ay)*(dy - ay)
        
        if dotproduct < 0:
            isBet= 0;
        
        squaredlengthba = (dx - ax)*(dx - ax) + (dy - ay)*(dy - ay)
        if dotproduct > squaredlengthba:
            isBet= 0;
        
        ################
        if isBet==0:    
            #Hitting boundaries
            # Use candx,candy to find vertex that is going to be hit

            if candx>=xold:
                tmp_extremes[0]=xold
                tmp_extremes[1]=yold
                tmp_extremes[2]=candx
                tmp_extremes[3]=candy
            else:
                tmp_extremes[0]=candx
                tmp_extremes[1]=candy
                tmp_extremes[2]=xold
                tmp_extremes[3]=yold

            for v in range(len(verts)):
                vcoords[0]=vert_df[verts[v],0]
                vcoords[1]=vert_df[verts[v],1]
                
                isBet2=1
                ax2=tmp_extremes[0]
                ay2=tmp_extremes[1]
                dx2=tmp_extremes[2]
                dy2=tmp_extremes[3]
                crossproduct2 = (vcoords[1] - ay2) * (dx2 - ax2) - (vcoords[0] - ax2) * (dy2 - ay2)
                if abs(crossproduct2) > epsilon:
                    isBet2= 0;
                dotproduct2 = (vcoords[0] - ax2) * (dx2 - ax2) + (vcoords[1] - ay2)*(dy2 - ay2)
                if dotproduct2 < 0:
                    isBet2= 0;
                squaredlengthba2 = (dx2 - ax2)*(dx2 - ax2) + (dy2 - ay2)*(dy2 - ay2)
                if dotproduct2 > squaredlengthba2:
                    isBet= 0;
                if isBet2==1:
                    vi = verts[v]
                    start=vcoords
                    break;
            
            # Extract a new edge with equal probability from those ending in vertex vi
            my_edges_list=np.asarray(vert_edges_arr[vi])
            my_edges_list=my_edges_list[my_edges_list != -1]
            
            
            # This could be implemented with Cython and GSL
            new_seg=np.random.choice(my_edges_list,1)
                        
            #Update properties of the new edge
            ext_points[0]=ext_points_lst[new_seg,0]
            ext_points[1]=ext_points_lst[new_seg,1]
            ext_points[2]=ext_points_lst[new_seg,2]
            ext_points[3]=ext_points_lst[new_seg,3]
            
            verts[0]=verts_lst[new_seg,0]
            verts[1]=verts_lst[new_seg,1]
            
            cos_theta0=theta0_lst[new_seg,0]
            sin_theta0=theta0_lst[new_seg,1]
            
            midpoint[0]=midpoint_lst[new_seg,0]
            midpoint[1]=midpoint_lst[new_seg,1]
            
            tmp_ext_points[0]=start[0]
            tmp_ext_points[1]=start[1]
            tmp_ext_points[2]=midpoint[0]
            tmp_ext_points[3]=midpoint[1]
            
            p0[0]=(1.0-u_arr[i])*tmp_ext_points[0] + u_arr[i]*tmp_ext_points[2]
            p0[1]=(1.0-u_arr[i])*tmp_ext_points[1] + u_arr[i]*tmp_ext_points[3]
            
            xnew=p0[0]
            ynew=p0[1]
                        
            xold=xnew
            yold=ynew
        
        else: # If the molecule did not hit any boundary, just update the position with Gaussian increment
            xnew=candx
            ynew=candy
            
            xold=xnew
            yold=ynew

    results[0]=np.int32(abs_cell)
    results[1]=np.int32(abs_ind_new)
    return results;
