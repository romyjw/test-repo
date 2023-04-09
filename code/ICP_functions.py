import sys,os

import pyglet
pyglet.options['shadow_window'] = True

import pyrender#to display mesh
import numpy as np
import trimesh#to load mesh

import matplotlib
import matplotlib.pyplot as plt
import igl

from sklearn.neighbors import KDTree

def scene_factory(render_list, return_nodes=False):
    
    scene = pyrender.Scene(ambient_light=0.5*np.array([1.0, 1.0, 1.0, 1.0]))
    nd_list=[]
    for m in render_list:
        nd=scene.add(m)
        nd_list.append(nd)
    
    if return_nodes:
        return scene, nd_list
    else:
        return scene

def show_mesh_gui(rdobj):
    scene = scene_factory(rdobj)
    v=pyrender.Viewer(scene, use_raymond_lighting=True,show_world_axes=True)
    del v

def PCA_normal_estimation(surface_points, k,tree=None):
    num_points, _ = surface_points.shape
    
    if tree==None:
        tree = KDTree(surface_points)
   
    _, indices = tree.query(surface_points, k)
    normals = np.zeros([num_points,3])
    for point in range(num_points):
        neighbours = surface_points[indices[point],:]
        mean = np.mean(neighbours,axis=0)
        neighbours_adjust = neighbours - mean
        covariance_matrix = np.cov(neighbours_adjust.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_values = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[sorted_values]
        eigenvectors = eigenvectors[:,sorted_values]
        normal = eigenvectors[:,2]
        nn = np.linalg.norm(normal)
        normal = normal / nn
        normals[point,:] = normal
    return normals

def render_from_obj(fp):
    
    tm = trimesh.load_mesh(fp)#load mesh
    surface_points = tm.sample(80000)#sample points for a point cloud
    normals = PCA_normal_estimation(surface_points,20)#call PCA normal estimation
    colors=np.abs(normals)
    mesh_rd = pyrender.Mesh.from_points(surface_points, colors)#make a point cloud object
    
    return mesh_rd
    
    
def best_rigid_transformation(P,Q):#input is just the arrays of vertices. faces not required.
    
    P,Q=P.T,Q.T
    
    p_bar,q_bar = np.mean(P,axis=1),np.mean(Q,axis=1)
    P_tilde,Q_tilde = (P.T-p_bar.T).T,(Q.T-q_bar.T).T
    
    U,Sigma,Vt = np.linalg.svd(Q_tilde@(P_tilde.T))
    V = Vt.T
    
    R_hat = V@U.T
    if np.linalg.det(R_hat)<0.0:
        R_hat = V@np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-1.0]])@U.T
    
    t_hat = p_bar-R_hat@q_bar
    
    return R_hat,t_hat

def best_rigid_transformation_weighted(P,Q,W):#input is just the arrays of vertices. faces not required.
    
    P,Q=P.T,Q.T
    
    p_star,q_star = np.mean(W*P,axis=1)/np.sum(W),np.mean(W*Q,axis=1)/np.sum(W)#this is different
    P_tilde,Q_tilde = np.sqrt(W)*((P.T-p_bar.T).T),np.sqrt(W)*((Q.T-q_bar.T).T)#this is different
    
    U,Sigma,Vt = np.linalg.svd(Q_tilde@(P_tilde.T))#same as best_rigid_transform, from here on.
    V = Vt.T
    
    R_hat = V@U.T
    if np.linalg.det(R_hat)<0.0:
        R_hat = V@np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-1.0]])@U.T
    
    t_hat = p_bar-R_hat@q_bar
    
    return R_hat,t_hat

def best_rigid_transformation_PTPlane(P,Q,P_normals):#input is just the arrays of vertices. faces not required.
    
    P,Q=P.T,Q.T
    P_normals=P_normals.T
    
    col1 = np.array([np.cross(Q[:,i],P_normals[:,i]) for i in range(P.shape[1])])
    A=np.hstack((col1,P_normals.T))
  
    b = np.array([(-np.dot(P_normals[:,i],(Q[:,i]-P[:,i]))) for i in range(P.shape[1])])
    
    x = np.linalg.inv(A.T@A)@A.T@b#solve linear system
    
    alpha,beta,gamma = x[0],x[1],x[2]
    
    R_hat = np.eye(3) + np.array([[0,-gamma,beta],[gamma,0,-alpha],[-beta,alpha,0]])
    t_hat = x[3:]
    
    return R_hat,t_hat

    
    
def make_correspondences(P,P_tree,Q,rejection_rate=20):
    
    dist, ind = P_tree.query(Q, k=1)
    filter_arr = dist<rejection_rate*np.median(dist)
    
    if np.sum(filter_arr)<100:
        filter_arr=np.ones_like(filter_arr)
    
    
    P_refined = P[ind[filter_arr]]
    Q_refined = Q[filter_arr[:,0],:]
    
    return P_refined,Q_refined

def make_PTPlane_correspondences(P,P_tree,Q,rejection_rate=20,P_normals=None):
    
    rejection_rate=50
    
    dist, ind = P_tree.query(Q, k=1)
    filter_arr = dist<rejection_rate*np.median(dist)
    
    if np.sum(filter_arr)<100:
        filter_arr=np.ones_like(filter_arr)
    
    
    P_refined = P[ind[filter_arr]]
    Q_refined = Q[filter_arr[:,0],:]
    
    if not (P_normals is None):
        P_normals_refined = P_normals[ind[filter_arr]]
        return P_refined,Q_refined,P_normals_refined
    
    return P_refined,Q_refined
    
def ICP(P,Q,max_n=100,tol=1e-8):
    intermediate_Q = Q
    total_R=np.eye(3)
    total_t=np.array([[0.0,0.0,0.0]]).T

    P_tree = KDTree(P)
    mse_losses=[]

    previous_loss=1000#any big number
    for i in range(max_n):
        intermediate_P,intermediate_Q=make_correspondences(P,P_tree,intermediate_Q)
        intermediate_P,intermediate_Q=intermediate_P.squeeze(),intermediate_Q.squeeze()

        this_R,this_t=best_rigid_transformation(intermediate_P,intermediate_Q)
        total_R = this_R@(total_R)
        total_t = ((this_R@total_t).T + (this_t).T).T

        intermediate_Q = (this_R@intermediate_Q.T).T + this_t
        
        this_loss = np.mean(np.sum((intermediate_P-intermediate_Q)**2, axis=1))
    
        mse_losses.append(this_loss)
        if previous_loss-this_loss<1e-8:
            break
            
        previous_loss=this_loss
        
    if i+1==max_n:
        print ('ICP did not converge after '+str(i+1)+' iterations.')
    else:
        print ('ICP converged after '+str(i+1)+' iterations.')
    
    return total_R,total_t,mse_losses,i+1

def ICP_with_initial_rotation(P,Q,max_n=100,tol=1e-8):
    #Make test rotations. we will use these to perform initial rotations on the meshes that are hard to align.
    angle_increment=np.pi/8
    test_rotations=[np.array([[np.cos(i*angle_increment),np.sin(i*angle_increment),0.0],
                  [-np.sin(i*angle_increment),np.cos(i*angle_increment),0.0],
                  [0.0,0.0,1.0]]) for i in [0,1,-1,2,-2,3,-3,4,5]]
                  
    
    
    best_rotation=None
    best_translation=None
    best_mses=[1000]#big number
    best_iterations=[]
    best_i=0
    
    for i in range(len(test_rotations)):
        
        R,t,mse_losses,iterations=ICP(P,(test_rotations[i]@Q.T).T,max_n=50)
        if i==0 or mse_losses[-1]<best_mses[-1]:
            best_mses=mse_losses
            best_rotation=R@test_rotations[i]
            best_translation=t
            best_iterations=iterations
            best_i=i

    R,t,mse_losses,iterations=ICP(P,(test_rotations[best_i]@Q.T).T,max_n=max_n,tol=tol)
    best_mses=mse_losses
    best_rotation=R@test_rotations[i]
    best_translation=t
    best_iterations=iterations
    best_i=i
    

    return best_rotation,best_translation,best_mses,best_iterations

def get_main_direction(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T@P)
    sorted_values = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:,sorted_values]
    return eigenvectors[0]
            
def ICP_with_PCA_initial_rotation(P,Q,max_n=100,tol=1e-8):

    P_dir = get_main_direction(P)
    Q_dir = get_main_direction(Q)
    
    P_dir_xy_proj,P_dir_z_proj = P_dir[:2],P_dir[2]#project into x-y plane and z axis
    Q_dir_xy_proj,Q_dir_z_proj = Q_dir[:2],Q_dir[2]

    P_dir_xy_proj = P_dir_xy_proj/np.linalg.norm(P_dir_xy_proj)#normalise
    Q_dir_xy_proj = Q_dir_xy_proj/np.linalg.norm(Q_dir_xy_proj)

    P_dir_xy_proj = P_dir_xy_proj*np.sign(P_dir_z_proj)#make sure the direction corresponds to 'up' not 'down'
    Q_dir_xy_proj = Q_dir_xy_proj*np.sign(Q_dir_z_proj)
    
    cos_theta = np.sum(P_dir_xy_proj*Q_dir_xy_proj)#find cosine of angle between models
    P_perp = np.array([[P_dir_xy_proj[1],-P_dir_xy_proj[0]]])#normal to P_dir_xy_proj
                      
    if np.sum(P_perp*Q_dir_xy_proj)<0:#check whether to use theta in 0 to pi or pi to 2pi
        sin_theta = (1-cos_theta**2)**0.5
    else: 
        sin_theta = -(1-cos_theta**2)**0.5
    
    pre_rotation=np.array([[cos_theta,sin_theta,0.0],
                [-sin_theta,cos_theta,0.0],
                [0.0,0.0,1.0]])
    
    print('angle',np.arcsin(sin_theta)*360/(2*np.pi))
    pre_translation =  np.array([np.mean(P,axis=0)-np.mean(Q,axis=0)]).T
    return pre_rotation, pre_translation, [],[]
    
    R,t,mses,iterations = ICP(P ,(pre_rotation@Q.T).T + pre_translation.T,max_n=max_n,tol=tol)

    return R@pre_rotation, R@pre_translation + t, mses, iterations
    
    
def point_to_plane_ICP(P,Q,max_n=100,tol=1e-8):
    cur_Q = Q
    total_R=np.eye(3)
    total_t=np.array([[0.0,0.0,0.0]]).T

    P_tree = KDTree(P)
    mse_losses=[]
    P_normals=PCA_normal_estimation(P,20,P_tree)#calculate vertex normals

    previous_loss=1000#any big number
    
    for i in range(max_n):
        cur_P,cur_Q,cur_P_normals=make_PTPlane_correspondences(P,P_tree,cur_Q,rejection_rate=20,P_normals=P_normals)
        
        cur_P,cur_Q=cur_P.squeeze(),cur_Q.squeeze()
        
        this_R,this_t=best_rigid_transformation_PTPlane(cur_P,cur_Q,cur_P_normals)
        
        total_R = this_R@(total_R)
        total_t = ((this_R@total_t).T + (this_t).T).T

        cur_Q = (this_R@cur_Q.T).T + this_t
        
        this_loss = np.mean(np.sum((cur_P-cur_Q)**2, axis=1))
    
        mse_losses.append(this_loss)
        if previous_loss-this_loss<tol:
            break
            
        previous_loss=this_loss
        
    if i+1==max_n:
        print ('ICP did not converge after '+str(i+1)+' iterations.')
    else:
        print ('ICP converged after '+str(i+1)+' iterations.')
    
    return total_R,total_t,mse_losses,i+1

def make_obj_test_files(obj_filenames):
	#for i in range(len(ply_filenames)):
	#	tm = trimesh.load_mesh(ply_filenames[i])
	#	trimesh.exchange.export.export_mesh(tm,'M'+str(i+1),file_type='obj',resolver=None)
        #(Trimesh aligned meshes unintentionally.)
	
	rotation1=np.array([[np.cos(1*np.pi/2.0),-np.sin(1*np.pi/2),0.0],
                  [np.sin(1*np.pi/2.0),np.cos(1*np.pi/2.0),0.0],
                  [0.0,0.0,1.0]])
                  
	theta=np.pi/2
	rotation2=np.array([[np.cos(theta),0,np.sin(theta)],
                  [0.0,1.0,0.0],
                  [-np.sin(theta),0.0,np.cos(theta)]])
                  
	for i in range(len(obj_filenames)):
		v,f = igl.read_triangle_mesh(obj_filenames[i])
		v=(rotation1@v.T).T
		v=(rotation2@v.T).T
		igl.write_triangle_mesh("M"+str(i+1)+".obj", v, f)

	return True

def plot_summary_data(x_vals,mses,iterations_used,x_label,invert=False,x_log=False):

	fig,ax=plt.subplots(1,2,figsize=(20,5))
	if x_log==True:
            x_vals=np.log(x_vals)
            
	ax[0].plot(x_vals,iterations_used,'bo-')
	ax[0].set_xlabel(x_label)
	ax[0].set_ylabel('iterations to converge')
	ax[0].set_title('Number of Iterations to Converge')
	
            
	ax[1].plot(x_vals,[mses[i][-1] for i in range(0,len(x_vals))],'bo-')
	ax[1].set_xlabel(x_label)
	ax[1].set_ylabel('final MSE')
	ax[1].set_title('Error at Point of Convergence')

	plt.show()
	
	return True

def plot_all_losses(rows,cols,experiment_mses,titles,figsize=(20,29)):
	fig,ax=plt.subplots(rows,cols,figsize=(20,29))
	for i in range(len(experiment_mses)):
    		ax[i//cols,i%cols].plot(experiment_mses[i])
    		ax[i//cols,i%cols].set_title(titles[i])
    		ax[i//cols,i%cols].set_xlabel('iteration')
    		ax[i//cols,i%cols].set_ylabel('MSE')
	plt.show()
	return True

def find_axis(R):
    #Find eigenvectors/eigenvalues of rotation matrix.
    #Select eigenvector corresponding to eigenvalue with smallest imaginary part.
    #(The axis is the only eigenvector corresponding to a real eigenvalue).
    eigenvalues, eigenvectors = np.linalg.eig(R)
    sorted_values = np.argsort(abs(eigenvalues.imag))
    eigenvalues = eigenvalues[sorted_values]
    eigenvectors = eigenvectors[:,sorted_values]
    
    axis = np.real(eigenvectors[:,0])#It should be a real vector.
    return axis/np.linalg.norm(axis)#Normalise, just in case the numerics made it non-unit.
    
def distance_between_rotations(R1,R2):
    R = R1@R2.T
    theta = np.arccos((np.trace(R)-1)/2)
    return theta

def plot_rotational_error(rotations,best_rotation,x_vals,x_label,invert=False,x_log=False):
    rotation_distances=[(distance_between_rotations(best_rotation,rotations[i])) for i in range(len(x_vals))]

    fig,ax=plt.subplots()
    if x_log==True:
        ax.plot(np.log(x_vals),rotation_distances,'bo-')
    else:
        ax.plot((x_vals),rotation_distances,'bo-')
    if invert==True:
       plt.gca().invert_xaxis()

    ax.set_xlabel(x_label)
    ax.set_ylabel('Distance from Correct Rotation')
    ax.set_title('Rotational Error')
    plt.show()
    return True
