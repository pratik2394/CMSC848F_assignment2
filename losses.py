import torch
import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d

	#instead of using binary cross entropy loss and applying sigmoid layer seperately, I am using BCEWithLogitsLoss.
	bceloss = torch.nn.BCEWithLogitsLoss()	
	loss = bceloss(voxel_src, voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src, point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	source2target_knn = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1, norm=2)
	target2source_knn = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1, norm=2)
	# Calculate chamfer loss
	source2target_dist = source2target_knn.dists[..., 0]  # (B, N)
	target2source_dist = target2source_knn.dists[..., 0]  # (B, M)
	loss_chamfer = source2target_dist.mean() + target2source_dist.mean()
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian =  mesh_laplacian_smoothing(mesh_src, method="uniform")
	# implement laplacian smoothening loss
	
	return loss_laplacian