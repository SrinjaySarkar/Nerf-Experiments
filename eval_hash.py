import numpy as np
import os
import torch
# torch.manual_seed(5)
from tqdm import tqdm
import imageio
import cv2
import time
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from load_data import load_llff
from model import hash_nerf,hash_embedding
from utils import get_image_rays,meshgrid,ndc_rays,get_minibatches,cumprod,sampling,psnr_loss,mse



validation_phase=0
device=torch.device("cpu")

"""things to look at carefully
1:hashing function in utils"""

"""list of changes:
1:changed direction embedding from spherical harmonics to standard hash embedder maybe spheircal harmonics doesn't need to be optimized, 
2:using Adam optimizer instead of rectified Adam"""


def volume_renderer(radiance_field,depth,ray_direction,noise,white_bg):
    one_e_10=torch.tensor([1e10],dtype=ray_direction.dtype,device=ray_direction.device)
    dists=depth[:,1:]-depth[:,:-1] #(4096,63)
    dists=torch.cat((dists,one_e_10.expand(depth[:,:1].shape)),dim=-1)#4096,64
    # dists=dists*ray_direction.unsqueeze(1).norm(p=2,dim=-1)
    dists=dists*ray_direction[...,None,:].norm(p=2, dim=-1)
    # dists=dists*ray_direction.unsqueeze(1)
    
    rgb=torch.sigmoid(radiance_field[...,:3])   
    density=radiance_field[...,3]
    noise=0.0
    sigma=F.relu(density+noise)

    # print(sigma.shape)    
    # print(dists.shape)
    alpha=1.0-torch.exp(-sigma*dists)
    
    # weights=alpha*torch.cumprod()
    weights=alpha*cumprod(torch.exp(-sigma*dists)+1e-10)#(Ti*(1-exp(-sigma*delta)))see eq 3
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map=rgb*weights.unsqueeze(2)
    rgb_map=torch.sum(rgb_map,dim=-2)#(n_rays,3(4096,3))
    depth_map=depth*weights
    depth_map=torch.sum(depth_map,dim=-1)
    acc_map=weights.sum(dim=-1)
    disp_map=1.0/torch.max(1e-10*torch.ones_like(depth_map),depth_map/acc_map) # inverse of depth map
    if white_bg:
        rgb_map=rgb_map+(1.0-acc_map[...,None])
    # print(rgb_map.shape,disp_map.shape,acc_map.shape,depth_map.shape)

    mask=weights.sum(-1)>0.5
    entropy=torch.distributions.Categorical(probs=weights+1e-5).entropy()
    sparsity_loss=entropy * mask

    return (rgb_map,disp_map,acc_map,weights,depth_map,sparsity_loss)

def nerf_net(model,points,rays,chunksize,point_embedding,dir_embedding):
    points1=points.reshape((-1,points.shape[-1]))
    # print("points1",points1.shape)
    pt_embedding=point_embedding(points1)
    if dir_embedding is not None:
        # print(rays.shape)
        viewdirs=rays[...,None,-3:]
        input_dirs=viewdirs.expand(points.shape)
        # viewdirs=viewdirs.unsqueeze(1).expand(4096,64+128,3) or (4096,64,3)
        # print(viewdirs.shape)
        input_dirs1=input_dirs.reshape((-1,input_dirs.shape[-1]))
        viewdir_embedding=dir_embedding(input_dirs1)
        embedded=torch.cat((pt_embedding,viewdir_embedding),dim=-1)
    
    batches=get_minibatches(embedded,chunksize=chunksize) 
    pred=[model(batch) for batch in batches]
    radiance_field=torch.cat(pred,dim=0)#(4096*64,4)
    # print(radiance_field.shape)
    # print(points.shape)
    radiance_field=radiance_field.reshape(list(points.shape[:-1])+[radiance_field.shape[-1]])#(4096,64,4)
    # print(radiance_field.shape)
    return (radiance_field)

def pred_radiance(rays,coarse_model,fine_model,mode,encode_pos,encode_dir):
    n_rays=rays.shape[0]
    r_origin,r_direction=rays[...,:3],rays[...,3:6]
    bounds=rays[...,6:8].view((-1,1,2))
    near,far=bounds[...,0],bounds[...,1]

    # print(near,far)

    coarse_points=torch.linspace(0.0,1.0,64,dtype=r_origin.dtype,device=r_origin.device)
    z_vals=near*(1.0-coarse_points)+far*coarse_points#Sample linearly in disparity space, as opposed to in depth space.
    z_vals=z_vals.expand([n_rays,64])
    # print(z_vals)
    if mode == "train":
        mids=0.5*(z_vals[...,1:]+z_vals[...,:-1])#samples between every set of coarse points 
        # print("Mids:",mids.shape)
        # print(z_vals[0,1:],z_vals[0,:-1])
        # print(mids[0,0])
        upper=torch.cat((mids,z_vals[...,-1:]),dim=-1)#appending last and first points to the interval samples
        lower=torch.cat((z_vals[...,:1],mids),dim=-1)
        """CHANGE TO torch.rand"""
        r1=torch.rand(z_vals.shape,dtype=r_origin.dtype,device=r_origin.device)########################
        z_vals=lower+(upper-lower)*r1 #stratified sampling (eq2 of paper)#4096,64
    # print("Z_vals:",z_vals)
    #o+td
    # print(r_origin.shape)#4096,3
    # print(r_direction.shape)#4096,3
    #origin(4096,64,3),direction(4096,64,3),points(4096,64,3)
    ro1=r_origin.unsqueeze(1).expand(r_origin.shape[0],64,3)
    rd1=r_direction.unsqueeze(1).expand(r_direction.shape[0],64,3)
    z1=z_vals.unsqueeze(2).expand(z_vals.shape[0],64,3)
    coarse_rays=ro1+rd1*z1
    # print("coarse:",coarse_rays)
    radiance_field=nerf_net(coarse_model,coarse_rays,rays,16384,encode_pos,encode_dir)  
    # print(radiance_field.shape)
    # print(radiance_field[1,1,:])
    coarse_rgb_map,coarse_disp_map,coarse_acc_map,weights,coarse_depth_map,sparsity_loss=volume_renderer(radiance_field,z_vals,r_direction,0,False)
    return (coarse_rgb_map,coarse_disp_map,coarse_acc_map,weights,coarse_depth_map,sparsity_loss,radiance_field)


def run_1_nerf(height,width,focal_length,coarse_model,fine_model,ray_origin,ray_direction,mode,encode_pos,encode_dir):
    viewdirs=ray_direction#(directions64*64*3 pixels)
    viewdirs=viewdirs/viewdirs.norm(p=2,dim=-1).unsqueeze(-1)
    # print(viewdirs.shape)
    # print(viewdirs[1])
    viewdirs=viewdirs.view((-1,3))
    # print("Normalized ray directions:",viewdirs.shape)
    restore_shapes=[ray_direction.shape,ray_direction.shape[:-1],ray_direction.shape[:-1]]
    # print("Shape:",restore_shapes)
    if fine_model:
        restore_shapes+=restore_shapes
    # print("restore shapes:",restore_shapes)
    #ndc ryas
    ro,rd=ndc_rays(height,width,focal_length,1.0,ray_origin,ray_direction)#ndc samples from 1 to infinity
    # print("ndc:",ro.shape)
    # print("ndc:",rd.shape)
    # print(torch.norm(rd,dim=-1))
    ro=ro.view((-1,3))
    rd=rd.view((-1,3))

    #clip all depth not betn far and near
    near=0*torch.ones_like(rd[...,:1])
    far=1*torch.ones_like(rd[...,:1])
    
    rays=torch.cat((ro,rd,near,far,viewdirs),dim=-1)
    print("DDDDDDDD",rays[0])
    batches=get_minibatches(rays,chunksize=16384)
    pred=[pred_radiance(batch,coarse_model,fine_model,"train",encode_pos,encode_dir) for batch in batches]
    synthesized_images=list(zip(*pred))
    synthesized_images=[torch.cat(image,dim=0) if image[0] is not None else (None) for image in synthesized_images]

    return (tuple(synthesized_images))


llffhold=8
images,poses,bds,render_poses,test_idx,llff_bbox=load_llff(basedir="/vinai/sskar/NERF/nerf_llff_data/fern",factor=8)
hwf=poses[5,:3,-1]
poses=poses[:,:3,:4]
print('Loaded llff',images.shape,render_poses.shape,hwf)

if not isinstance(test_idx,list):   
    test_idx=[test_idx]
    if llffhold>0:
        test_idx=np.arange(images.shape[0])[::llffhold]
    val_idx=test_idx
    train_idx=np.array([i for i in np.arange(images.shape[0]) if (i not in test_idx and i not in val_idx)])

#near=0,far=1
height,width,focal_length=hwf
height,width=int(height),int(width)
hwf=[height,width,focal_length]
K=np.array([[focal_length,0,0.5*width],[0,focal_length,0.5*height],[0,0,1]])

# images=torch.from_numpy(images)
# poses=torch.from_numpy(poses)
render_poses=np.array(poses[test_idx])
render_poses=torch.Tensor(render_poses)

#we can either use positional embedding or hash embedding
#expname="fern_test_hash_XYZ_fine512_log2T19_lr0.01_decay10_RAdam_sparse1e-10_TV1e-6"

#point embedding
point_embedding_model=hash_embedding(llff_bbox,n_levels=16,n_features_per_level=2,log2_hashmap_size=19,base_resolution=16,finest_resolution=512).to(device)
point_input_ch=point_embedding_model.out_dim
print(point_input_ch)
point_embedding_params=list(point_embedding_model.parameters())

#dir embedding
#dir embedding model should be spherical harmonics
dir_embedding_model=hash_embedding(llff_bbox,n_levels=16,n_features_per_level=2,log2_hashmap_size=19,base_resolution=16,finest_resolution=512).to(device)
dir_input_ch=dir_embedding_model.out_dim
print(dir_input_ch)
dir_embedding_params=list(dir_embedding_model.parameters())

output_ch=5
coarse_model=hash_nerf(n_layers=2,hidden_dim=64,geo_feat_dim=15,n_layers_color=3,hidden_dim_color=64,input_ch=point_input_ch,input_ch_views=dir_input_ch)
coarse_model=coarse_model.to(device)
grad_vars=list(coarse_model.parameters())

#netowrk query  
#inputs:points,viewdirs:rays,network_fn:coarse_model,point_embedding_model,dir_embedding_model
network_query_fn=lambda inputs,viewdirs,network_fn:run_network(inputs,viewdirs,network_fn,embed_fn=point_embedding_model,
                                                                embeddirs_fn=dir_embedding_model, netchunk=1024*16)

optimizer=optim.Adam([{"params":grad_vars,"weight_decay":1e-6},{"params":point_embedding_params,"eps":1e-15}],lr=5e-4,betas=(0.9,0.99))
start=0
global_step=start

if validation_phase:
    weights_path="/vinai/sskar/NERF/ckpts/ckpt_25000.ckpt"
    ckpts=torch.load(weights_path)

    start=ckpt['global_step']
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    coarse_model.load_state_dict(ckpt['network_fn_state_dict'])
    if fine_model is not None:
        fine_model.load_state_dict(ckpt['network_fine_state_dict'])
    point_embedding_model.load_state_dict(ckpt['embed_fn_state_dict'])


N_samples=4096
mode="train"
#random rays from all images
rays=[get_image_rays(height,width,focal_length,pose_target) for pose_target in torch.from_numpy(poses[:,:3,:4])]
rays=[(i[0].numpy(),i[1].numpy()) for i in rays]
rays=np.stack(rays,0)
# print(rays.shape)#(20,2(ro,rd),h,w,3)
rays=np.concatenate([rays,images[:,None]],1)
# print(rays.shape)(20,3(ro,rd,pixels),h,w,3)
rays=np.stack([rays[idx] for idx in train_idx],0)#selecting train images
rays=np.reshape(rays,[-1,3,3])
# print(rays.shape)(17*378*504,3(ro,rd,pixels),3)
rays=rays.astype(np.float32)
np.random.shuffle(rays)

images=torch.from_numpy(images).to(device)
poses=torch.from_numpy(poses).to(device)
rays=torch.from_numpy(rays).to(device)

batch_idx=0
n_epochs=1
start=0
for iter in range(start,n_epochs):
    rays_batch=rays[batch_idx:batch_idx+N_samples]
    # print(rays_batch.shape)
    rays_batch=torch.transpose(rays_batch,0,1)
    # print(rays_batch.shape)
    rays_batch,target_pixels=rays_batch[0:2,],rays_batch[2]

    batch_idx=batch_idx+N_samples
    if batch_idx >= rays.shape[0]:
        idx=torch.randperm(rays.shape[0])
        rays=rays[idx]
        batch_idx=0
    ro,rd=rays_batch
    
    ref_list=render(height,width,K,chunk=16384,rays=rays_batch,network_fn=coarse_model,
    network_query_fn=network_query_fn,embed_fn=point_embedding_model)
    print(len(ref_list))

    my_list=run_1_nerf(height,width,focal_length,coarse_model,None,ro,rd,mode,point_embedding_model,dir_embedding_model)
    print(len(my_list))

    optimizer.zero_grad()
    img_loss=F.mse_loss(my_list[0],target_pixels)
    loss=img_loss
    psnr=psnr_loss(img_loss)

    n_levels=point_embedding_model.n_levels
    min_res=point_embedding_model.base_resolution
    max_res=point_embedding_model.finest_resolution
    hash_size=point_embedding_model.log2_hashmap_size

    tv_loss=sum(total_var_loss(point_embedding_model[i],min_res,max_res,i,hash_size,n_levels=n_levels) for i in range(n_levels))
    loss=loss+(1e-6)*tv_loss

    if iter>1000:
        loss=loss
    
    loss.backward()
    optimizer.step()

    decay_rate=0.1
    decay_steps=250*1000
    new_rate=5e-4*(decay_rate ** (global_stop/decay_steps))
    
    for param_group in optimizer.param_groups:
            param_group['lr']=new_rate
    
    if iter % 1000 == 0:
        path=os.path.join("/vinai/sskar/NERF/",'{:06d}.tar'.format(iter))
        torch.save({'global_step': global_step,
        'coarse_state_dict': coarse_model.state_dict(),
        'point_emb_state_dict': point_embedding_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},path)

        print('Saved checkpoints at',path)
