import numpy as np
import os
import torch
torch.manual_seed(5)
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

from utils import get_rays_np

validation_phase=0
device=torch.device("cpu")

"""things to look at carefully
1:hashing function in utils"""

"""list of changes:
1:changed direction embedding from spherical harmonics to standard hash embedder maybe spheircal harmonics doesn't need to be optimized, 
2:using Adam optimizer instead of rectified Adam"""


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # Calculate weights sparsity loss
    mask = weights.sum(-1) > 0.5
    entropy = torch.distributions.Categorical(probs = weights+1e-5).entropy()
    sparsity_loss = entropy * mask

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

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

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*16):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

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
        r1=torch.ones(z_vals.shape,dtype=r_origin.dtype,device=r_origin.device)########################
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
    print("################### MY ###############################")
    print(coarse_rgb_map.shape)
    print(coarse_disp_map.shape)
    print(coarse_acc_map.shape)
    print(weights.shape)
    print(coarse_depth_map.shape)
    print(sparsity_loss)
    # return (coarse_rgb_map,coarse_disp_map,coarse_acc_map,weights,coarse_depth_map,sparsity_loss,radiance_field)


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
    # for j in pred:
    #     print(j.shape)


    # print(rays.shape)
    # return (rays)



def render(H,W,K,chunk=1024*16,rays=None,c2w=None,ndc=True,near=0.,far=1.,use_viewdirs=True,c2w_staticcam=None,network_fn=None,network_query_fn=None,
            embed_fn=None):
    
    rays_o,rays_d=rays
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    
    # print(viewdirs.shape)
    # print(viewdirs[1])

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3])
    rays_d = torch.reshape(rays_d, [-1,3])

    # print("rayso:",rays_o.shape)
    # print("raysd:",rays_d.shape)

    # return(rays_o,rays_d)

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    
    # print(rays.shape)
    # return (rays)

    # Render and reshape
    all_ret = batchify_rays(rays,network_fn,network_query_fn,embed_fn,chunk)
    
    # render_kwargs_train = {'network_query_fn' : network_query_fn,'perturb' : args.perturb,'N_importance' : args.N_importance,
    # network_fine' : model_fine,'N_samples' : args.N_samples,'network_fn' : model,'embed_fn': embed_fn,'use_viewdirs' : args.use_viewdirs,
# 'white_bkgd' : args.white_bkgd,'raw_noise_std' : args.raw_noise_std,}
    # for k in all_ret:
    #     k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
    #     all_ret[k] = torch.reshape(all_ret[k], k_sh)

#     k_extract = ['rgb_map', 'disp_map', 'acc_map']
#     ret_list = [all_ret[k] for k in k_extract]
#     ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
#     return ret_list + [ret_dict] 


def batchify_rays(rays_flat,network_fn,network_query_fn,embed_fn,chunk=1024*16):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}

    print("SSSS:",rays_flat[0])
    for i in range(0, rays_flat.shape[0], chunk):
        # print("ref:",rays_flat[i:i+chunk].shape)
        ret = render_rays(rays_flat[i:i+chunk],network_fn=coarse_model,network_query_fn=network_query_fn,embed_fn=point_embedding_model)
    #     for k in ret:
    #         if k not in all_ret:
    #             all_ret[k] = []
    #         all_ret[k].append(ret[k])
    
    # all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    # return all_ret

# render_kwargs_train = {'network_query_fn' : network_query_fn,'perturb' : args.perturb,'N_importance' : args.N_importance,
# network_fine' : model_fine,'N_samples' : args.N_samples,'network_fn' : model,'embed_fn': embed_fn,'use_viewdirs' : args.use_viewdirs,
# 'white_bkgd' : args.white_bkgd,'raw_noise_std' : args.raw_noise_std,}

def render_rays(ray_batch,network_fn,network_query_fn,embed_fn,N_samples=64,retraw=True,
                lindisp=False,perturb=1.,N_importance=0,network_fine=None,white_bkgd=False,raw_noise_std=0.,verbose=True,pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # print(near,far)

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    # print(z_vals)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.ones(z_vals.shape).to("cpu")

        # Pytest, overwrite u with numpy's fixed random numbers
        # if pytest:
        #     np.random.seed(0)
        #     t_rand = np.random.rand(*list(z_vals.shape))
        #     t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    
    # print("ref z_vals",z_vals)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    # print("pts",pts)
    # print(pts.shape)

#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    # print("network op:",raw.shape)
    # print(raw[1,1,:])
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss=raw2outputs(raw,z_vals,rays_d,raw_noise_std,white_bkgd, pytest=pytest)
    print("############REF###################")
    print(rgb_map.shape)
    print(disp_map.shape)
    print(acc_map.shape)
    print(weights.shape)
    print(depth_map.shape)
    print(sparsity_loss)



#     ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss}
#     if retraw:
#         ret['raw'] = raw
#     if N_importance > 0:
#         ret['rgb0'] = rgb_map_0
#         ret['disp0'] = disp_map_0
#         ret['acc0'] = acc_map_0
#         ret['sparsity_loss0'] = sparsity_loss_0
#         ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

#     for k in ret:
#         if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
#             print(f"! [Numerical Error] {k} contains nan or inf.")

    # return ret

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
    render(height,width,K,chunk=16384,rays=rays_batch,network_fn=coarse_model,
            network_query_fn=network_query_fn,embed_fn=point_embedding_model)

    run_1_nerf(height,width,focal_length,coarse_model,None,ro,rd,mode,point_embedding_model,dir_embedding_model)
    # assert(torch.equal(r1,r2))
    # rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
    #                                             verbose=i < 10, retraw=True,
    #                                             **render_kwargs_train)




# for i in trange(start, N_iters):
#         # Sample random ray batch
#         if use_batching:
#             # Random over all images
#             batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
#             batch = torch.transpose(batch, 0, 1)
#             batch_rays, target_s = batch[:2], batch[2]

#             i_batch += N_rand
#             if i_batch >= rays_rgb.shape[0]:
#                 print("Shuffle data after an epoch!")
#                 rand_idx = torch.randperm(rays_rgb.shape[0])
#                 rays_rgb = rays_rgb[rand_idx]
#                 i_batch = 0

#         i_batch = 0

    # render_kwargs_train = {
    #     'network_query_fn' : network_query_fn,
    #     'perturb' : args.perturb,
    #     'N_importance' : args.N_importance,
    #     'network_fine' : model_fine,
    #     'N_samples' : args.N_samples,
    #     'network_fn' : model,
    #     'embed_fn': embed_fn,
    #     'use_viewdirs' : args.use_viewdirs,
    #     'white_bkgd' : args.white_bkgd,
    #     'raw_noise_std' : args.raw_noise_std,
    # }

    # # NDC only good for LLFF-style forward facing data
    # if args.dataset_type != 'llff' or args.no_ndc:
    #     print('Not ndc!')
    #     render_kwargs_train['ndc'] = False
    #     render_kwargs_train['lindisp'] = args.lindisp

    # render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    # render_kwargs_test['perturb'] = False
    # render_kwargs_test['raw_noise_std'] = 0.

    # return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# global_step=start

