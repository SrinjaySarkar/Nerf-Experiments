#meshgrid,ray_bundle,pos encdoing,ndc rays,sampled pdf 
import numpy as np
import torch
import math
import torch.nn.functional as F
device=torch.device("cpu")
# from load_data import load_llff

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o,rays_d    


def meshgrid(input1,input2):
    ii,jj=torch.meshgrid(input1,input2)
    ii=ii.transpose(-1,-2)
    jj=jj.transpose(-1,-2)
    return (ii,jj)

def get_image_rays(height,width,focal_length,c2w):
    """this is the same operation like in implicit SLAM where pixels are mapped into 3d space. p=(R(K_inv(pix))). unlike in implicit SLAM do this 
    for all pixels of the image.since the y value indexes from top to bottom, we flip it, and since the camera looks along the negative z axis, 
    we negative it The /2 comes from the fact that the optical center is located at the center pixel."""

    #matmul(k_inv,pix)
    ii,jj=meshgrid(torch.arange(width,dtype=c2w.dtype,device=c2w.device),
                   torch.arange(height,dtype=c2w.dtype,device=c2w.device))
    ray_directions=torch.stack([(ii-width*0.5)/focal_length,-(jj-height*0.5)/focal_length,-torch.ones_like(ii)],dim=-1)#K_inv*pixel
    ray_directions=c2w[:3,:3]*ray_directions.unsqueeze(2)#R*(K_inv(pixel))
    # print(ray_directions.shape)
    # ray_directions3=torch.matmul(c2w[:3,:3],ray_directions[2,3,:,None])
    # assert (torch.equal(ray_directions1,ray_directions2))
    # print(ray_directions1.shape)
    # print(ray_directions2.shape)
    ray_directions=torch.sum(ray_directions,dim=-1)
    ray_origins=c2w[:3,-1].expand(ray_directions.shape)
    return (ray_origins,ray_directions)





def ndc_rays(height,width,focal,near,r_origin,r_direction):
    """read ndc derivation"""
    #shift rays origin to near plane
    t=-(near+r_origin[...,2])/r_direction[...,2]
    r_origin=r_origin+t[...,None]*r_direction
    #project
    o0=-1.0/(width/(2.0*focal))*r_origin[...,0]/r_origin[...,2]
    o1=-1.0/(height/(2.0*focal))*r_origin[...,1]/r_origin[...,2]
    o2=1.0+2.0*near/r_origin[...,2]
    d0=(-1.0/(width/(2.0*focal))*(r_direction[..., 0]/r_direction[..., 2]-r_origin[..., 0]/r_origin[..., 2]))
    d1=(-1.0/(height/(2.0*focal))*(r_direction[...,1]/r_direction[..., 2]-r_origin[..., 1]/r_origin[..., 2]))
    d2=-2.0*near/r_origin[..., 2]
    r_origin=torch.stack([o0,o1,o2],-1)
    r_direction=torch.stack([d0,d1,d2],-1)

    return (r_origin,r_direction)


def sampling(bins,weights,nf,det): # try naive hierarchical sampling
    weights+=1e-5#avoid underflow
    # print(weights.shape)
    # print(torch.sum(weights,dim=-1,keepdim=True).shape)
    pdf=weights/(torch.sum(weights,dim=-1,keepdim=True))
    # print(pdf.shape)
    cdf=torch.cumsum(pdf,dim=-1)#get cdf by integrating pdf
    # print(cdf.shape)
    cdf=torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1)
    #inverse transform sampling
    if det:
        u=torch.linspace(0.0,1.0,steps=nf,dtype=weights.dtype,device=weights.device)
        u=u.expand(list(cdf.shape[:-1])+[nf])
    else:
        u=torch.rand(list(cdf.shape[:-1])+[nf],dtype=weights.dtype,device=weights.device)#unformly sampled 
    # print(u.shape)
    u=u.contiguous()
    cdf=cdf.contiguous()
    idxs=torch.searchsorted(cdf,u,right=True)
    low=torch.max(torch.zeros_like(idxs-1),idxs-1)
    high=torch.min((cdf.shape[-1]-1)*torch.ones_like(idxs),idxs)
    idxs_g=torch.stack((low,high),dim=-1)

    

    matched_shape=(idxs_g.shape[0],idxs_g.shape[1],cdf.shape[-1])
    cdf_g=torch.gather(cdf.unsqueeze(1).expand(matched_shape),2,idxs_g)
    bins_g=torch.gather(bins.unsqueeze(1).expand(matched_shape),2,idxs_g)
    
    denom=cdf_g[...,1]-cdf_g[...,0]
    denom=torch.where(denom<1e-5,torch.ones_like(denom),denom)
    t=(u-cdf_g[...,0])/denom
    samples=bins_g[...,0]+t*(bins_g[...,1]-bins_g[...,0])
    # samples_cat,_=torch.sort(torch.cat([samples,bins],-1),dim=-1)
    return(samples)

def get_minibatches(inputs,chunksize=1024*8):
    return [inputs[i:i+chunksize] for i in range(0,inputs.shape[0],chunksize)]


def cumprod(tensor: torch.Tensor) -> torch.Tensor:
    dim = -1
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.0
    return cumprod

def mse(src,target):
    loss=F.mse_loss(src,target)
    return (loss)

def psnr_loss(mse):
    if mse==0:
        mse=1e-5
    psnr_val=-10.0*math.log10(mse)
    return (psnr_val)

def hash_obj(points,hashmap_size):
    primes=[1,2654435761,805459861,3674653429,2097192037,1434869437,2165219737]
    xor_result=torch.zeros_like(points)[...,0]
    
    #per dimension xor
    for i in range(points.shape[-1]):
        xor_result^=points[...,i]*primes[i]#equation 4 in hash nerf paper
    
    #xor_result mod hashmap_size
    #(xor_result % T)
    return (torch.tensor((1<<hashmap_size)-1).to(xor_result.device)&xor_result)


BOX_OFFSETS=torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device=device)

def voxel_vertices(xyz,bounding_box,resolution,hashmap_size):
    box_min,box_max=bounding_box
    box_max=box_max.to(device)
    box_min=box_min.to(device)
    
    if not torch.all(xyz<=box_max) or not torch.all(xyz>=box_min):
        xyz=torch.clamp(xyz,min=box_min,max=box_max)
    grid_size=(box_max-box_min)/resolution

    bottom_left_idx=torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex=bottom_left_idx*grid_size+box_min
    voxel_max_vertex=voxel_min_vertex+torch.tensor([1.0,1.0,1.0]).to(device)*grid_size
    voxel_indices=bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices=hash_obj(voxel_indices,hashmap_size)

    return (voxel_min_vertex,voxel_max_vertex,hashed_voxel_indices)





# height=504
# width=378
# focal_length=512
# i,p,b,sp,tdx=load_llff(basedir="/vinai/sskar/NERF/nerf_llff_data/fern")
# p=torch.from_numpy(p)
# p1=p[tdx,:3,:4].to("cpu")
# o,d=get_image_rays(height,width,focal_length,p1)
# print(o.shape)
# print(d.shape)
# point=torch.rand(1,5,3)
# pos_encoding(input=point,L=10,include_input=True,log_sampling=True)
