import numpy as np
import os
import torch
from tqdm import tqdm
import imageio
import cv2
import time
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from load_data import load_llff
from model import get_embedding_function,nerf2,FlexibleNeRFModel,get_embedding_function
from utils import get_image_rays,meshgrid,ndc_rays,get_minibatches,cumprod,sampling,psnr_loss,mse

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path="/vinai/sskar/NERF/ckpts/checkpoint.ckpt"
img_save_path="/vinai/sskar/NERF/gen_imgs/"
depth_save_path="/vinai/sskar/NERF/depth_imgs/"


def volume_renderer(radiance_field,depth,ray_direction,noise,white_bg):
    one_e_10=torch.tensor([1e10],dtype=ray_direction.dtype,device=ray_direction.device)
    dists=depth[:,1:]-depth[:,:-1] #(4096,63)
    dists=torch.cat((dists,one_e_10.expand(depth[:,:1].shape)),dim=-1)#4096,64 (adding 1e10 as the distance for the last sample point; this means the the reiman sum with the last distance equals a big number)
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
    rgb_map=rgb*weights.unsqueeze(2)
    rgb_map=torch.sum(rgb_map,dim=-2)#(n_rays,3(4096,3))
    depth_map=depth*weights
    depth_map=torch.sum(depth_map,dim=-1)
    acc_map=weights.sum(dim=-1)
    disp_map=1.0/torch.max(1e-10*torch.ones_like(depth_map),depth_map/acc_map) # inverse of depth map
    if white_bg:
        rgb_map=rgb_map+(1.0-acc_map[...,None])
    # print(rgb_map.shape,disp_map.shape,acc_map.shape,depth_map.shape)
    return (rgb_map,disp_map,acc_map,weights,depth_map)
    # print(weights.shape)


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


# ((ro,rd,near,far,viewdirs),dim=-1)#4096,11(3+3+1+1+3)
def pred_radiance(rays,coarse_model,fine_model,mode,encode_pos,encode_dir):
    n_rays=rays.shape[0]
    r_origin,r_direction=rays[...,:3],rays[...,3:6]
    bounds=rays[...,6:8].view((-1,1,2))
    near,far=bounds[...,0],bounds[...,1]

    coarse_points=torch.linspace(0.0,1.0,64,dtype=r_origin.dtype,device=r_origin.device)    

    z_vals=near*(1.0-coarse_points)+far*coarse_points#Sample linearly in disparity space, as opposed to in depth space.
    z_vals=z_vals.expand([n_rays,64])
    # print("Z_vals:",z_vals.shape)

    if mode=="train":
        mids=0.5*(z_vals[...,1:]+z_vals[...,:-1])#samples between every set of coarse points 
        # print("Mids:",mids.shape)
        # print(z_vals[0,1:],z_vals[0,:-1])
        # print(mids[0,0])
        upper=torch.cat((mids,z_vals[...,-1:]),dim=-1)#appending last and first points to the interval samples
        lower=torch.cat((mids,z_vals[...,:1]),dim=-1)
        r1=torch.rand(z_vals.shape,dtype=r_origin.dtype,device=r_origin.device)
        z_vals=lower+(upper-lower)*r1 #stratified sampling (eq2 of paper)#4096,64
    # print("Z_vals:",z_vals.shape)
    #o+td
    # print(r_origin.shape)#4096,3
    # print(r_direction.shape)#4096,3
    #origin(4096,64,3),direction(4096,64,3),points(4096,64,3)
    
    ro1=r_origin.unsqueeze(1).expand(r_origin.shape[0],64,3)
    rd1=r_direction.unsqueeze(1).expand(r_direction.shape[0],64,3)
    z1=z_vals.unsqueeze(2).expand(z_vals.shape[0],64,3)
    coarse_rays=ro1+rd1*z1
    radiance_field=nerf_net(coarse_model,coarse_rays,rays,16384,encode_pos,encode_dir)
    #coarse_rgb,caorse_disp,coarse_acc,weights,coarse_depth
    coarse_rgb_map,coarse_disp_map,coarse_acc_map,weights,coarse_depth_map=volume_renderer(radiance_field,z_vals,r_direction,0,False)


    #fine points
    z_vals_mid=0.5*(z_vals[...,1:]+z_vals[...,:-1])#points between the points
    fine_points=sampling(z_vals_mid,weights[...,1:-1],nf=64,det=True)#all weights  except 1st and lst since we only take points in betwween
    fine_points=fine_points.detach()
    z_vals,_=torch.sort(torch.cat((z_vals,fine_points),dim=-1),dim=-1)

    ro2=r_origin.unsqueeze(1).expand(r_origin.shape[0],128,3)
    rd2=r_direction.unsqueeze(1).expand(r_direction.shape[0],128,3)
    z2=z_vals.unsqueeze(2).expand(z_vals.shape[0],128,3)
    fine_rays=ro2+rd2*z2
    radiance_field=nerf_net(fine_model,fine_rays,rays,16384,encode_pos,encode_dir)
    #coarse_rgb,caorse_disp,coarse_acc,weights,coarse_depth
    fine_rgb_map,fine_disp_map,fine_acc_map,_,fine_depth_map=volume_renderer(radiance_field,z_vals,r_direction,0,False)
    # fine_rgb_map,fine_disp_map,fine_acc_map=None,None,None

    return (coarse_rgb_map,coarse_disp_map,coarse_acc_map,fine_rgb_map,fine_disp_map,fine_acc_map)


def run_1_nerf(height,width,focal_length,coarse_model,fine_model,ray_origin,ray_direction,mode,encode_pos,encode_dir):
    viewdirs=ray_direction#(directions64*64*3 pixels)
    viewdirs=viewdirs/viewdirs.norm(p=2,dim=-1).unsqueeze(-1)
    # print(torch.norm(viewdirs,dim=-1))
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
    
    rays=torch.cat((ro,rd,near,far,viewdirs),dim=-1)#4096,11(3+3+1+1+3)
    # print(rays.shape)

    batches=get_minibatches(rays,chunksize=16384)
    pred=[pred_radiance(batch,coarse_model,fine_model,"val",encode_pos,encode_dir) for batch in batches]
    synthesized_images=list(zip(*pred))
    synthesized_images=[torch.cat(image,dim=0) if image[0] is not None else (None) for image in synthesized_images]
    # print(len(synthesized_images))

    if mode=="val":
        synthesized_images=[image.view(shape) if image is not None else None for (image,shape) in zip(synthesized_images,restore_shapes)]
        if fine_model:
            return(tuple(synthesized_images))
        else:
            tuple(synthesized_images+[None,None,None])
    
    return (tuple(synthesized_images))


images,poses,bds,render_poses,test_idx=load_llff(basedir="/vinai/sskar/NERF/nerf_llff_data/fern",factor=8)
hwf=poses[0,:3,-1]
H,W,focal=hwf
hwf=[int(H),int(W),focal]
render_poses=torch.from_numpy(render_poses) 

#model 
encode_postion=get_embedding_function(num_encoding_functions=6,include_input=True,log_sampling=True)#get_embedding_function(L=6,include_input=True,log_sampling=True)
encode_direction=get_embedding_function(num_encoding_functions=4,include_input=True,log_sampling=True)#get_embedding_function(L=4,include_input=True,log_sampling=True)

# num_layers=4,hidden_size=128,skip_connect_every=4,num_encoding_fn_xyz=6,num_encoding_fn_dir=4,include_input_xyz=True,include_input_dir=True,use_viewdirs=True
coarse_model=FlexibleNeRFModel(num_layers=4,hidden_size=64*2,skip_connect_every=4,num_encoding_fn_xyz=6,num_encoding_fn_dir=4,include_input_xyz=True,include_input_dir=True,use_viewdirs=True
).to(device)

fine_model=FlexibleNeRFModel(num_layers=4,hidden_size=64*2,skip_connect_every=4,num_encoding_fn_xyz=6,num_encoding_fn_dir=4,include_input_xyz=True,include_input_dir=True,use_viewdirs=True
).to(device)

checkpoint=torch.load(weights_path,map_location="cpu")
print(checkpoint.keys())
coarse_model.load_state_dict(checkpoint["model_coarse_state_dict"])
fine_model.load_state_dict(checkpoint["model_fine_state_dict"])

print("loaded weights for fern")
coarse_model=coarse_model.eval()
fine_model=fine_model.eval()

render_poses=render_poses.float().to(device)

for idx,pose in enumerate(tqdm(render_poses)):
    with torch.no_grad():
        pose=pose[:3,:4]
        ray_origin,ray_direction=get_image_rays(hwf[0],hwf[1],hwf[2],pose)
        rgb_coarse,disp_coarse,_,rgb_fine,disp_fine,_=run_1_nerf(hwf[0],hwf[1],hwf[2],coarse_model,fine_model,ray_origin,ray_direction,"val",
        encode_postion,encode_direction)
        rgb_img=rgb_fine#(378,504,3)
        disp=disp_fine
    
    #save rgb image 
    rgb=rgb_img[:,:,:3]
    rgb=rgb.permute(2,0,1)
    img=np.array(torchvision.transforms.ToPILImage()(rgb.detach().cpu()))
    savefile=os.path.join(img_save_path,str(idx)+".png")
    imageio.imwrite(savefile,img)

    #save depth image
    disp_img=disp
    disp_img=(disp_img-disp_img.min())/(disp_img.max()-disp_img.min())
    disp_img=disp_img.clamp(0,1)*255
    disp_img=disp_img.detach().cpu().numpy().astype(np.uint8)
    savefile=os.path.join(depth_save_path,str(idx)+".png")
    imageio.imwrite(savefile,disp_img) 
