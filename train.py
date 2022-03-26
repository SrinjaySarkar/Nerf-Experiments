import numpy as np
import torch
import tqdm
import torch.nn.functional as F
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

from load_data import load_llff
from model import get_embedding_function,nerf2
from utils import get_image_rays,meshgrid,ndc_rays,get_minibatches,cumprod,sampling,psnr_loss,mse

np.random.seed(42)
torch.manual_seed(42)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epochs=20#250000
print_every_n=100
save_every_n=2000


#need to check : disparity map(disp_map in volume render,not taking ; check all 3 functions against 
#https://github.com/bmild/nerf/blob/master/run_nerf.py and how points are sampled exactly both coarse and fine in pred _radiance)

def volume_renderer(radiance_field,depth,ray_direction,noise,white_bg):
    one_e_10=torch.tensor([1e10],dtype=ray_direction.dtype,device=ray_direction.device)
    dists=depth[:,1:]-depth[:,:-1] #(4096,63)
    dists=torch.cat((dists,one_e_10.expand(depth[:,:1].shape)),dim=-1)#4096,64
    dists=dists*ray_direction.unsqueeze(1).norm(p=2,dim=-1)
    # dists=dists*ray_direction.unsqueeze(1)
    
    rgb=torch.sigmoid(radiance_field[...,:3])   
    density=radiance_field[...,3]
    noise=torch.randn(density.shape,dtype=density.dtype,device=density.device)*noise
    sigma=F.relu(density+noise)

    # print(sigma.shape)
    # print(dists.shape)
    alpha=1.0-torch.exp(-sigma*dists)
    
    # weights=alpha*torch.cumprod()
    weights=alpha*cumprod(1.-alpha+1e-10)#(Ti*(1-exp(-sigma*delta)))see eq 3
    rgb_map=rgb*weights.clone().unsqueeze(2)
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
    radiance_field=radiance_field.reshape(points.shape[0],points.shape[1],radiance_field.shape[-1])#(4096,64,4)
    # print(radiance_field.shape)
    return (radiance_field)


# ((ro,rd,near,far,viewdirs),dim=-1)#4096,11(3+3+1+1+3)
def pred_radiance(rays,coarse_model,fine_model,mode,encode_pos,encode_dir):
    n_rays=rays.shape[0]
    r_origin,r_direction=rays[:,:3],rays[:,3:6]
    bounds=rays[:,6:8].view((-1,1,2))
    near,far=bounds[:,:,0],bounds[:,:,1]

    coarse_points=torch.linspace(0.0,1.0,64,dtype=r_origin.dtype,device=r_origin.device) 

    z_vals=near*(1.0-coarse_points)+far*coarse_points#Sample linearly in disparity space, as opposed to in depth space.
    z_vals=z_vals.expand([n_rays,64])
    # print("Z_vals:",z_vals.shape)

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
    # coarse_rays=r_origin[..., None, :] + r_direction[..., None, :] * z_vals[..., :, None]
    radiance_field=nerf_net(coarse_model,coarse_rays,rays,16384,encode_pos,encode_dir)
    #coarse_rgb,caorse_disp,coarse_acc,weights,coarse_depth
    coarse_rgb_map,coarse_disp_map,coarse_acc_map,weights,coarse_depth_map=volume_renderer(radiance_field,z_vals,r_direction,1.,False)


    #fine points
    z_vals_mid=0.5*(z_vals[...,1:]+z_vals[...,:-1])#points between the points
    fine_points=sampling(z_vals_mid,weights[...,1:-1],nf=128,det=False)#all weights  except 1st and lst since we only take points in betwween
    fine_points=fine_points.detach()
    z_vals,_=torch.sort(torch.cat((z_vals,fine_points),dim=-1),dim=-1)

    ro2=r_origin.unsqueeze(1).expand(r_origin.shape[0],128+64,3)
    rd2=r_direction.unsqueeze(1).expand(r_direction.shape[0],128+64,3)
    z2=z_vals.unsqueeze(2).expand(z_vals.shape[0],128+64,3)
    fine_rays=ro2+rd2*z2

    radiance_field=nerf_net(fine_model,fine_rays,rays,16384,encode_pos,encode_dir)
    #coarse_rgb,caorse_disp,coarse_acc,weights,coarse_depth
    fine_rgb_map,fine_disp_map,fine_acc_map,_,fine_depth_map=volume_renderer(radiance_field,z_vals,r_direction,1.,False)

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
    near=0*torch.ones_like(rd[:,:1])
    far=1*torch.ones_like(rd[:,:1])
    
    rays=torch.cat((ro,rd,near,far,viewdirs),dim=-1)#4096,11(3+3+1+1+3)
    # print(rays.shape)

    batches=get_minibatches(rays,chunksize=16384)
    pred=[pred_radiance(batch,coarse_model,fine_model,"train",encode_pos,encode_dir) for batch in batches]
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
       

#dataset
images,poses,bds,render_poses,test_idx=load_llff(basedir="/vinai/sskar/NERF/nerf_llff_data/fern",factor=8)
llffhold=8
hwf=poses[0,:3,-1]
poses=poses[:,:3,:4]#R|t

if not isinstance(test_idx,list):   
    test_idx=[test_idx]
    if llffhold>0:
        test_idx=np.arange(images.shape[0])[::llffhold]
    val_idx=test_idx
    train_idx=np.array([i for i in np.arange(images.shape[0]) if (i not in test_idx and i not in val_idx)])

# print("Training images:",test_idx)
# print("Validation images:",val_idx)
# print("Testing images:",train_idx)
height,width,focal=hwf
height,width=int(height),int(width)
hwf=[height,width,focal]
images=torch.from_numpy(images)
poses=torch.from_numpy(poses)


#model 
encode_postion=get_embedding_function(L=6,include_input=True,log_sampling=True)
encode_direction=get_embedding_function(L=4,include_input=True,log_sampling=True)


coarse_model=nerf2(num_layers=4,hidden_size=64,skip_connect_every=3 ,L_xyz=6,L_dir=4,include_xyz=True,include_dir=True,use_viewdirs=True).to(device)
coarse_model=coarse_model.train()
fine_model=nerf2(num_layers=4,hidden_size=64,skip_connect_every=3,L_xyz=6,L_dir=4,include_xyz=True,include_dir=True,use_viewdirs=True).to(device)
fine_model=fine_model.train()

trainable_parameters=list(coarse_model.parameters())+list(fine_model.parameters())
optimizer=torch.optim.Adam(trainable_parameters,lr=5e-3)

#TRAINING

for i in range(0,n_epochs):
    # coarse_model=coarse_model.train()
    # fine_model=fine_model.train()

    img_idx=np.random.choice(train_idx)
    img_target=images[img_idx].to(device)
    pose_target=poses[img_idx,:3,:4].to(device)
    
    ray_origin,ray_direction=get_image_rays(height,width,focal,pose_target)
    # print(ray_origin.shape)
    # print(ray_direction.shape)

    coords=torch.stack(meshgrid(torch.arange(height).to(device),torch.arange(width).to(device)),dim=-1)
    coords=coords.reshape((-1,2))
    # print(coords.shape)
    
    select_inds=np.random.choice(coords.shape[0],size=(4096),replace=False) 
    select_inds=coords[select_inds]
    # print(select_inds.shape)

    ray_origin=ray_origin[select_inds[:,0],select_inds[:,1],:]
    ray_direction=ray_direction[select_inds[:,0],select_inds[:,1],:]

    # print(ray_origin.shape)
    # print(ray_direction.shape)  

    target_s=img_target[select_inds[:,0],select_inds[:,1],:]

    rgb_coarse,_,_,rgb_fine,_,_=run_1_nerf(height,width,focal,coarse_model,fine_model,ray_origin,ray_direction,"train",encode_postion,encode_direction)
    # print(rgb_coarse.shape)
    # print(rgb_fine.shape)

    gt_vals=target_s
    coarse_loss=F.mse_loss(rgb_coarse[:,:3],gt_vals[:,:3])
    # coarse_loss=mse(rgb_coarse[:,:3],gt_vals[:,:3])
    print("coarse loss:",coarse_loss.item())
    fine_loss=F.mse_loss(rgb_fine[:,:3],gt_vals[:,:3])
    print("fine loss:",fine_loss.item())
    # fine_loss=mse(rgb_fine[:,:3],gt_vals[:,:3])
    loss=fine_loss+coarse_loss
    loss.backward()
    print("total loss:",loss.item())
    psnr=psnr_loss(loss.item())

    
    optimizer.step()
    optimizer.zero_grad()


    # n_decay_steps=250*1000
    # lr_new=5e-3*(0.1**(i/n_decay_steps))

    # for param_group in optimizer.param_groups:
    #     param_group["lr"]=lr_new

    # if i%print_every_n==0 or i==n_epochs-1:
    #     tqdm.write("[TRAIN] Iter:"+ str(i) + "Loss:" + str(loss.item()) + " PSNR: "+ str(psnr))
    

    # #save model
    # if i%save_every_n==0 or i==n_epochs-1:
    #     ckpt_dict={"epoch":i,"coarse_model_dict":coarse_model.state_dict(),"fine_model_dict":fine_model.state_dict(),
    #                 "optimizer_state_dict":optimizer.state_dict(),"loss":loss,"psnr":psnr}
    #     torch.save(ckpt_dict,os.path.join("/vinai/sskar/NERF/ckpts/","checkpoint",str(i),".ckpt"))
    #     print("############## Saved Checkpoint ##############")
