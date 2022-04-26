import numpy as np
import os
import imageio
from shutil import copy
from subprocess import check_output
from utils import *

def get_bbox(poses,hwf,near=0.0,far=1.0): #this is done for all posed images at once
    height,width,focal_length=hwf
    height,width=int(height),int(width)

    min_bound=[200,200,200]
    max_bound=[-200,-200,-200]

    points=[]
    poses=torch.FloatTensor(poses)
    for pose in poses:
        ray_origin,ray_direction=get_image_rays(height,width,focal_length,pose)
        ray_origin=ray_origin.view(-1,3)
        ray_direction=ray_direction.view(-1,3)
        ray_direction=ray_direction/ray_direction.norm(p=2,dim=-1).unsqueeze(-1)

        ro,rd=ndc_rays(height,width,focal_length,1.0,ray_origin,ray_direction)

        def min_max(pt):
            for i in range(3):
                if(min_bound[i]>pt[i]):
                    min_bound[i]=pt[i]
                if(max_bound[i]<pt[i]):
                    max_bound[i]=pt[i]
            return
        for i in [0,width-1,height*width-width,height*width-1]:
            #4 corner pixels of the image , bounds for all 
            min_point=ro[i]+near*rd[i]
            max_point=ro[i]+far*rd[i]

            min_max(min_point)
            min_max(max_point)
    
    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]),torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))
    
    
def imread(f):
    if f.endswith("png"):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def minify(basedir,factors=[],resolutions=[]):
    needtoload=False
    for r in factors:
        imgdir=os.path.join(basedir,'images_{}'.format(r))
        if not os.path.join(imgdir):
            needtoload=True
    for r in resolutions:
        imgdir=os.path.join(basedir,'images_{}x{}'.format(r[1],r[0]))
        if not os.path.exists(imgdir):
            needtoload=True
    if not needtoload:
        return 
    imgdir=os.path.join(basedir,"images")
    imgs=[os.path.join(imgdir,f) for f in sorted(os.listdir(imgdir))]
    imgs=[f for f in imgs if any([f.endswith(ex) for ex in ["JPG","jpg","png","jpeg","PNG"]])]
    imgdir_orig=img_img_dir
    wd=os.getcwd()

    for r in (factors+resolutions):
        if isinstacne(r,int):
            name="images_{}".format(r)
            resizearg="{}%".format(100.0/r)
        else:
            name="images_{}x{}".format(r[1],r[0])
            resizearg="{}x{}".fromat(r[1],r[0])
        imgdir=os.path.join(basedir,name)
        if os.path.exists(imgdir):
            continue
        
        print("Minifying",r,basedir)
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig,imgdir), shell=True)  

        ext=imgs[0].split(".")[-1]
        args=' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext!="png":
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print("removed dups")
        print("done")



def _load_data(basedir,factor=None,width=None,height=None,load_imgs=True):
    poses_arr=np.load(os.path.join(basedir,"poses_bounds.npy")) #(20,17)
    # print("POSE ARRAY SHAPE:",poses_arr.shape)
    poses=poses_arr[:,:-2].reshape([-1,3,5,]).transpose([1,2,0])#(3,5,20)
    #(3,5) matrices for 20 images (3,4 matrices for[R|t] and 1 column for [HWF] so a 3,5 matrix).    
    # ex=poses[:,4,:]
    # print("HWF:",ex.shape)
    # print(ex[:,1])
    bds=poses_arr[:,-2:].transpose([1,0])# the last 2 elements are the near and far bounds for every image. 
    imgs=[os.path.join(basedir,"images",f) for f in sorted(os.listdir(os.path.join(basedir,"images"))) if f.endswith("JPG") or f.endswith(".jpg") or f.endswith("png")][0]
    sh=imageio.imread(imgs).shape
    sfx = ""
    # print("Image shape orginal:",sh)
    if factor is not None :
        sfx="_{}".format(factor)
        minify(basedir,factors=[factor])
        factor=factor
    elif height is not None:
        factor=sh[0] /float(height)
        width=int(sh[1]/factor)
        minify(basedir,resoltuions=[[height,width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor=sh[1]/float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor=1
    # print(sfx)

    imgdir=os.path.join(basedir,"images"+sfx)
    if not os.path.exists(imgdir):
        print(imgdir,"does not exist, returning")
        return

    imgfiles=[os.path.join(imgdir,f) for f in sorted(os.listdir(imgdir)) if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")]
    if poses.shape[-1]!=len(imgfiles):
        print("mismatch between number of poses and images")
        return 

    sh=imageio.imread(imgfiles[0]).shape
    #replace with scaled height and scaled width and replace with scaled focal length
    poses[:2,4,:]=np.array(sh[:2]).reshape([2,1])#replacing scaled height and width
    poses[2,4,:]=poses[2,4,:]*1.0/factor #replace scaled focal length
    if not load_imgs:
        return (poses,bds)
    
    imgs=[imread(f)[...,:3]/255.0 for f in imgfiles]
    imgs=np.stack(imgs,axis=-1)
    return (poses,bds,imgs)

def recenter_pose(poses):
    """applies the inverse of this average pose to the dataset (a rigid rotation/translation) so that the identity extrinsic matrix 
    is looking at the scene, which is nice because normalizes the orientation of the scene for later rendering."""
    poses_=poses+0
    bottom=np.reshape([0,0,0,1.0],[1,4])
    c2w=avg_poses(poses)
    #c2w is the mean pose of the entire dataset
    c2w=np.concatenate([c2w[:3,:4],bottom],-2)#convert from 3x4 extrinsic to 4x4 extrinsic matrix
    bottom=np.tile(np.reshape(bottom,[1,1,4]),[poses.shape[0],1,1])
    poses=np.concatenate([poses[:,:3,:4],bottom],-2)#need to check this again?
    poses=np.linalg.inv(c2w)@poses
    poses_[:,:3,:4]=poses[:,:3,:4]
    poses=poses_
    return (poses)

def normalize(x):
    x_norm=x/np.linalg.norm(x)
    return (x_norm)

def view_matrix(z,up,pos):
    """calculates the central mean pose for the dataset based on mean translation(center);the mean z-axis(vec2);
        adopting mean y-axis(up) as up direxction so that the cross(up,z)=x and cross(z,x)=y. then rearrange the R matrix according to this."""
    #step 4
    vec2=normalize(z)
    vec1_avg=up
    vec0=normalize(np.cross(vec1_avg,vec2))
    #step 5
    vec1=normalize(np.cross(vec2,vec0))
    mat=np.stack([vec0,vec1,vec2,pos],1)
    return (mat)

def avg_poses(poses):
    """calculates the central mean pose for the dataset based on mean translation(center);the mean z-axis(vec2);
        adopting mean y-axis(up) as up direxction so that the cross(up,z)=x and cross(z,x)=y. then rearrange the R matrix according to this."""
    """Calculate the average pose, which is then used to center all poses
    using recenter_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x."""
    hwf=poses[0,:3,-1:]
    #step1
    center=poses[:,:3,3].mean(0)#(tx+ty+tz) is treated as the center;this is the same in Imap slam wher the ray origin was the translation vector.
    #step2
    vec2=poses[:,:3,2].sum(0)#(r13+r23+r33)
    vec2=normalize(vec2)
    #step3
    up=poses[:,:3,1].sum(0)#(r12+r22+r32)
    # print("center:",center)
    # print("vec2:",vec2)
    # print("up:",up)
    #strp4 and step 5 in above function
    c2w=np.concatenate([view_matrix(vec2,up,center),hwf],1)
    # print(c2w.shape)
    return (c2w)



def spiral_path(c2w,up,rads,focal,zrate,rots,N):
    render_poses=[]
    rads=np.array(list(rads)+[1.0])
    # print("Rads:",rads)
    for theta in np.linspace(0.0,2.0*rots*np.pi,N+1)[:-1]:
        # print("DEFAULT THETA:",theta)
        #dot product between camera matrix and spiral poses (new camera centers for the new poses)
        # don't understand the need for the dot product maybe a projection of the optical center on the new spiral pose.
        center=np.dot(c2w[:3,:4],np.array([np.cos(theta),-np.sin(theta),-np.sin(theta*zrate),1.0])*rads)        
         # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z=normalize(center-np.dot(c2w[:3,:4],np.array([0,0,-focal,1.0])))
        # print("z:",z)
        render_poses.append(np.concatenate([view_matrix(z,up,center),c2w[:,4:5]],1))
    return (render_poses)



def load_llff(basedir,factor=8,recenter=True,bd_factor=0.75,spherify=False):
    poses,bounds,images=_load_data(basedir,factor=8)
    #(3,5,20);(2,20),(h,w,20);the 3,4 pose matrix is actually [R_{c}|-C]

    # change from [down right back] to [right up back]
    poses=np.concatenate([poses[:,1:2,:],-poses[:,0:1,:],poses[:,2:,:]],1)
    
    #change image dimension to 0th position
    poses=np.moveaxis(poses,-1,0).astype(np.float32)
    images=np.moveaxis(images,-1,0).astype(np.float32)
    bounds=np.moveaxis(bounds,-1,0).astype(np.float32)

    # print(bounds) # min and max bounds of each image
    
    # Rescale if bd_factor is provided
    """We want to sample the points from -n (n=1 after scaling) to infinity, to do that we first "move o to the ray’s intersection with 
    the near plane at z = -n" and "simply sample t0 linearly from 0 to 1". So the near=0 and far=1 actually correspond to the 0 and 1 in this 
    last quote!"""
    sc=1.0 if bd_factor is None else 1.0/(bounds.min()*bd_factor)
    poses[:,:3,3]*=sc#scaling optical center
    bounds*=sc # scaling bounds
    # print(poses.shape)
    # print(bounds.shape)
    # print(images.shape)

    
    # recenter poses   
    if recenter:
        poses=recenter_pose(poses)#after this function all the poses are centerd
            #take pose of all recenetered pose
    if spherify:
        print("this is a foward facing dataset")
    else:
        c2w=avg_poses(poses)
        # print("recentered and averaged pose:",c2w.shape)
        # print("recentered and averge pose of the dataset (right,up,back):",c2w[:3,:4])# the matrix is (right up back)

        # print(c2w[:,-1])
        up=normalize(poses[:,:3,1].sum(0))
        #zfar and zmin ()

        # hardcoded
        close_depth=bounds.min()*0.9
        far_depth=bounds.max()*5.0

        #check this part again
        # Find a reasonable "focus depth" for this dataset
        dt=0.75
        mean_dz=1.0/(((1.0-dt)/close_depth+dt/far_depth))
        focal=mean_dz
    
        #rads is radius of the spiral path along the x,y,z so 3 dimensional radius
        rads=np.percentile(np.abs(poses[:,:3,3]),90,axis=0)#90th percetnile point in abs(tt)
        # print(np.abs(poses[:,:3,3]))
        # print(rads)
    
        c2w_path=c2w
        N_views=120
        N_rots=2

    #c2w_path is avg_pose of recentered poses, up is mean y axis, (focal length is mean_dz,zdelta=0.2*close_depth dist between)
    #(rads is radius so mean of translation vectro)
    #zrate is the at which it rotates about z-axis check custom function ,N_rots is number of rotations about z axis ; N_views in the number of samples
        spiral_poses=spiral_path(c2w_path,up,rads,focal,zrate=0.5,rots=N_rots,N=N_views)
    spiral_poses=np.array(spiral_poses).astype(np.float32)
    c2w=avg_poses(poses)
    # print("Data:")
    # print("poses:",poses.shape)
    # print("images:",images.shape)
    # print("bounds:",bounds.shape)

    #distance between the recentered poses and averaged pose
    dists=np.sum(np.square(c2w[:3,3]-poses[:,:3,3]),-1)
    test_idx=np.argmin(dists)
    # print("holdout view is:",test_idx)
    images=images.astype(np.float32)
    poses=poses.astype(np.float32)
    print("MY FUNCTION")
    bounding_box=get_bbox(poses[:,:3,:4],poses[0,:3,-1],near=0.0,far=1.0)
    return (images,poses,bounds,spiral_poses,test_idx,bounding_box)

# img,img_pose,bds,spiral_pose,val_idx=load_llff(basedir="/vinai/sskar/NERF/nerf_llff_data/fern")
