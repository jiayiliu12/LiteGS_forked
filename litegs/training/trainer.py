import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import numpy as np
import math
import os
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
import json
import wandb


from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify
from .. import utils


def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):
    
    VERBOSE = dp.verbosity
    
    if VERBOSE:
        densification_pruning_start = torch.cuda.Event(enable_timing=True)
        densification_pruning_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        preprocess_start = torch.cuda.Event(enable_timing=True)
        preprocess_end = torch.cuda.Event(enable_timing=True)
        total_iteration_start = torch.cuda.Event(enable_timing=True)
        total_iteration_end = torch.cuda.Event(enable_timing=True)
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.ImageFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        if os.path.exists(os.path.join(lp.source_path,"train_test_split.json")):
            with open(os.path.join(lp.source_path,"train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames=[c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames=[c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            # test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
            test_frames = [camera_frames[idx] for idx in range(0, 40, 8)] # ensures 5 testing imgs, which are idx % 8
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    init_points_num=init_xyz.shape[0]
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
    if pp.cluster_size:
        cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #learnable view matrix
    if op.learnable_viewproj:
        noise_extr=torch.cat([frame.extr_params[None,:] for frame in trainingset.frames])
        denoised_training_extr=torch.nn.Embedding(noise_extr.shape[0],noise_extr.shape[1],_weight=noise_extr.clone(),sparse=True)
        noise_intr=torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0)
        denoised_training_intr=torch.nn.Parameter(torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0))#todo fix multi cameras
        view_opt=torch.optim.SparseAdam(denoised_training_extr.parameters(),lr=1e-4)
        proj_opt=torch.optim.Adam([denoised_training_intr,],lr=1e-5)

    #init
    total_epoch=int(op.iterations/len(trainingset))
    if dp.densify_until<0:
        dp.densify_until=int(total_epoch*0.8/dp.opacity_reset_interval)*dp.opacity_reset_interval+1
    density_controller=densify.DensityControllerTamingGS(norm_radius,dp,pp.cluster_size>0,init_points_num)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    #variables for wandb
    with torch.no_grad():
        iteration = 0
        sum_time=0
        sum_time_with_prune=0

    for epoch in range(start_epoch,total_epoch):

        with torch.no_grad():
            if pp.cluster_size>0 and (epoch-1)%dp.densification_interval==0:
                xyz,scale,rot,sh_0,sh_rest,opacity=scene.spatial_refine(pp.cluster_size>0,opt,xyz)
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

            total_iteration_with_pruning_time=0
            total_iteration_time=0
            densification_pruning_time=0

            prune_bool = (
                (epoch >= dp.soft_prune_from_epoch and epoch < dp.hard_prune_from_epoch and epoch % dp.soft_prune_epoch_interval == 0) or
                (epoch >= dp.hard_prune_from_epoch and epoch % dp.hard_prune_epoch_interval == 0)
            )
            scores = torch.zeros(opacity.numel(), device=opacity.device, dtype=opacity.dtype) if prune_bool else None
            if prune_bool:
                print(f"Start soft/hard pruning at epoch {epoch} ####")

        torch.cuda.synchronize()
        with StatisticsHelperInst.try_start(epoch):
            for i,(view_matrix,proj_matrix,frustumplane,gt_image,idx) in enumerate(train_loader):
                if VERBOSE:
                    total_iteration_start.record()

                nvtx.range_push("Iter Init")
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                idx=idx.cuda()
                if op.learnable_viewproj:
                    #fix view matrix
                    extr=denoised_training_extr(idx)
                    intr=denoised_training_intr
                    view_matrix,proj_matrix,viewproj_matrix,frustumplane=utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)
                nvtx.range_pop()

                #cluster culling
                if VERBOSE:
                    preprocess_start.record()
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,actived_sh_degree)
                if VERBOSE:
                    preprocess_end.record()

                img_scores = (
                    torch.zeros((1, culled_opacity.numel()), device=culled_opacity.device,
                                dtype=culled_opacity.dtype, requires_grad=True).contiguous()
                    if prune_bool else None
                )
                img,transmitance,depth,normal,primitive_visible,elapsed_times=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                        actived_sh_degree,gt_image.shape[2:],pp,scores=img_scores)
                                                            
                l1_loss=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight
                if pp.enable_transmitance:
                    loss+=(1-transmitance).abs().mean()

                if VERBOSE:
                    backward_start.record()
                loss.backward()
                if VERBOSE:
                    backward_end.record()

                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.sparse_grad:
                    opt.step(visible_chunkid,primitive_visible)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
                if op.learnable_viewproj:
                    view_opt.step()
                    view_opt.zero_grad()
                    # proj_opt.step()
                    # proj_opt.zero_grad()
                schedular.step()

                if VERBOSE:
                    total_iteration_end.record()
                    total_iteration_end.synchronize()

                    total_iteration_time = total_iteration_start.elapsed_time(total_iteration_end)
                    sum_time+=total_iteration_time

                    wandb.log({
                        "train/total_loss": loss.item(),
                        "train/L1": l1_loss.item(),
                        "gaussians/count": xyz.shape[1] * xyz.shape[2],
                        "time/render_preprocess(cluster culling) [ms]": preprocess_start.elapsed_time(preprocess_end),
                        "time/backward [ms]": backward_start.elapsed_time(backward_end),
                        "time/total_iteration [ms]": total_iteration_time,
                        "time/sum_time [ms]": sum_time,
                        "time/render [ms]": elapsed_times["render_time"],
                        "time/render/CreateTransformMatrix [ms]": elapsed_times["CreateTransformMatrix_time"],
                        "time/render/CreateRaySpaceTransformMatrix [ms]": elapsed_times["CreateRaySpaceTransformMatrix_time"],
                        "time/render/CreateCov2dDirectly [ms]": elapsed_times["CreateCov2dDirectly_time"],
                        "time/render/EighAndInverse2x2Matrix [ms]": elapsed_times["EighAndInverse2x2Matrix_time"],
                        "time/render/Binning [ms]": elapsed_times["Binning_time"],
                        "time/render/rasterize_forward [ms]": elapsed_times["GaussiansRasterFunc_time"],
                    }, iteration)
                
                if prune_bool:
                    density_controller.accumulate_scores(scores, visible_chunkid, img_scores, pp.cluster_size)
                    del img_scores
                
                iteration+=1


        if VERBOSE and (epoch in test_epochs or epoch==total_epoch-1):
            with torch.no_grad():
                _cluster_origin=None
                _cluster_extend=None
                if pp.cluster_size:
                    _cluster_origin,_cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    l1_loss_test_list=[]
                    psnr_list=[]
                    ssim_list=[]
                    logged_images = []
                    num_log_images = 6
                    # Pick 6 evenly-spaced frame indices across the loader
                    log_indices = set(np.linspace(0, len(loader) - 1, num_log_images, dtype=int).tolist())
                    for batch_i,(view_matrix,proj_matrix,frustumplane,gt_image,idx) in enumerate(loader):
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0
                        idx=idx.cuda()
                        if op.learnable_viewproj:
                            if name=="Trainingset":
                                #fix view matrix
                                extr=denoised_training_extr(idx)
                                intr=denoised_training_intr
                            else:
                                nearest_idx=(extr-denoised_training_extr._parameters['weight']).abs().sum(dim=1).argmin()
                                delta=denoised_training_extr(nearest_idx)-noise_extr[nearest_idx]
                                extr=extr+delta
                            view_matrix,proj_matrix,viewproj_matrix,frustumplane=utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)

                        #cluster culling
                        visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,actived_sh_degree)
                        img,transmitance,depth,normal,primitive_visible,elapsed_times=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
                        l1_loss_test_list.append(__l1_loss(img,gt_image).unsqueeze(0))
                        psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
                        ssim_list.append(fused_ssim.fused_ssim(img,gt_image).unsqueeze(0))

                        # --- Wandb image logging ---
                        # img and gt_image are [1, 3, H, W] tensors in [0, 1].
                        # We build a side-by-side panel: rendered (left) | GT (right).
                        if name == "Testset" and batch_i in log_indices:
                            rendered_np = img[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()   # [H, W, 3]
                            gt_np       = gt_image[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                            panel       = np.concatenate([rendered_np, gt_np], axis=1)          # [H, 2W, 3]
                            panel_uint8 = (panel * 255).astype(np.uint8)
                            logged_images.append(
                                wandb.Image(
                                    panel_uint8,
                                    caption=f"epoch {epoch} | {name} | frame {batch_i} | left: render  right: GT"
                                )
                            )

                    l1_loss_test_mean=torch.concat(l1_loss_test_list,dim=0).mean()
                    psnr_mean=torch.concat(psnr_list,dim=0).mean()
                    ssim_mean=torch.concat(ssim_list,dim=0).mean()

                    wandb.log({
                        f"test/l1_loss_{name}" : l1_loss_test_mean.item(),
                        f"test/psnr_{name}" : psnr_mean.item(),
                        f"test/ssim_{name}" : ssim_mean.item(),
                    }, iteration)
                    if name=="Testset":
                        wandb.log({
                            f"test/renders_{name}" : logged_images,   # <-- 6 side-by-side images
                        }, iteration)
                        
                    tqdm.write("\n[EPOCH {}] {} Evaluating: PSNR {} with xyz.shape {}".format(epoch,name,psnr_mean, str(xyz.shape)))

        if VERBOSE:
            densification_pruning_start.record()
        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch,iteration,scores,len(trainingset))
        if VERBOSE:
            densification_pruning_end.record()

        progress_bar.update()  

        if VERBOSE:
            densification_pruning_end.synchronize()

            densification_pruning_time=densification_pruning_start.elapsed_time(densification_pruning_end)
            total_iteration_with_pruning_time=total_iteration_time+densification_pruning_time
            sum_time_with_prune+=total_iteration_with_pruning_time

            wandb.log({
                "time/densification_pruning [ms]": densification_pruning_time,
                "time/total_iteration_with_pruning [ms]": total_iteration_with_pruning_time,
                "time/sum_time_with_prune [ms]": sum_time_with_prune,
            }, iteration)

        if epoch in save_ply or epoch==total_epoch-1:
            if epoch==total_epoch-1:
                progress_bar.close()
                print("{} takes: {} seconds.".format(lp.model_path,progress_bar.format_dict['elapsed']))
                save_path=os.path.join(lp.model_path,"point_cloud","finish")
            else:
                save_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch))    

            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path,"point_cloud.ply"),*param_nyp)
            if op.learnable_viewproj:
                torch.save(list(denoised_training_extr.parameters())+[denoised_training_intr],os.path.join(save_path,"viewproj.pth"))

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    return