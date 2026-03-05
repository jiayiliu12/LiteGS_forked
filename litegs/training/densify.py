import torch
import math

from ..arguments import DensifyParams
from ..utils.statistic_helper import StatisticsHelperInst
from ..utils import qvec2rotmat
from ..scene import cluster
from ..utils import wrapper

import sys
from .. import render
import tqdm

class DensityControllerBase:
    def __init__(self,densify_params:DensifyParams,bCluster:bool) -> None:
        self.densify_params=densify_params
        self.bCluster=bCluster
        return
    
    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int,scores):
        return self._get_params_from_optimizer(optimizer)
    
    @torch.no_grad()
    def _get_params_from_optimizer(self,optimizer:torch.optim.Optimizer)->list[torch.Tensor]:
        param_dict:dict[str,torch.Tensor]={}
        for param_group in optimizer.param_groups:
            name=param_group['name']
            tensor=param_group['params'][0]
            param_dict[name]=tensor
        xyz=param_dict["xyz"]
        rot=param_dict["rot"]
        scale=param_dict["scale"]
        sh_0=param_dict["sh_0"]
        sh_rest=param_dict["sh_rest"]
        opacity=param_dict["opacity"]
        return xyz,scale,rot,sh_0,sh_rest,opacity

    @torch.no_grad()
    def _cat_tensors_to_optimizer(self, tensors_dict:dict,optimizer:torch.optim.Optimizer):
        cat_dim=-1
        if self.bCluster:
            cat_dim=-2
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and stored_state["exp_avg"].shape==group["params"][0].shape
            if stored_state is not None:
                stored_state["exp_avg"].data=torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=cat_dim).contiguous()
                stored_state["exp_avg_sq"].data=torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=cat_dim).contiguous()
            new_param=torch.cat((group["params"][0], extension_tensor), dim=cat_dim).contiguous()
            optimizer.state.pop(group['params'][0])#pop param
            group["params"][0]=torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]]=stored_state#assign to new param
            assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and stored_state["exp_avg"].shape==group["params"][0].shape
        return
    
    @torch.no_grad()
    def _replace_tensor_to_optimizer(self, tensor:torch.Tensor, name:str,optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                #stored_state["step"]=0#bugfix

                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
        return
    
    @torch.no_grad()
    def _prune_optimizer(self,valid_mask:torch.Tensor,optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if self.bCluster:
                    chunk_size=stored_state["exp_avg"].shape[-1]
                    uncluster_avg,uncluster_avg_sq=cluster.uncluster(stored_state["exp_avg"],stored_state["exp_avg_sq"])
                    uncluster_avg=uncluster_avg[...,valid_mask]
                    uncluster_avg_sq=uncluster_avg_sq[...,valid_mask]
                    new_avg,new_avg_sq=cluster.cluster_points(chunk_size,uncluster_avg,uncluster_avg_sq)
                else:
                    new_avg=stored_state["exp_avg"][...,valid_mask]
                    new_avg_sq=stored_state["exp_avg_sq"][...,valid_mask]
                stored_state["exp_avg"].data=new_avg
                stored_state["exp_avg_sq"].data=new_avg_sq
            
            if self.bCluster:
                chunk_size=group["params"][0].shape[-1]
                uncluster_param,=cluster.uncluster(group["params"][0])
                uncluster_param=uncluster_param[...,valid_mask]
                new_param,=cluster.cluster_points(chunk_size,uncluster_param)
            else:
                new_param=group["params"][0][...,valid_mask]
            optimizer.state.pop(group['params'][0])#pop param
            group["params"][0]=torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]]=stored_state#assign to new param
        return
    
class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self,screen_extent:float,densify_params:DensifyParams,bCluster:bool,init_points_num:int)->None:
        self.grad_threshold=densify_params.densify_grad_threshold
        self.min_opacity=densify_params.opacity_threshold
        self.percent_dense=densify_params.percent_dense
        self.screen_extent=screen_extent
        self.max_screen_size=densify_params.screen_size_threshold
        self.init_points_num=init_points_num
        super(DensityControllerOfficial,self).__init__(densify_params,bCluster)
        return
    
    @torch.no_grad()
    def get_prune_mask(self,actived_opacity:torch.Tensor,actived_scale:torch.Tensor)->torch.Tensor:
        transparent = (actived_opacity < self.min_opacity).squeeze()
        invisible = StatisticsHelperInst.get_global_culling()
        invisible.shape[0]
        prune_mask=transparent
        prune_mask[:invisible.shape[0]]|=invisible
        return prune_mask

    @torch.no_grad()
    def get_clone_mask(self,actived_scale:torch.Tensor)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad').squeeze()
        abnormal_mask = mean2d_grads >= self.grad_threshold
        tiny_pts_mask = actived_scale.max(dim=0).values <= self.percent_dense*self.screen_extent
        selected_pts_mask = abnormal_mask&tiny_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def get_split_mask(self,actived_scale:torch.Tensor,N=2)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad').squeeze()
        abnormal_mask = mean2d_grads >= self.grad_threshold
        large_pts_mask = actived_scale.max(dim=0).values > self.percent_dense*self.screen_extent
        selected_pts_mask=abnormal_mask&large_pts_mask
        return selected_pts_mask
    
    # Grad is enabled
    def score_func_speedysplat(self,view_matrix,proj_matrix,frustumplane,gt_image,scores,optimizer:torch.optim.Optimizer,actived_sh_degree,op,pp):
        
        # Set these to None so that they are initialized in the render_preprocess() function
        cluster_origin=None
        cluster_extend=None

        # Get clustered params to calculate the score
        xyz,scale,rot,sh_0,sh_rest,opacity = self._get_params_from_optimizer(optimizer)

        visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,actived_sh_degree)
        
        # Chunk-level ids -> per-point ids
        ar = torch.arange(pp.cluster_size, device=visible_chunkid.device, dtype=torch.long)
        culled_idx = (visible_chunkid.to(torch.long).unsqueeze(-1) * pp.cluster_size + ar).reshape(-1).detach()

        img_scores = torch.zeros((1, culled_opacity.numel()), # Shape [B,N_culled] with B=1, according to what CUDA expects
                                device=culled_opacity.device,
                                dtype=culled_opacity.dtype,
                                requires_grad=True).contiguous()

        # print("view_matrix", view_matrix.shape) -> view_matrix torch.Size([1, 4, 4])
        # print("proj_matrix", proj_matrix.shape) -> proj_matrix torch.Size([1, 4, 4])
        # print("frustumplane", frustumplane.shape) -> frustumplane torch.Size([1, 6, 4])

        img,_,_,_,_,elapsed_times=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                    actived_sh_degree,gt_image.shape[2:],pp,scores=img_scores)

        img.sum().backward()
        scores.index_add_(0,culled_idx,img_scores.grad.detach().reshape(-1)) #.to("cpu", non_blocking=True)) # img_scores.grad: shape [1,N_culled]; scores: shape [N]
        del img, img_scores
        del culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity, culled_idx
        return
    
    # def get_prune_mask_speedysplat(self,optimizer:torch.optim.Optimizer,percent,opacity,train_loader,actived_sh_degree,op,pp):
    #     # All model params are unclustered here
    #     with torch.enable_grad():
    #         scores = torch.zeros(opacity.numel(), device="cuda", dtype=opacity.dtype) #, pin_memory=True) # one score for each Gaussian in the model! -> shape [N]
    #         for i,(view_matrix,proj_matrix,frustumplane,gt_image,idx) in enumerate(train_loader):
    #             optimizer.zero_grad(set_to_none=True) # Reset all parameter gradients to save space in VRAM
    #             # Get scores param
    #             self.score_func_speedysplat(view_matrix,proj_matrix,frustumplane,gt_image,scores,optimizer,actived_sh_degree,op,pp)

    #     sorted_tensor, _ = torch.sort(scores)
    #     index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    #     value_nth_percentile = sorted_tensor[index_nth_percentile]
    #     # scores[scores==0] = 1 # TODO: Don't prune the Gaussians that were not scored???
    #     prune_mask = ((scores <= value_nth_percentile))  # [N] bool on GPU
    #     del scores, sorted_tensor
    #     return prune_mask # Must have shape [N]

    def get_prune_mask_speedysplat_new(self,percent,scores):
        # All model params are unclustered here
        sorted_tensor, _ = torch.sort(scores)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        # scores[scores==0] = 1 # TODO: Don't prune the Gaussians that were not scored???
        prune_mask = ((scores <= value_nth_percentile))  # [N] bool on GPU
        del scores, sorted_tensor
        return prune_mask # Must have shape [N]

    # @torch.no_grad()
    # def prune_speedysplat(self,optimizer:torch.optim.Optimizer,prune_ratio,train_loader,actived_sh_degree,op,pp):

    #     xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
    #     chunk_size = 1
    #     if self.bCluster:
    #         chunk_size=xyz.shape[-1]
    #         xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
    #         # xyz torch.Size([3, N])
    #         # sh_0 torch.Size([1, 3, N])
    #         # sh_rest torch.Size([15, 3, N])
    #         # opacity torch.Size([1, N])
    #         # scale torch.Size([3, N])
    #         # rot torch.Size([4, N])

    #     # prune_mask: shape [N]
    #     prune_mask=self.get_prune_mask_speedysplat(optimizer,prune_ratio,opacity,train_loader,actived_sh_degree,op,pp) 

    #     if prune_mask.sum() > 0.9 * opacity.shape[1]:
    #         raise RuntimeError("Pruning would remove >90% of Gaussians")
    #     if self.bCluster:
    #         N=prune_mask.sum()
    #         chunk_num=int(N/chunk_size)
    #         del_limit=chunk_num*chunk_size
    #         del_indices=prune_mask.nonzero()[:del_limit,0]
    #         prune_mask=torch.zeros_like(prune_mask)
    #         prune_mask[del_indices]=True
    #     self._prune_optimizer(~prune_mask,optimizer)
    #     del prune_mask
    #     return
    

    @torch.no_grad()
    def prune_speedysplat_new(self,optimizer:torch.optim.Optimizer,prune_ratio,scores):

        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        chunk_size=1
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            # xyz torch.Size([3, N])
            # sh_0 torch.Size([1, 3, N])
            # sh_rest torch.Size([15, 3, N])
            # opacity torch.Size([1, N])
            # scale torch.Size([3, N])
            # rot torch.Size([4, N])
        
        # Append ones to scores to match the new number of Gaussians after densification!
        # padding_size=max(xyz.shape[-1]-scores.shape[0],scores.shape[0]-xyz.shape[-1])
        # scores_padded=torch.nn.functional.pad(scores,(0,padding_size),value=1)
        N = xyz.shape[-1]  # unclustered N
        if scores.shape[0] < N:
            scores_adj = torch.cat([scores, scores.new_full((N - scores.shape[0],), float("inf"))])
        else:
            raise RuntimeError("scores should not have more elements than the number of Gaussians in the model!")

        # prune_mask: shape [N]
        prune_mask=self.get_prune_mask_speedysplat_new(prune_ratio,scores_adj)

        if prune_mask.sum() > 0.9 * opacity.shape[1]:
            raise RuntimeError("Pruning would remove >90% of Gaussians")
        if self.bCluster:
            N=prune_mask.sum()
            chunk_num=int(N/chunk_size)
            del_limit=chunk_num*chunk_size
            del_indices=prune_mask.nonzero()[:del_limit,0]
            prune_mask=torch.zeros_like(prune_mask)
            prune_mask[del_indices]=True
        self._prune_optimizer(~prune_mask,optimizer)
        del prune_mask
        return
    
    # for g in optimizer.param_groups:
    #     p = g["params"][0]
    #     print(g["name"], p.shape)
    # sys.exit()

    # xyz torch.Size([3, 665, 128])
    # sh_0 torch.Size([1, 3, 665, 128])
    # sh_rest torch.Size([15, 3, 665, 128])
    # opacity torch.Size([1, 665, 128])
    # scale torch.Size([3, 665, 128])
    # rot torch.Size([4, 665, 128])

    @torch.no_grad()
    def prune(self,optimizer:torch.optim.Optimizer):
        
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        prune_mask=self.get_prune_mask(opacity.sigmoid(),scale.exp())
        if prune_mask.sum()>0.8*opacity.shape[1]:
            assert(False) #debug
        if self.bCluster:
            N=prune_mask.sum()
            chunk_num=int(N/chunk_size)
            del_limit=chunk_num*chunk_size
            del_indices=prune_mask.nonzero()[:del_limit,0]
            prune_mask=torch.zeros_like(prune_mask)
            prune_mask[del_indices]=True
        #print("\n #prune:{0} #points:{1}".format(prune_mask.sum(),(~prune_mask).sum()))
        self._prune_optimizer(~prune_mask,optimizer)
        return

    @torch.no_grad()
    def split_and_clone(self,optimizer:torch.optim.Optimizer,epoch:int):
        
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        clone_mask=self.get_clone_mask(scale.exp())
        split_mask=self.get_split_mask(scale.exp())

        #split
        stds=scale[...,split_mask].exp()
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(torch.ones_like(scale[...,split_mask].exp()),torch.nn.functional.normalize(rot[...,split_mask],dim=0))
        transform_matrix=transform_matrix[:3,:3]
        shift=(samples.permute(2,0,1))@transform_matrix.permute(2,0,1)
        shift=shift.permute(1,2,0).squeeze(0)
        
        split_xyz=xyz[...,split_mask]+shift
        clone_xyz=xyz[...,clone_mask]
        append_xyz=torch.cat((split_xyz,clone_xyz),dim=-1)
        
        split_scale = (scale[...,split_mask].exp() / (0.8*2)).log()
        clone_scale = scale[...,clone_mask]
        append_scale = torch.cat((split_scale,clone_scale),dim=-1)

        split_rot=rot[...,split_mask]
        clone_rot=rot[...,clone_mask]
        append_rot = torch.cat((split_rot,clone_rot),dim=-1)

        split_sh_0=sh_0[...,split_mask]
        clone_sh_0=sh_0[...,clone_mask]
        append_sh_0 = torch.cat((split_sh_0,clone_sh_0),dim=-1)

        split_sh_rest=sh_rest[...,split_mask]
        clone_sh_rest=sh_rest[...,clone_mask]
        append_sh_rest = torch.cat((split_sh_rest,clone_sh_rest),dim=-1)

        split_opacity=opacity[...,split_mask]
        clone_opacity=opacity[...,clone_mask]
        append_opacity = torch.cat((split_opacity,clone_opacity),dim=-1)

        if self.bCluster:
            N=append_xyz.shape[-1]
            chunk_num=int(N/chunk_size)
            append_limit=chunk_num*chunk_size
            append_xyz,append_scale,append_rot,append_sh_0,append_sh_rest,append_opacity=cluster.cluster_points(
                chunk_size,append_xyz[...,:append_limit],append_scale[...,:append_limit],
                append_rot[...,:append_limit],append_sh_0[...,:append_limit],
                append_sh_rest[...,:append_limit],append_opacity[...,:append_limit])

        dict_clone = {"xyz": append_xyz,
                      "scale": append_scale,
                      "rot" : append_rot,
                      "sh_0": append_sh_0,
                      "sh_rest": append_sh_rest,
                      "opacity" : append_opacity}
        
        #print("\n#clone:{0} #split:{1} #points:{2}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),xyz.shape[-1]+append_xyz.shape[-1]*append_xyz.shape[-2]))
        self._cat_tensors_to_optimizer(dict_clone,optimizer)
        return
    
    @torch.no_grad()
    def reset_opacity(self,optimizer:torch.optim.Optimizer,epoch:int):
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        actived_opacities=opacity.sigmoid()
        if self.densify_params.opacity_reset_mode=='decay':
            decay_rate=0.5
            opacity.data=inverse_sigmoid((actived_opacities*decay_rate).clamp_min(1.0/128))
            optimizer.state.clear()
        elif self.densify_params.opacity_reset_mode=='reset':
            opacity.data=inverse_sigmoid(actived_opacities.clamp_max(0.005))
            self._replace_tensor_to_optimizer(opacity,"opacity",optimizer)

        return
    
    @torch.no_grad()
    def is_densify_actived(self,epoch:int):

        return epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from and (
            epoch%self.densify_params.densification_interval==0)

    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int,scores):
        if epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from:
            bUpdate=False
            if epoch%self.densify_params.densification_interval==0:
                self.split_and_clone(optimizer,epoch)
                # self.prune(optimizer) // TODO: merge LiteGS prune with Speedy prune!!

                # Speedy-Splat soft pruning during densification
                if  (epoch >= self.densify_params.soft_prune_from_epoch) and \
                    (epoch < self.densify_params.hard_prune_from_epoch) and \
                    (epoch % self.densify_params.soft_prune_epoch_interval == 0):
                    print(f"Soft pruning at epoch {epoch} ####")
                    # print(f"self.densify_params.densify_until {self.densify_params.densify_until} ####")
                    self.prune_speedysplat_new(optimizer,self.densify_params.soft_prune_ratio,scores)

                bUpdate=True
            if epoch%self.densify_params.opacity_reset_interval==0:
                self.reset_opacity(optimizer,epoch)
                bUpdate=True
            if bUpdate:
                xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
                StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],self.is_densify_actived)
                torch.cuda.empty_cache()

        # Speedy-Splat hard pruning after densification
        if  (epoch >= self.densify_params.hard_prune_from_epoch) and \
            (epoch % self.densify_params.hard_prune_epoch_interval == 0):

            print(f"Hard pruning at epoch {epoch} ####")
            self.prune_speedysplat_new(optimizer,self.densify_params.hard_prune_ratio,scores)

            xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
            StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],self.is_densify_actived)
            torch.cuda.empty_cache()
            

        return self._get_params_from_optimizer(optimizer)
    

class DensityControllerTamingGS(DensityControllerOfficial):
    @torch.no_grad()
    def __init__(self,screen_extent:int,densify_params:DensifyParams,bCluster:bool,init_points_num:int)->None:

        assert(densify_params.target_primitives!=0.0)
        self.target_points_num=densify_params.target_primitives
        super(DensityControllerTamingGS,self).__init__(screen_extent,densify_params,bCluster,init_points_num)
        return
    
    @torch.no_grad()
    def get_prune_mask(self,actived_opacity:torch.Tensor,actived_scale:torch.Tensor)->torch.Tensor:
        if self.densify_params.prune_mode == 'weight': # LiteGS uses 'weight' as default
            prune_mask=torch.zeros(actived_opacity.shape[1],device=actived_opacity.device).bool()

            frag_weight,frag_count=StatisticsHelperInst.get_mean('fragment_weight')
            weight_sum=(frag_weight*frag_count).nan_to_num(0).squeeze()
            invisible = weight_sum==0 # TODO: Change this back?? weight_sum<(weight_sum[weight_sum!=0].quantile(0.05)) 
            prune_mask[:invisible.shape[0]]|=invisible
        elif self.densify_params.prune_mode == 'threshold':
            prune_mask=super(DensityControllerTamingGS,self).get_prune_mask(actived_opacity,actived_scale)
        
        return prune_mask
    
    def get_score(self,xyz,scale,rot,sh_0,sh_rest,opacity)->torch.Tensor:
        var,frag_count=StatisticsHelperInst.get_var('fragment_err')
        #score=(var*frag_count).sqrt()*(opacity.sigmoid())
        score=var*frag_count*(opacity.sigmoid()*opacity.sigmoid())
        score=score.squeeze().nan_to_num(0)
        score.clamp_min_(0)
        return score
    
    @torch.no_grad()
    def split_and_clone(self,optimizer:torch.optim.Optimizer,epoch:int):
        
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        prune_num=self.get_prune_mask(opacity.sigmoid(),scale.exp()).sum()

        cur_target_count = (self.target_points_num - self.init_points_num) / (self.densify_params.densify_until - self.densify_params.densify_from) * (epoch-self.densify_params.densify_from)+self.init_points_num
        budget=min(max(int(cur_target_count-xyz.shape[-1]),1)+prune_num,xyz.shape[-1])

        score=self.get_score(xyz,scale,rot,sh_0,sh_rest,opacity)
        densify_index = torch.multinomial(score, budget, replacement=False)
        clone_index=densify_index[(scale[:,densify_index].exp().max(dim=0).values <= self.percent_dense*self.screen_extent)]
        split_index=densify_index[(scale[:,densify_index].exp().max(dim=0).values > self.percent_dense*self.screen_extent)]

        #split
        stds=scale[...,split_index].exp()
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(torch.ones_like(scale[...,split_index]),torch.nn.functional.normalize(rot[...,split_index],dim=0))
        transform_matrix=transform_matrix[:3,:3]
        shift=(samples.permute(2,0,1))@transform_matrix.permute(2,0,1)
        shift=shift.permute(1,2,0).squeeze(0)
        
        split_xyz=xyz[...,split_index]+shift
        clone_xyz=xyz[...,clone_index]
        append_xyz=torch.cat((split_xyz,clone_xyz),dim=-1)
        
        split_scale = (scale[...,split_index].exp() / (0.8*2)).log()
        clone_scale = scale[...,clone_index]
        append_scale = torch.cat((split_scale,clone_scale),dim=-1)

        split_rot=rot[...,split_index]
        clone_rot=rot[...,clone_index]
        append_rot = torch.cat((split_rot,clone_rot),dim=-1)

        split_sh_0=sh_0[...,split_index]
        clone_sh_0=sh_0[...,clone_index]
        append_sh_0 = torch.cat((split_sh_0,clone_sh_0),dim=-1)

        split_sh_rest=sh_rest[...,split_index]
        clone_sh_rest=sh_rest[...,clone_index]
        append_sh_rest = torch.cat((split_sh_rest,clone_sh_rest),dim=-1)

        split_opacity=opacity[...,split_index]
        clone_opacity=opacity[...,clone_index]
        append_opacity = torch.cat((split_opacity,clone_opacity),dim=-1)

        if self.bCluster:
            N=append_xyz.shape[-1]
            chunk_num=int(N/chunk_size)
            append_limit=chunk_num*chunk_size
            append_xyz,append_scale,append_rot,append_sh_0,append_sh_rest,append_opacity=cluster.cluster_points(
                chunk_size,append_xyz[...,:append_limit],append_scale[...,:append_limit],
                append_rot[...,:append_limit],append_sh_0[...,:append_limit],
                append_sh_rest[...,:append_limit],append_opacity[...,:append_limit])

        dict_clone = {"xyz": append_xyz,
                      "scale": append_scale,
                      "rot" : append_rot,
                      "sh_0": append_sh_0,
                      "sh_rest": append_sh_rest,
                      "opacity" : append_opacity}
        
        #print("\n#clone:{0} #split:{1} #points:{2}".format(clone_index.sum().cpu(),split_index.sum().cpu(),xyz.shape[-1]+append_xyz.shape[-1]*append_xyz.shape[-2]))
        self._cat_tensors_to_optimizer(dict_clone,optimizer)
        return
    