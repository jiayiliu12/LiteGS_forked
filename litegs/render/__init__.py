import torch
import math
import typing
import torch.cuda.nvtx as nvtx

from .. import utils
from ..utils.statistic_helper import StatisticsHelperInst,StatisticsHelper
from .. import arguments
from .. import scene

from litegs.render.rasterizer_pytorch import rasterize_pytorch_view

def render_preprocess(cluster_origin:torch.Tensor|None,cluster_extend:torch.Tensor|None,frustumplane:torch.Tensor,view_matrix:torch.Tensor,
                      xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,sh_0:torch.Tensor,sh_rest:torch.Tensor,opacity:torch.Tensor,
                      op:arguments.OptimizationParams,pp:arguments.PipelineParams,actived_sh_degree:int):

    if pp.cluster_size:
        if cluster_origin is None or cluster_extend is None:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))

        if pp.sparse_grad:#enable sparse gradient
            visible_chunkid,culled_xyz,culled_scale,culled_rot,color,culled_opacity=utils.wrapper.CullCompactActivateWithSparseGrad.apply(
                cluster_origin,cluster_extend,frustumplane,view_matrix,actived_sh_degree,xyz,scale,rot,sh_0,sh_rest,opacity)
            culled_xyz,culled_scale,culled_rot,color,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,color,culled_opacity)  
            if StatisticsHelperInst.bStart:
                StatisticsHelperInst.set_compact_mask(visible_chunkid)
            return visible_chunkid,culled_xyz,culled_scale,culled_rot,color,culled_opacity
        else:
            visibility,visible_num,visible_chunkid=utils.wrapper.litegs_fused.frustum_culling_aabb_cuda(cluster_origin,cluster_extend,frustumplane)
            visible_chunkid=visible_chunkid[:visible_num]
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.culling(visible_chunkid,xyz,scale,rot,sh_0,sh_rest,opacity)
            culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=scene.cluster.uncluster(culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity)

        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.set_compact_mask(visible_chunkid)
    else:
        culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=xyz,scale,rot,sh_0,sh_rest,opacity
        visible_chunkid=None

    nvtx.range_push("Activate")
    pad_one=torch.ones((1,culled_xyz.shape[-1]),dtype=culled_xyz.dtype,device=culled_xyz.device)
    culled_xyz=torch.concat((culled_xyz,pad_one),dim=0)
    culled_scale=culled_scale.exp()
    culled_rot=torch.nn.functional.normalize(culled_rot,dim=0)
    culled_opacity=culled_opacity.sigmoid()
    with torch.no_grad():
        camera_center=(-view_matrix[...,3:4,:3]@(view_matrix[...,:3,:3].transpose(-1,-2))).squeeze(1)
        dirs=culled_xyz[:3]-camera_center.unsqueeze(-1)
        dirs=torch.nn.functional.normalize(dirs,dim=-2)
    color=utils.wrapper.SphericalHarmonicToRGB.call_fused(actived_sh_degree,culled_sh_0,culled_sh_rest,dirs)
    nvtx.range_pop()

    return visible_chunkid,culled_xyz,culled_scale,culled_rot,color,culled_opacity

def render(view_matrix:torch.Tensor,proj_matrix:torch.Tensor,
           xyz:torch.Tensor,scale:torch.Tensor,rot:torch.Tensor,color:torch.Tensor,opacity:torch.Tensor,
           actived_sh_degree:int,output_shape:tuple[int,int],pp:arguments.PipelineParams)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    MAX_GAUSS_PER_VIEW = 15000   # try 10k–20k
    MAX_RADIUS_EST = 64.0        # pixels (projection-space)
    MIN_DEPTH = 1e-2

    #gs projection
    nvtx.range_push("Proj")
    transform_matrix=utils.wrapper.CreateTransformMatrix.call_fused(scale,rot)
    J=utils.wrapper.CreateRaySpaceTransformMatrix.call_fused(xyz,view_matrix,proj_matrix,output_shape,False)#todo script
    cov2d=utils.wrapper.CreateCov2dDirectly.call_fused(J,view_matrix,transform_matrix)
    eigen_val,eigen_vec,inv_cov2d=utils.wrapper.EighAndInverse2x2Matrix.call_fused(cov2d)
    #ndc_pos=utils.wrapper.World2NdcFunc.apply(xyz,view_matrix@proj_matrix)
    hom_pos=(xyz.transpose(0,1)@(view_matrix@proj_matrix)).transpose(1,2).contiguous()
    ndc_pos=hom_pos/(hom_pos[:,3:4,:]+1e-7)

    view_depth=(view_matrix.transpose(1,2)@xyz)[:,2]
    nvtx.range_pop()
    
    H, W = output_shape
    V = ndc_pos.shape[0]

    imgs = []
    depths_out = []
    alphas = []

    for v in range(V):
        # ---------- slice per view ----------
        ndc_v = ndc_pos[v]            # [4, P]
        depth_v = view_depth[v]       # [P]
        inv_cov_v = inv_cov2d[v]      # [2,2,P]
        color_v = color[v]            # [C,P]

        # ---------- NDC → pixel ----------
        mean_x = ((ndc_v[0] + 1) * W - 1.0) * 0.5
        mean_y = ((ndc_v[1] + 1) * H - 1.0) * 0.5
        means2D_v = torch.stack([mean_x, mean_y], dim=-1)   # [P,2]

        # ---------- reshape tensors ----------
        inv_cov_v = inv_cov_v.permute(2,0,1).contiguous()   # [P,2,2]
        color_v = color_v.permute(1,0).contiguous()         # [P,C]

        if opacity.dim() == 1:
            opacity_v = opacity.unsqueeze(-1)
        elif opacity.dim() == 2 and opacity.shape[0] == V:
            opacity_v = opacity[v].unsqueeze(-1)
        else:
            opacity_v = opacity.view(-1,1)

        # ============================================================
        # PRUNING (THIS IS THE IMPORTANT PART)
        # ============================================================

        # 1) depth validity
        valid = depth_v > MIN_DEPTH

        means2D_v = means2D_v[valid]
        depth_v = depth_v[valid]
        inv_cov_v = inv_cov_v[valid]
        color_v = color_v[valid]
        opacity_v = opacity_v[valid]

        # 2) estimate projected radius from inv_cov2d (cheap)
        a = inv_cov_v[:, 0, 0]
        b = inv_cov_v[:, 0, 1]
        c = inv_cov_v[:, 1, 1]

        trace = a + c
        det = a * c - b * b
        disc = (trace * trace - 4.0 * det).clamp(min=0.0)
        lambda_min = 0.5 * (trace - torch.sqrt(disc)).clamp(min=1e-6)

        radius_est = torch.sqrt(1.0 / lambda_min)

        # 3) radius clamp
        valid = radius_est < MAX_RADIUS_EST

        means2D_v = means2D_v[valid]
        depth_v = depth_v[valid]
        inv_cov_v = inv_cov_v[valid]
        color_v = color_v[valid]
        opacity_v = opacity_v[valid]
        radius_est = radius_est[valid]

        # 4) importance score
        importance = opacity_v.squeeze(-1) * radius_est

        # 5) top-K selection
        if importance.numel() > MAX_GAUSS_PER_VIEW:
            topk = torch.topk(
                importance,
                MAX_GAUSS_PER_VIEW,
                sorted=False
            ).indices

            means2D_v = means2D_v[topk]
            depth_v = depth_v[topk]
            inv_cov_v = inv_cov_v[topk]
            color_v = color_v[topk]
            opacity_v = opacity_v[topk]


        # ---------- rasterize ----------
        img_v, depth_v_img, alpha_v = rasterize_pytorch_view(
            means2D=means2D_v,
            depths=depth_v,
            inv_cov2d=inv_cov_v,
            color=color_v,
            opacity=opacity_v,
            H=H,
            W=W,
            white_background=True
        )

        imgs.append(img_v)
        depths_out.append(depth_v_img)
        alphas.append(alpha_v)

    # ---------- stack back ----------
    img = torch.stack(imgs, dim=0)        # [V,H,W,C]
    depth = torch.stack(depths_out, dim=0)
    alpha = torch.stack(alphas, dim=0)


    # compatibility placeholders
    transmitance = 1 - alpha
    normal = None
    lst_contributor = None
    primitive_visible = None


    if StatisticsHelperInst.bStart:
        StatisticsHelperInst.update_tile_blend_count(lst_contributor)

    return img,transmitance,depth,normal,primitive_visible
