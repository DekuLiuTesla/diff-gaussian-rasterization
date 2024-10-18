import torch
import math
from diff_trim_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ipdb import set_trace
import time


def render():
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration

    raster_settings = GaussianRasterizationSettings(
        image_height=545,
        image_width=980,
        tanfovx=0.8446965112441064,
        tanfovy=0.4679476755039769,
        bg=torch.tensor([0., 0., 0.], device='cuda:0'),
        scale_modifier=1.0,
        viewmatrix=torch.tensor(
            [[-6.9366e-02,  9.5589e-03, -9.9755e-01,  0.0000e+00],
            [ 1.5418e-01,  9.8804e-01, -1.2535e-03,  0.0000e+00],
            [ 9.8560e-01, -1.5389e-01, -7.0010e-02,  0.0000e+00],
            [-7.0630e-01,  8.6978e-03,  3.2257e+00,  1.0000e+00]], device='cuda:0'
        ),
        projmatrix=torch.tensor([[-8.2119e-02,  2.0427e-02, -9.9765e-01, -9.9755e-01],
            [ 1.8253e-01,  2.1114e+00, -1.2537e-03, -1.2535e-03],
            [ 1.1668e+00, -3.2887e-01, -7.0017e-02, -7.0010e-02],
            [-8.3616e-01,  1.8587e-02,  3.2160e+00,  3.2257e+00]], device='cuda:0'),
        sh_degree=0,
        campos=torch.tensor([3.1687, 0.1043, 0.9233], device='cuda:0'),
        prefiltered=False,
        record_transmittance=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    input_data = torch.load('./render_input_example.pth')
    means3D = input_data['means3D'].cuda('cuda:0')

    num_input = 10000
    inds = torch.randperm(means3D.shape[0]).cuda('cuda:0')[:num_input]

    means3D = means3D[inds]
    means2D = input_data['means2D'].cuda('cuda:0')[inds]
    shs = input_data['shs'].cuda('cuda:0')[inds]
    opacities = input_data['opacities'].cuda('cuda:0')[inds]
    scales = input_data['scales'].cuda('cuda:0')[inds]
    rotations = input_data['rotations'].cuda('cuda:0')[inds]

    opacities = torch.rand(opacities.shape, device=opacities.device) # initial opacities are same, make it random 


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    shared_input = torch.rand(means2D.shape[0], 3).cuda('cuda:0')
    colors_precomp = torch.nn.parameter.Parameter(shared_input)
    extra_feats = torch.nn.parameter.Parameter(shared_input.clone() * 1.0)
    rendered_image, rendered_feat, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        extra_feats = extra_feats,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    rendered_rand = torch.rand(rendered_image.shape, device='cuda:0')

    loss_img = ((rendered_image - rendered_rand) ** 2).mean()
    loss_feat = ((rendered_feat - rendered_rand) ** 2).mean()
    loss_tot = (loss_img + loss_feat) * 100
    loss_tot.backward()

    print(f'max error of forward:{(rendered_feat - rendered_image).abs().max()}')
    print(f'max relative error of forward:{((rendered_feat - rendered_image) / (rendered_feat.abs() + 1e-7)).abs().max()}')
    print(f'max error of backward:{(colors_precomp.grad - extra_feats.grad).abs().max()}')
    print(f'max relative error of backward:{((colors_precomp.grad - extra_feats.grad) / (colors_precomp.grad.abs() + 1e-7)).abs().max()}')

    # high dimension test
    shared_input = torch.rand(means2D.shape[0], 512).cuda('cuda:0')
    extra_feats = torch.nn.parameter.Parameter(shared_input)

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    
    iters = 10
    torch.cuda.synchronize()
    beg = time.time()
    for i in range(iters):
        rendered_image, rendered_feat, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            extra_feats = extra_feats,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
    torch.cuda.synchronize()
    end = time.time()
    print((end - beg) / iters)

def calculate_transmittance():
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration

    raster_settings = GaussianRasterizationSettings(
        image_height=545,
        image_width=980,
        tanfovx=0.8446965112441064,
        tanfovy=0.4679476755039769,
        bg=torch.tensor([0., 0., 0.], device='cuda:0'),
        scale_modifier=1.0,
        viewmatrix=torch.tensor(
            [[-6.9366e-02,  9.5589e-03, -9.9755e-01,  0.0000e+00],
            [ 1.5418e-01,  9.8804e-01, -1.2535e-03,  0.0000e+00],
            [ 9.8560e-01, -1.5389e-01, -7.0010e-02,  0.0000e+00],
            [-7.0630e-01,  8.6978e-03,  3.2257e+00,  1.0000e+00]], device='cuda:0'
        ),
        projmatrix=torch.tensor([[-8.2119e-02,  2.0427e-02, -9.9765e-01, -9.9755e-01],
            [ 1.8253e-01,  2.1114e+00, -1.2537e-03, -1.2535e-03],
            [ 1.1668e+00, -3.2887e-01, -7.0017e-02, -7.0010e-02],
            [-8.3616e-01,  1.8587e-02,  3.2160e+00,  3.2257e+00]], device='cuda:0'),
        sh_degree=0,
        campos=torch.tensor([3.1687, 0.1043, 0.9233], device='cuda:0'),
        prefiltered=False,
        record_transmittance=True,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    input_data = torch.load('./render_input_example.pth')
    means3D = input_data['means3D'].cuda('cuda:0')

    inds = torch.randperm(means3D.shape[0]).cuda('cuda:0')

    means3D = means3D
    means2D = input_data['means2D'].cuda('cuda:0')
    shs = input_data['shs'].cuda('cuda:0')
    opacities = input_data['opacities'].cuda('cuda:0')
    scales = input_data['scales'].cuda('cuda:0')
    rotations = input_data['rotations'].cuda('cuda:0')

    # opacities = torch.rand(opacities.shape, device=opacities.device) # initial opacities are same, make it random 


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    transmittance_sum, num_covered_pixels, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    avg_transmittance = transmittance_sum / (num_covered_pixels + 1e-6)

    set_trace()


if __name__ == '__main__':
    calculate_transmittance()
    # render()
