# 3rd party
import torch
from pytorch3d.renderer.mesh.textures import Textures, TexturesVertex, \
    TexturesUV
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import \
    look_at_view_transform, FoVPerspectiveCameras, FoVOrthographicCameras, \
    PointLights, DirectionalLights, Materials, RasterizationSettings,\
    MeshRenderer, MeshRasterizer, SoftPhongShader, OpenGLPerspectiveCameras, \
    PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, \
    PointsRenderer
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Project files.
from tcsr.models.common import Device
import externals.jblib.mesh as jbm
import externals.jblib.vis3d as jbv3

# Python std.
import math


class CameraAnimation:
    def __init__(self, motion='circular', motion_args={'speed': 90.}, dist=1.0,
                 azi=0., ele=0., fps=30):
        # Initial camera params.
        self._d_init = dist
        self._azi_init = azi
        self._ele_init = ele

        self._update_fn = {
            'circular': self._update_circular
        }[motion]
        self._motion_args = motion_args
        self._fps = fps

        self.reset()

    def reset(self):
        self._d = self._d_init
        self._azi = self._azi_init
        self._ele = self._ele_init

    def update(self):
        self._update_fn(self._motion_args)
        R, T = look_at_view_transform(self._d, self._ele, self._azi)
        return R, T

    def _update_circular(self, kwargs):
        azi_step = kwargs['speed'] / self._fps
        self._azi = math.fmod(self._azi + azi_step, 360.)


class Renderer(Device):
    def_cam = {'type': 'perspective', 'dist': 1.0, 'azi': 0., 'ele': 0.}

    def __init__(self, camera=None, camera_animation=None, T_offs=None,
                 gpu=True):
        Device.__init__(self, gpu=gpu)

        T_offs = torch.zeros((3, ), dtype=torch.float32) \
            if T_offs is None else torch.from_numpy(T_offs)
        assert T_offs.shape == (3, )
        self._T_offs = T_offs[None]

        # Placeholder.
        self._renderer = None

        ### Create camera.
        camera = Renderer.def_cam if camera is None else camera

        # Get camera tf.
        R, T = look_at_view_transform(
            camera['dist'], camera['ele'], camera['azi'])
        self._R_init = R.detach().clone()
        self._T_init = T.detach().clone()

        T += T_offs
        if camera['type'] == 'perspective':
            self._cameras = OpenGLPerspectiveCameras(
                device=self.device, R=R, T=T)
        elif camera['type'] == 'orthographic':
            self._cameras = FoVOrthographicCameras(
                znear=0., max_y=camera['max_y'], min_y=camera['min_z'],
                max_x=camera['max_x'], min_x=camera['min_x'],
                device=self.device, R=R, T=T)

        # Get camera animation.
        self._cam_anim = None
        if camera_animation is not None:
            for k in ['motion', 'motion_args', 'fps']:
                assert k in camera_animation
            camera_animation.update(
                {k: camera[k] for k in ['dist', 'azi', 'ele']})
            self._cam_anim = CameraAnimation(**camera_animation)

    def render(self):
        raise NotImplementedError

    def reset(self):
        self._set_RT(self._R_init, self._T_init)
        if self._cam_anim is not None:
            self._cam_anim.reset()

    def _update_cam(self):
        if self._cam_anim is not None:
            R, T = self._cam_anim.update()
            T += self._T_offs
            self._set_RT(R, T)

    def _set_RT(self, R, T):
        self._renderer.rasterizer.cameras.R = R.to(self.device)
        self._renderer.rasterizer.cameras.T = T.to(self.device)


class RendererPointcloud(Renderer):
    """ Point cloud renderer class.

        Args:
            img_size:
            camera:
            T_offs (np.array): Shape (3, ).
            gpu:
        """

    def __init__(self, img_size=512, pts_radius=0.012, bgrd_clr=(1., 1., 1., 0),
                 camera=None, camera_animation=None, T_offs=None, gpu=True):
        super(RendererPointcloud, self).__init__(
            camera=camera, camera_animation=camera_animation, T_offs=T_offs,
            gpu=gpu)

        raster_settings = PointsRasterizationSettings(
            image_size=img_size,
            radius=pts_radius,
            points_per_pixel=1
        )
        rasterizer = PointsRasterizer(
            cameras=self._cameras, raster_settings=raster_settings)
        self._renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=bgrd_clr))

    def render(self, pts, pts_colors=None, keep_alpha=False):
        # Check and get shapes.
        assert pts.ndim == 3
        assert pts_colors is None or pts_colors.shape == pts.shape
        B, N,  = pts.shape[:2]

        # Prepare point colors.
        pts_colors = torch.ones_like(pts) * 0.75 if pts_colors is None \
            else pts_colors
        pts_colors = torch.cat(
            [pts_colors, torch.ones((B, N, 1), device=pts_colors.device)],
            dim=2)

        # Render
        if self._cam_anim is not None:
            imgs = []
            for i in range(B):
                pc = Pointclouds(
                    points=pts[i:i + 1], features=pts_colors[i:i + 1]).to(
                    self.device)
                imgs.append(self._renderer(pc))
                self._update_cam()
            imgs = torch.cat(imgs, dim=0)
        else:
            pc = Pointclouds(
                points=pts, features=pts_colors).to(self.device)
            imgs = self._renderer(pc)
        imgs = torch.clamp(imgs, 0., 1.)[..., :(3, 4)[keep_alpha]]. \
            detach().cpu().numpy()

        # Fix weird point cloud transparency.
        imgs[..., 3][imgs[..., 3] != 0.0] = 1.

        return imgs


class RendererMesh(Renderer):
    """ Mesh renderer abstract base class.

    Args:
        img_size:
        camera:
        light_loc:
        light_colors:
        T_offs (np.array): Shape (3, ).
        gpu:
    """
    def_clrs = {'ambient': (0.8, 0.8, 0.8), 'diffuse': (0.6, 0.6, 0.6),
                'specular': (0.6, 0.6, 0.6)}

    def __init__(self, img_size=512, camera=None,
                 light_loc=np.array([[0.0, 0.0, -1.0]]),
                 light_colors=None, camera_animation=None, T_offs=None,
                 gpu=True):
        super(RendererMesh, self).__init__(
            camera=camera, camera_animation=camera_animation, T_offs=T_offs,
            gpu=gpu)

        # Get light colors.
        light_colors = RendererMesh.def_clrs if light_colors is None \
            else light_colors

        raster_settings = RasterizationSettings(
            image_size=img_size, blur_radius=0.0, faces_per_pixel=1)

        assert light_loc.ndim == 2 and light_loc.shape[1] == 3
        lights = PointLights(
            device=self.device, ambient_color=(light_colors['ambient'], ),
            diffuse_color=(light_colors['diffuse'], ),
            specular_color=(light_colors['specular'], ), location=light_loc)

        self._renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self._cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=self.device, cameras=self._cameras, lights=lights))

    def render(self):
        raise NotImplementedError


class RendererMeshVertColor(RendererMesh):
    """

    Args:
        faces (torch.Tensor[int32]): Faces, shape (F, 3).
        verts_colors (torch.Tensor[float32]): Vertices' colors, shape (V, 3).
    """
    def __init__(self, faces=None, verts_colors=None, img_size=512, camera=None,
                 light_loc=(0.0, 0.0, -1.0), light_colors=None,
                 camera_animation=None, T_offs=None, gpu=True):
        super(RendererMeshVertColor, self).__init__(
            img_size=img_size, camera=camera, light_loc=light_loc,
            light_colors=light_colors, camera_animation=camera_animation,
            T_offs=T_offs, gpu=gpu)
        if faces is not None:
            assert faces.ndim == 2 and faces.shape[1] == 3
            assert faces.dtype == torch.int32
        self._faces = faces.to(self.device) if faces is not None else faces
        if verts_colors is not None:
            assert verts_colors.ndim == 2 and verts_colors.shape[1] == 3
            assert verts_colors.dtype == torch.float32
        self._verts_colors = verts_colors.to(self.device) \
            if verts_colors is not None else verts_colors

    def render(self, verts, verts_colors=None, keep_alpha=False, faces=None):
        """

        Args:
            verts (torch.Tensor): Vertices, shape (B, N, 3).
            verts_colors (torch.Tensor): Vertices' colors, shape (B, N, 3).
            keep_alpha (bool): Whethe to keep the 4th alpha channel.
            faces (torch.Tensor): Faces, shape (F, 3). If not provided,
                the object property `_faces` is used.

        Returns:

        """
        assert verts.ndim == 3
        B = verts.shape[0]

        # Prepare vertices and vertex colors.
        verts = verts.to(self.device)
        if verts_colors is not None:
            assert verts_colors.shape == verts.shape
            assert verts_colors.dtype == torch.float32
            vert_colors = verts_colors.to(self.device)
        else:
            # vert_colors = self._verts_colors[None].expand(B, -1, -1)
            vert_colors = torch.ones_like(verts)

        # Prepare faces.
        faces = (faces, self._faces)[faces is None]
        assert faces is not None and faces.ndim == 2 and faces.shape[1] == 3
        faces = faces.expand(B, -1, -1).to(self.device)

        # Update cam and render.
        if self._cam_anim is not None:
            imgs = []
            for i in range(B):
                textures = Textures(verts_rgb=vert_colors[i:i + 1])
                mesh = Meshes(verts[i:i + 1], faces[i:i + 1], textures)
                imgs.append(self._renderer(mesh))
                self._update_cam()
            imgs = torch.cat(imgs, dim=0)
        else:
            textures = Textures(verts_rgb=vert_colors)
            mesh = Meshes(verts, faces, textures)
            imgs = self._renderer(mesh)

        imgs = torch.clamp(imgs, 0., 1.)[..., :(3, 4)[keep_alpha]]. \
            detach().cpu().numpy()

        # Fix weird transparency.
        imgs[..., 3][imgs[..., 3] > 0.4] = 1.

        return imgs


class RendererPatchesUV(RendererMesh):
    """

    Args:
        faces (torch.Tensor[int32]): Faces, shape (F, 3).
    """
    def __init__(self, mesh_edge_verts, num_patches, path_texture,
                 patch_margin=0.02, img_size=512, camera=None,
                 light_loc=(0.0, 0.0, -1.0), light_colors=None,
                 camera_animation=None, T_offs=None, uv_style='multi_patch',
                 uv_style_kwargs=None, gpu=True):
        super(RendererPatchesUV, self).__init__(
            img_size=img_size, camera=camera, light_loc=light_loc,
            light_colors=light_colors, camera_animation=camera_animation,
            T_offs=T_offs, gpu=gpu)
        assert uv_style in ('multi_patch', 'camera_plane')

        self._mev = mesh_edge_verts
        self._spp = mesh_edge_verts ** 2
        self._P = num_patches

        faces = jbm.grid_faces(mesh_edge_verts, mesh_edge_verts)
        self._faces = self._prepare_faces_multiple_patches(
            faces, num_patches, self._spp)  # (P * F, 3)

        # Prepare UVs and texture.
        self._textorch = None
        if uv_style == 'multi_patch':
            self._uv = self._prepare_uvs_cat_texture(
                num_patches, mesh_edge_verts)  # (M, 2)
            self._textorch = torch.from_numpy(
                self._generate_textures_with_margin(
                    Image.open(path_texture), num_patches, m=patch_margin,
                    cat=True)).to(self.device)  # (H, P * W, 3)
        elif uv_style == 'camera_plane':
            if uv_style_kwargs is not None:
                self._prepare_uvs_cam_plane(
                    uv_style_kwargs['pts_ref'], uv_style_kwargs['cam_azi'],
                    uv_style_kwargs['cam_ele'])
            self._textorch = torch.from_numpy(
                plt.imread(path_texture)[..., :3]).to(self.device)  #(H,W,3)

    def _generate_textures_with_margin(self, tex, n, m=0.05, cat=False):
        """ Creates `n` new images by adding a colored margin of size
        `m` * tex.shape[0] to the original image. Each new image has
        different color of a margin.

        Args:
            tex (PIL.Image): Original image.
            n (int): Number of images to create.
            m (float): Margin size as a fraction of original image size.

        Returns:
            np.array[float32]: New textures, shape (N, H, W, 3).
        """
        # Convert to numpy.
        texnp = np.asarray(tex)[..., :3].astype(np.float32) / 255.
        assert texnp.ndim == 3
        h, w = texnp.shape[:2]

        # Get new size.
        smarg = int(np.round(m * h))
        snew = h + smarg + smarg

        # Get colors.
        clr_dict = jbv3.get_contrast_colors()
        clr_keys = list(clr_dict.keys())

        # Process all images.
        imgs = []
        for i in range(n):
            clr = np.array(clr_dict[clr_keys[i]])
            texnpm = np.ones((snew, snew, 3), dtype=np.float32) * clr
            texnpm[smarg:smarg + h, smarg:smarg + w] = texnp
            imgs.append(texnpm)
        imgs = np.stack(imgs, axis=0)

        # Concatenate along width.
        if cat:
            imgs = np.concatenate(imgs, axis=1)

        return imgs.astype(np.float32)

    def _prepare_faces_multiple_patches(self, faces, num_patches, spp):
        """
        Returns:
            torch.Tensor[int32]: (P * F, 3)
        """
        P = num_patches
        faces = np.tile(faces, (P, 1, 1)) + \
        np.arange(P)[:, None, None] * spp  # (P, spp, 3)
        faces = faces.reshape((-1, 3))  # (P*F, 3)
        return torch.from_numpy(faces).to(self.device)

    def _prepare_uvs_cat_texture(self, num_patches, mesh_edge_verts):
        """
        Returns:
            torhc.Tensor[float32]: (M, 2)
        """
        P = num_patches
        uv = jbm.grid_verts_2d(
            mesh_edge_verts, mesh_edge_verts, 1., 1.)  # (spp, 2)
        uv = np.tile(uv, (P, 1, 1))  # (P, spp, 2)
        uv[..., 0] = uv[..., 0] / P
        uv += np.stack([np.linspace(0., 1., P + 1)[:-1],
                        np.zeros((P,))], axis=1)[:, None, :]
        uv = uv.reshape((-1, 2)).astype(np.float32)  # (P * spp, 2)
        return torch.from_numpy(uv).to(self.device)

    def _prepare_uvs_cam_plane(self, pts_ref, cam_azi=0., cam_ele=0.):
        """ Projects the object `pts_ref` to the camera plane orthographically.
        The camera orientation is given by `cam_azi` and `cam_ele`. Both azi
        and ele are w.r.t. the base vector along the Z axis (0, 0, 1). Moving
        camera up gives positive ele. Moving camare to the right (when looking
        against the Z axis) gives positive azi.

        Args:
            pts_ref: Pcloud, shape (N, 3).
            cam_azi:
            cam_ele:

        Returns:

        """
        # Get rotation of the object.
        Ra = Rotation.from_euler(
            'xyz', [0., -cam_azi, 0.], degrees=True).as_matrix()
        Re = Rotation.from_euler(
            'xyz', [cam_ele, 0., 0.], degrees=True).as_matrix()
        R = Re @ Ra

        # Rotate the object.
        ptsr = (R @ pts_ref.T).T

        # Get uvs - hacky
        mn = -0.5
        mx = 1.5
        ptsn = ((ptsr - np.min(ptsr, axis=0)) /
                (np.max(ptsr, axis=0) - np.min(ptsr, axis=0))) * \
               (mx - mn) + mn

        uv = ptsn[:, :2].astype(np.float32)
        self._uv = torch.from_numpy(uv).to(self.device)

    def render(self, verts, keep_alpha=False):
        """

        Args:
            verts (torch.Tensor): Vertices, shape (B, N, 3).
            verts_colors (torch.Tensor): Vertices' colors, shape (B, N, 3).
            keep_alpha (bool): Whethe to keep the 4th alpha channel.

        Returns:

        """
        B = verts.shape[0]

        # Prepare faces.
        faces = self._faces[None].expand(B, -1, -1)

        # Update cam and render.
        if self._cam_anim is not None:
            imgs = []
            for i in range(B):
                t = Textures(
                    maps=self._textorch[None], faces_uvs=faces[i:i + 1],
                    verts_uvs=self._uv[None])
                m = Meshes(verts[i:i + 1], faces[i:i + 1], t)
                imgs.append(self._renderer(m))
                self._update_cam()
            imgs = torch.cat(imgs, dim=0)
        else:
            t = TexturesUV(
                maps=self._textorch[None].expand(B, -1, -1, -1),
                faces_uvs=self._faces[None].expand(B, -1, -1),
                verts_uvs=self._uv[None].expand(B, -1, -1),
                padding_mode='reflection')
            m = Meshes(verts=verts, faces=faces, textures=t)
            imgs = self._renderer(m)

        imgs = torch.clamp(imgs, 0., 1.)[..., :(3, 4)[keep_alpha]]. \
            detach().cpu().numpy()

        # Fix weird transparency.
        imgs[..., 3][imgs[..., 3] > 0.4] = 1.

        return imgs


class RendererMeshVaryingUV(RendererMesh):
    """

    Args:
        faces (torch.Tensor[int32]): Faces, shape (F, 3).
    """
    def __init__(self, path_texture, faces, img_size=512, camera=None,
                 light_loc=(0.0, 0.0, -1.0), light_colors=None,
                 camera_animation=None, T_offs=None, gpu=True):
        super(RendererMeshVaryingUV, self).__init__(
            img_size=img_size, camera=camera, light_loc=light_loc,
            light_colors=light_colors, camera_animation=camera_animation,
            T_offs=T_offs, gpu=gpu)

        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces.astype(np.int32))

        self._faces = faces.to(self.device)
        self._textorch = torch.from_numpy(
            plt.imread(path_texture)[..., :3]).to(self.device)  # (H, W, 3)

    def render(self, verts, uvs, keep_alpha=False):
        """

        Args:
            verts (torch.Tensor): Vertices, shape (B, N, 3).
            uvs (torch.Tensor[float32]): UVs, shape (B, N, 2).
            keep_alpha (bool): Whethe to keep the 4th alpha channel.

        Returns:

        """
        B, N = verts.shape[:2]
        assert uvs.ndim == 3
        assert uvs.shape == (B, N, 2)

        # Prepare UVs.
        if isinstance(uvs, np.ndarray):
            uvs = torch.from_numpy(uvs.astype(np.float32))
        if uvs.device != verts.device:
            uvs.to(verts.device)

        # Prepare faces.
        faces = self._faces[None].expand(B, -1, -1)

        # Update cam and render.
        if self._cam_anim is not None:
            imgs = []
            for i in range(B):
                t = Textures(
                    maps=self._textorch[None], faces_uvs=faces[i:i + 1],
                    verts_uvs=uvs[i:i + 1])
                m = Meshes(verts[i:i + 1], faces[i:i + 1], t)
                imgs.append(self._renderer(m))
                self._update_cam()
            imgs = torch.cat(imgs, dim=0)
        else:
            t = TexturesUV(
                maps=self._textorch[None].expand(B, -1, -1, -1),
                faces_uvs=self._faces[None].expand(B, -1, -1).type(torch.int64),
                verts_uvs=uvs, padding_mode='reflection')
            m = Meshes(verts=verts, faces=faces, textures=t)
            imgs = self._renderer(m)

        imgs = torch.clamp(imgs, 0., 1.)[..., :(3, 4)[keep_alpha]]. \
            detach().cpu().numpy()

        # Fix weird transparency.
        imgs[..., 3][imgs[..., 3] > 0.4] = 1.

        return imgs
