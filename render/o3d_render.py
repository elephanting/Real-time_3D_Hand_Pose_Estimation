'''
@author: lingteng qiu
@version: 1.0
render using o3d, Note that this method actually is Non-differentiable 
'''
from transforms3d.axangles import axangle2mat
import open3d as o3d
import numpy as np
import torch
import cv2
from manopth.manolayer import ManoLayer


class o3d_render():
    def __init__(self, window_size , num_hand, visible=True):
        #why need view_mat
        #To my best know, it could make sure that the solver z >0 in uv coordinate in camera function. 
        '''
        __viewer :view vision window
        hand_mesh : hand mesh graph
        view_mat : view matrix
        '''
        self.window_size = window_size
        self.num_hand = num_hand
        self.__viewer, self.hand_mesh, self.mesh, self.view_mat = self.__init_platform__(window_size, visible)
        print("viewer done")
        self.__view_control = self.render.get_view_control()
        self.__camera_params = self.__view_control.convert_to_pinhole_camera_parameters()
        self.render.update_renderer()

    def capture_img(self):
        render_img = self.render.capture_screen_float_buffer()#(do_render=True)
        render_img = np.asarray(render_img)*255
        render_img = render_img.astype(np.uint8)
        #self.display.blit(pygame.surfarray.make_surface(cv2.resize(render_img, (self.window_size, self.window_size))), (0, 0))
        #pygame.display.update()
        # RGB 2 BGR
        return render_img[...,::-1].copy()
    
    def projector(self):  
        mat = np.matmul(self.intrinsic,self.extrinsic[:3,:])
        vertices = np.asarray(self.mesh.vertices)       
        mat = mat[:,:3]
        uv = np.matmul(mat,vertices.T).T
        z = uv[:,2:3]
        uv = uv/z 
        
        return uv

    def __init_platform__(self, window_size, visible=False, device=torch.device('cuda')):
        view_mat = axangle2mat([1, 0, 0], 1*np.pi) # align different coordinate systems
        hand_mesh = ManoLayer(mano_root='mano/models', use_pca=False)

        viewer = o3d.visualization.Visualizer()
        viewer.create_window(
            width=window_size[1] + 1, height=window_size[0] + 1,
            window_name='Hand',visible=visible)

        # transfering meter to mm
        mano_mesh = np.matmul(view_mat, hand_mesh.th_v_template[0].numpy().T).T * 1000
        meshes = []
        for i in range(self.num_hand):
            mesh = o3d.geometry.TriangleMesh()
            mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.th_faces.numpy())
            mesh.vertices = o3d.utility.Vector3dVector(mano_mesh)
            mesh.compute_vertex_normals()
            viewer.add_geometry(mesh)
            meshes.append(mesh)

        self.hand_mesh = hand_mesh
        self.mesh = meshes
        self.view_mat = view_mat
        return viewer, hand_mesh, meshes, view_mat
    def environments(self,json_config,depth=None):
        if depth is not None:
            self.__view_control.set_constant_z_far(depth)
        render_option = self.render.get_render_option()
        render_option.load_from_json(json_config)
        self.render.update_renderer()
    def updata_params(self):
        self.__view_control.convert_from_pinhole_camera_parameters(self.camera_params)
        self.__camera_params = self.__view_control.convert_to_pinhole_camera_parameters()

    @property
    def render(self):
        return self.__viewer
    @property
    def camera_params(self):
        return self.__camera_params
    @property 
    def extrinsic(self):
        return self.camera_params.extrinsic.copy()
    @property
    def intrinsic(self):
        return self.camera_params.intrinsic.intrinsic_matrix
    @extrinsic.setter
    def extrinsic(self,extri):
        self.camera_params.extrinsic = extri
    @intrinsic.setter
    def intrinsic(self,para):
        CAM_FX,CAM_FY = para
        self.camera_params.intrinsic.set_intrinsics(
        self.window_size[1] + 1, self.window_size[0] + 1, CAM_FX, CAM_FY,
        self.window_size[1] // 2, self.window_size[0] // 2
        )
    
    def rendering(self, v, hand_color, index):
        if v is not None:
            self.mesh[index].triangles = o3d.utility.Vector3iVector(self.hand_mesh.th_faces)
            self.mesh[index].vertices = o3d.utility.Vector3dVector(np.matmul(self.view_mat, v.T).T)
            self.mesh[index].compute_triangle_normals()
            self.mesh[index].compute_vertex_normals()
            self.mesh[index].paint_uniform_color(hand_color)
        else:
            self.mesh[index].triangles = o3d.utility.Vector3iVector([])
            self.mesh[index].vertices = o3d.utility.Vector3dVector([])
        self.render.update_geometry(self.mesh[index])
        self.render.poll_events()
        self.render.update_renderer()
