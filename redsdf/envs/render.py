import os
import threading
import time
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tr
from pybullet_data import getDataPath
from scipy.spatial.transform.rotation import Rotation

pybullet_path = getDataPath()


def transformation_matrix(translation, quat=None):
    M = np.eye(4)
    if quat:
        M[:3, :3] = Rotation.from_quat(quat).as_matrix()
    if translation:
        M[:3, 3] = translation
    return M


def transformation_scale(scale):
    M = np.eye(4)
    if np.any(scale != 1):
        M[:3, :3] *= scale
    return M


class CustomViewer(pyrender.Viewer):
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        mindim = 0.3 * np.min(self._trackball._size)
        dx_, dy_ = np.array([x, y]) - self._trackball._pdown

        if self._trackball._state == pyrender.viewer.Trackball.STATE_ROTATE:
            x_angle = -dx_ / mindim
            y_angle = dy_ / mindim
            x_rot_mat = tr.rotation_matrix(x_angle, np.array([0., 0., 1.]), self._trackball._target)
            y_axis = self._trackball._n_pose[:3, 0].flatten()
            y_axis = y_axis - y_axis.dot(np.array([0., 0., 1.])) * np.array([0., 0., 1.])
            y_rot_mat = tr.rotation_matrix(y_angle, y_axis, self._trackball._target)
            self._trackball._n_pose = y_rot_mat.dot(x_rot_mat.dot(self._trackball._pose))


class Render:
    def __init__(self, pb_client=None, refresh_rate=30.):
        self.pb_client = pb_client
        self.refresh_rate = refresh_rate
        self.scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.5, 0.5, 0.5))

        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5)
        light_pose = tr.euler_matrix(0., -np.pi / 6, 0., axes='rzyz')
        self.scene.add(self.light, pose=light_pose)

        self.camera = pyrender.PerspectiveCamera(np.pi / 2, 0.1, 500)
        camera_pose = tr.translation_matrix([0.5, 0., 1.0]) @ tr.euler_matrix(np.pi / 6, np.pi / 4, 0., axes='rzxz') @ tr.translation_matrix([0., 0., 1.5])
        self.scene.add(self.camera, pose=camera_pose)

        self.ground = self.scene.add(pyrender.Mesh.from_trimesh(
            trimesh.load_mesh(os.path.join(pybullet_path, "plane.obj"))))

        self.viewer = CustomViewer(self.scene, viewport_size=(1024, 768), run_in_thread=True)
        self.thread = threading.Thread(target=self.start_render)
        self.point_cloud_color = [1., 0., 0.]
        time.sleep(0.5)

        self.pb_models = list()
        self.smpl_models = dict()
        self.pcl_models = dict()

    def register_smpl_model(self, model_name, mesh, translation=None, rotation=None):
        self.viewer.render_lock.acquire()
        node = pyrender.Node(name=model_name, mesh=mesh, translation=translation, rotation=rotation)
        self.scene.add_node(node)
        self.smpl_models[model_name] = {'mesh': mesh, 'node': node,
                                        'translation': node.translation, 'rotation': node.rotation}
        self.viewer.render_lock.release()

    def update_smpl_model(self, model_name, mesh=None, translation=None, rotation=None):
        if mesh is not None:
            self.smpl_models[model_name]['mesh'] = mesh
        if translation is not None:
            self.smpl_models[model_name]['translation'] = translation
        if rotation is not None:
            self.smpl_models[model_name]['rotation'] = rotation

    def register_pcl_model(self, model_name, pcl, translation=None, rotation=None):
        self.viewer.render_lock.acquire()
        mesh = pyrender.Mesh.from_points(pcl, colors=self.point_cloud_color)
        node = pyrender.Node(name=model_name, mesh=mesh, translation=translation, rotation=rotation)
        self.scene.add_node(node)
        self.pcl_models[model_name] = {'pcl': mesh, 'node': node,
                                       'translation': node.translation, 'rotation': node.rotation}
        self.viewer.render_lock.release()

    def update_pcl_model(self, model_name, pcl=None, translation=np.array([0, 0, 0]), rotation=np.array([0, 0, 0, 1])):
        if pcl is not None:
            self.pcl_models[model_name]['pcl'] = pyrender.Mesh.from_points(pcl, colors=self.point_cloud_color)
        if translation is not None:
            self.pcl_models[model_name]['translation'] = translation
        if rotation is not None:
            self.pcl_models[model_name]['rotation'] = rotation


    def init_pybullet_model(self, model_id, link_mask=None):
        self.viewer.render_lock.acquire()
        visual_data_list = self.pb_client.getVisualShapeData(model_id)
        for visual_data in visual_data_list:
            link_idx = visual_data[1]
            if link_mask:
                if link_idx not in link_mask:
                    continue
            # link_name = self.pb_client.getJointInfo(model_id, link_idx)[12].decode("ascii")
            size = visual_data[3]
            mesh_type = visual_data[2]
            if mesh_type == self.pb_client.GEOM_MESH:
                mesh_file = visual_data[4].decode("ascii")
                mesh = trimesh.load(mesh_file)
            elif mesh_type == self.pb_client.GEOM_SPHERE:
                mesh = trimesh.primitives.Sphere(radius=size[0])
                size = np.ones(3)
            elif mesh_type == self.pb_client.GEOM_BOX:
                mesh = trimesh.primitives.Box(extents=size)
                size = np.ones(3)
            elif mesh_type == self.pb_client.GEOM_CYLINDER:
                mesh = trimesh.primitives.Cylinder(height=size[0], radius=size[1])
                size = np.ones(3)
            elif mesh_type == self.pb_client.GEOM_CAPSULE:
                transform = np.eye(4)
                transform[2, 3] = -size[0]/2
                mesh = trimesh.primitives.Capsule(height=size[0], radius=size[1], transform=transform)
                size = np.ones(3)
            else:
                continue
            mesh.visual.vertex_colors = visual_data[7]

            if self.pb_client.getNumJoints(model_id) > 0:
                link_state = self.pb_client.getLinkState(model_id, link_idx)
                local_transform = np.linalg.inv(transformation_matrix(link_state[2], link_state[3])) @ \
                    transformation_matrix(visual_data[5], visual_data[6]) @ transformation_scale(size)
            else:
                link_state = self.pb_client.getBasePositionAndOrientation(model_id)
                local_transform = transformation_matrix(visual_data[5], visual_data[6]) @ transformation_scale(size)

            link_mesh = pyrender.Mesh.from_trimesh(mesh, poses=local_transform)

            link_world_frame = transformation_matrix(link_state[0], link_state[1])
            node = self.scene.add(link_mesh, pose=link_world_frame)

            self.pb_models.append([model_id, link_idx, node])
        self.viewer.render_lock.release()

    def start_render(self):
        while self.viewer.is_active:
            t_start = time.time()
            self.viewer.render_lock.acquire()
            self.render_pybullet_model()
            self.render_smpl_model()
            self.render_pcl_model()
            self.viewer.render_lock.release()
            time.sleep(max(0., 1. / self.refresh_rate - (time.time() - t_start)))

    def render_pybullet_model(self):
        for model_id, link_id, node in self.pb_models:
            if self.pb_client.getNumJoints(model_id) > 0:
                link_state = self.pb_client.getLinkState(model_id, link_id)
            else:
                link_state = self.pb_client.getBasePositionAndOrientation(model_id)
            # link_pose = tr.translation_matrix(link_state[0]) @ tr.quaternion_matrix(quat_pb_to_tr(link_state[1]))
            link_pose = transformation_matrix(link_state[0], link_state[1])
            self.scene.set_pose(node, link_pose)

    def render_smpl_model(self):
        for key, value in self.smpl_models.items():
            value['node'].mesh = value['mesh']
            value['node'].translation = value['translation']
            value['node'].rotation = value['rotation']

    def render_pcl_model(self):
        for key, value in self.pcl_models.items():
            value['node'].mesh = value['pcl']
            value['node'].translation = value['translation']
            value['node'].rotation = value['rotation']

    def start(self):
        self.thread.start()

    def close(self):
        self.viewer.close_external()
