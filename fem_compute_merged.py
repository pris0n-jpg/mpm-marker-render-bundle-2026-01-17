# Deepresearch bundle (FEM compute merged)
#
# 说明：为了减少文件数量（便于一次性上传/投喂给 deepresearch AI），把 xenmpm 里的 FEM 计算核心模块合并到单文件。
# 注意：该文件主要用于代码阅读/分析，不保证可直接运行。
# 源：F:/workspace/xenmpm
#


# ===== BEGIN FILE: xengym/fem/simpleCSR.py =====
import os
import numpy as np
# from ezgl.experimental import CUDA_AVAILABLE
CUDA_AVAILABLE = False
CUDA_AVAILABLE = CUDA_AVAILABLE and os.environ.get("CUDA_CSR", True) # 关闭 CUDA 加速

if CUDA_AVAILABLE:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg
    from cupyx.scipy.sparse import csr_matrix
else:
    from scipy.sparse.linalg import cg
    from scipy.sparse import csr_matrix


class SimpleCSR:

    def __init__(self, data, indices, indptr, shape, top_diag_offset_in_data=None):
        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.shape = shape

        # 实际的 csr_matrix 对象
        self._csr_matrix = None
        self._csr_matrix_pending = True

        if top_diag_offset_in_data is None:
            # 把顶面节点 xyz 对应的对角元都添加上值用于在 data 中占位, 避免频繁 insert, 顶面节点是前 1/3 个节点
            self.top_diag_offset_in_data = np.zeros(self.shape[0]//3, dtype=np.int32)
            for i in range(self.shape[0]//3):
                offset = self._set((i, i), self[i, i])
                self.top_diag_offset_in_data[i] = offset
        else:
            self.top_diag_offset_in_data = top_diag_offset_in_data  # no need to copy

    @property
    def csr_matrix(self):
        if self._csr_matrix_pending:
            self._csr_matrix = self._to_csr_matrix()
            self._csr_matrix_pending = False
        return self._csr_matrix

    def set_row_zero(self, start_row, end_row):
        """
        把矩阵的连续行设为 0
        """
        self._csr_matrix_pending = True
        self.data[self.indptr[start_row]:self.indptr[end_row]] = 0

    def set_diag_one(self, idx):
        """
        把矩阵的对角元设为 1
        """
        self._csr_matrix_pending = True
        offset = self.top_diag_offset_in_data[idx]
        self.data[offset] = 1

    def __getitem__(self, idx):
        """获取矩阵元素，通过 (row, col) 索引"""
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            # 查找对应的列
            start, end = self.indptr[row], self.indptr[row + 1]
            for i in range(start, end):
                if self.indices[i] == col:
                    return self.data[i]
            return 0  # 若未找到，返回0
        else:
            raise NotImplementedError("仅支持 (row, col) 索引方式")

    def _set(self, idx, value):
        """通过 (row, col) 索引设置矩阵元素, 返回元素在 data 中的索引"""
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            start, end = self.indptr[row], self.indptr[row + 1]
            for i in range(start, end):
                if self.indices[i] == col:
                    self.data[i] = value
                    return i
            # 若没有找到该位置的元素，则需要在数据中插入
            self.indices = np.insert(self.indices, end, col)
            self.data = np.insert(self.data, end, value)
            self.indptr[row + 1:] += 1  # 更新后续行的指针
        else:
            raise NotImplementedError("仅支持 (row, col) 索引方式")
        return end

    def copy(self):
        return SimpleCSR(self.data, self.indices, self.indptr, self.shape, self.top_diag_offset_in_data)

    def _to_csr_matrix(self):
        if CUDA_AVAILABLE:
            return csr_matrix((cp.array(self.data), cp.array(self.indices), cp.array(self.indptr)), shape=tuple(self.shape))
        else:
            return csr_matrix((self.data, self.indices, self.indptr), shape=tuple(self.shape))

    def __matmul__(self, other):
        if CUDA_AVAILABLE:
            return (self.csr_matrix @ other).get()
        else:
            return self.csr_matrix @ other

    @classmethod
    def solve(cls, A: 'SimpleCSR', P: np.ndarray, x0: np.ndarray):
        """
        使用共轭梯度法求解线性方程组 Ax = P

        Parameters:
        - A : SimpleCSR, 系数矩阵
        - P : array-like,
        - x0 : array-like, 初始估计

        Returns:
        - x : array-like, 位移
        """
        if CUDA_AVAILABLE:
            return cg(A.csr_matrix, cp.array(P), x0=cp.array(x0), tol=0.01, maxiter=90)
        else:
            return cg(A.csr_matrix, P, x0=x0, rtol=0.02, maxiter=50)
# ===== END FILE: xengym/fem/simpleCSR.py =====


# ===== BEGIN FILE: xengym/fem/simulation.py =====
import numpy as np
from pathlib import Path
from typing import Sequence

from xensesdk.ezgl import Matrix4x4
from xensesdk.ezgl.experimental import compute_normals

from .simpleCSR import SimpleCSR


def make_neighbors(num_nodes, indices):
    neighbors = [[] for _ in range(num_nodes)]

    for quad in indices:
        for i in range(4):
            neighbors[quad[i]].extend([quad[(i+1)%4], quad[(i+3)%4]])

    # 去重邻居顶点
    for i in range(num_nodes):
        neigh = list(set(neighbors[i]))
        if len(neigh) > 4:
            neigh = neigh[:4]
        elif len(neigh) < 4:
            neigh.extend([i] * (4 - len(neigh)))
        neigh.append(i)
        neighbors[i] = neigh
    return np.array(neighbors, np.int32)

def get_smoother(neighbors):
    """
    Parameters:
    - neighbors : np.ndarray of shape(n, m), n 为节点数, m 为每个节点的邻居数
    """
    def smoother(vert_wise_data, niters=1):
        """
        平滑顶点数据

        Parameters:
        - vert_wise_data : np.ndarray(n, k), 顶点数据, n为顶点数, k为数据维度
        - niters : int, optional, default: 1, 迭代次数
        """
        weights = (np.array([1,1,1,1,3]) / 7).reshape(1, -1, 1)
        for _ in range(niters):
            extracted_data = np.take_along_axis(vert_wise_data[None, ...], neighbors[..., None], axis=1)
            vert_wise_data = np.einsum("ijk,ijk->ik", extracted_data, weights)  # shape=(n, k)

        return vert_wise_data
    return smoother

class FemSolver:

    def __init__(self, Mf: SimpleCSR, Kf: SimpleCSR, nonlinear_coef=0.0):
        """
        有限元求解器
        已知 K, P, 求解 V, 满足 K * V = P, 刚度矩阵K, 位移V, 外部力载荷 P

        Parameters:
        - Mf : _type_, 带底部约束质量矩阵
        - Kf : _type_, 带底部约束刚度矩阵
        """
        self.Mf = Mf        # 带底部约束质量矩阵M
        self.Kf_init = Kf        # 带底部约束刚度矩阵K
        self.Kf = None
        self.node_num = Kf.shape[0] // 3

        self.alpha = 0.01       # 瑞利阻尼 α: 0.005-0.02
        self.beta = 0.06        # 瑞利阻尼 β: 0.02-0.1
        self.nonlinear_coef = nonlinear_coef

    def solve_fric(self, constrain_idx, node_disp, P, update_Kf=True):
        """
        已知位移约束, 求解外部力载荷

        Parameters:
        - constrain_idx : List[int], 被约束的节点编号
        - node_disp : array-like of (n_nodes, 3), 所有节点的位移初始估计, 其中被约束的位置为约束位移
        - update_Kf : bool, optional, default: False, 是否更新约束矩阵
        - P : array-like of (n_nodes, 3), optional, 外部力载荷

        Returns:
        - displacement: array-like of (n_nodes, 3), 位移
        - F : array-like of (n_nodes, 3), 外部力载荷
        """
        node_disp = node_disp.reshape(-1, 3)
        # 构造约束
        P = np.zeros_like(node_disp) if P is None else P.reshape(-1, 3)
        P[constrain_idx] = node_disp[constrain_idx]
        P = P.flatten()

        if update_Kf:
            self.setup_Kf(constrain_idx)

        # displacement, info = sp.linalg.cg(K, P, x0=x0.flatten(), tol=0.01, maxiter=100)
        displacement, _ = SimpleCSR.solve(self.Kf, P, x0=node_disp.flatten())
        F = self.Kf_init @ displacement
        # 引入非线性参数
        # if self.nonlinear_coef != 0:
        F += self.nonlinear_coef * displacement**2

        if not isinstance(displacement, np.ndarray):
            displacement = displacement.get()

        return displacement.reshape(-1, 3), F.reshape(-1, 3)

    def solve_static(self, constrain_idx, node_disp, update_Kf=True):
        return self.solve_fric(constrain_idx, node_disp, None, update_Kf)

    def setup_Kf(self, constrain_idx):
        """
        施加约束

        Parameters:
        - K : 刚度矩阵
        - constrain_idx : array-like of (n, ), 约束节点编号
        """
        self.Kf = self.Kf_init.copy()

        if len(constrain_idx) == 0:
            return

        # 把连续的行归并
        n = len(constrain_idx)
        start = 0
        for i in range(1, n+1):
            if i==n or constrain_idx[i]-constrain_idx[i-1]>1:
                self.Kf.set_row_zero(constrain_idx[start]*3, constrain_idx[i-1]*3+3)
                start = i

        idx = np.repeat(3 * constrain_idx[:, None], repeats=3, axis=1) + np.array([0, 1, 2])
        idx = idx.flatten()
        self.Kf.set_diag_one(idx)


class ContactState:

    def __init__(self, node_idx):
        """
        用于记录表层节点的接触状态

        Parameters:
        - node_idx : List[int], 顶层节点的全局编号
        """
        num_nodes = len(node_idx)
        self.flag = np.zeros(num_nodes, dtype=np.bool_)
        self.idx = np.vstack([np.arange(num_nodes), node_idx]).astype(np.uint32)

    def contact_idx(self) -> np.ndarray:
        # [0] - 表层节点编号 top_idx
        # [1] - 节点编号  node_idx
        return self.idx[:, self.flag]

    def no_contact_idx(self) -> np.ndarray:
        # [0] - 表层节点编号 top_idx
        # [1] - 节点编号  node_idx
        return self.idx[:, ~self.flag]

    def set(self, top_idx: Sequence[int], flag: int):
        self.flag[top_idx] = flag


class MarkerInterp:

    def __init__(self, marker_row_col, dx_mm, dy_mm, top_vert):
        """
        斑点插值

        Parameters:
        - marker_row_col : tuple, default: (10, 14), 斑点行列数
        - dx_mm, dy_mm : float, default: (1.3, 1.3), marker 间距
        - top_vert : np.ndarray, shape=(n_top_nodes, 3), dtype=np.float32, 表层节点坐标
        """
        self.row, self.col = marker_row_col
        self.dx_mm = dx_mm
        self.dy_mm = dy_mm

        x = np.linspace(0, self.dx_mm * (self.col - 1), self.col) - self.dx_mm * (self.col - 1) / 2
        y = np.linspace(0, self.dy_mm * (self.row - 1), self.row) - self.dy_mm * (self.row - 1) / 2 + top_vert[:, 1].mean()
        # y = np.linspace(0, self.dy_mm * (self.row - 1), self.row) - self.dy_mm * (self.row - 1) / 2

        xx, yy = np.meshgrid(x, y, indexing="xy")
        xy = np.stack((xx, yy), axis=2).reshape(-1, 1, 2)

        top_xy = top_vert[None, :, :2]  # shape=(1, n_top_nodes, 2)

        # 查找距离每个点最近的四个点
        dist_mat = np.linalg.norm(xy - top_xy, axis=2, keepdims=False)  # shape=(row*col, n_top_nodes)
        self.idx = np.argsort(dist_mat, axis=1)[:, :4]  # shape=(row*col, 4)
        nearest_xy = np.take_along_axis(top_xy, self.idx[:, :, None], axis=1)  # shape=(row*col, 4, 2)

        # 计算权重
        self.weights = np.zeros((self.row * self.col, 4), dtype=np.float32)
        for i in range(self.row * self.col):
            dxy = np.abs(xy[i] - nearest_xy[i])  # shape=(4, 2)
            u1, v1 = np.min(dxy, axis=0, keepdims=False)  # shape=(2,)
            u2, v2 = np.max(dxy, axis=0, keepdims=False)
            self.weights[i] = np.sort([u1*v1, u1*v2, u2*v1, u2*v2])[::-1] / ((u1+u2) * (v1+v2))

    def interp(self, top_vert):
        """
        插值

        Parameters:
        - top_vert : np.ndarray, shape=(n_top_nodes, 3), dtype=np.float32, 表层节点坐标, 单位 mm
        """
        top_vert = top_vert[None, :, :]  # shape=(1, n_top_nodes, 3) -> broadcast to (row*col, n_top_nodes, 3)
        marker = np.einsum("ijk,ijk->ik", self.weights[:, :, None], np.take_along_axis(top_vert, self.idx[:, :, None], axis=1))  # shape=(row*col, 3)
        return marker.reshape(self.row, self.col, 3)


class FEMSimulator:
    def __init__(
        self,
        view,
        fem_data_path,
        marker_row_col,
        marker_dx_dy_mm,  # unit: mm
        depth_size,
        gel_size_mm,  # unit: mm
        raw_data=None,
        nonlinear_coef=0.0,
    ):
        """
        传感器有限元仿真

        Parameters:
        - fem_data_path : Path, fem 数据路径
        - marker_row_col : list, optional, default: [10, 14], marker 行列数
        - marker_dx_dy_mm : tuple, (dx, dy), marker 间距
        - depth_size : int, (img_width, img_height), depth camera 像素尺寸
        - gel_size_mm : int, (width, height), mm 硅胶尺寸
        """
        self.view = view
        self.fric_coef = 0.4   # 摩擦系数
        self._mesh_smoother = None
        MF, KF, node, top_nodes, top_indices, self._top_vert_indices, self.mesh_shape = self.read_data(fem_data_path, raw_data)
        self.solver = FemSolver(MF, KF, nonlinear_coef)
        self.gel_size_mm = gel_size_mm  # mm

        self.cam_to_gel = self.get_cam_to_gel()        # 相机到gelpad的变换矩阵
        self.depth_to_gel = self.get_depth_to_gel(depth_size, gel_size_mm)

        self.marker_interp = MarkerInterp(marker_row_col, marker_dx_dy_mm[0], marker_dx_dy_mm[1], node[top_nodes])
        self.marker_offset = node[top_nodes][:, 1].mean()

        self.top_nodes = top_nodes.astype(np.uint32)            # gelpad 表层节点全局编号
        self.top_mesh_xyz = node[top_nodes].reshape(self.mesh_shape[0], self.mesh_shape[1], 3)  # gelpad 表层节点坐标
        self._top_normals = np.zeros((len(self.top_nodes), 3), np.float32)  # 表层节点法向量
        self._top_normals[:, 2] = 1

        self.P_gel_orig = node.astype(np.float32)            # 节点初始坐标
        self.P_gel_curr = node.copy()                       # 当前节点坐标
        self.P_obj = np.zeros((len(self.top_nodes), 3), np.float32)  # 只有表层节点的坐标

        self.node_num = self.P_gel_orig.shape[0]
        self.load_force = np.zeros((self.node_num, 3), np.float32)       # 节点力

        self.contact_state = ContactState(self.top_nodes)   # 接触状态
        self.depth = np.zeros((depth_size[1], depth_size[0]), np.float32)  # 上一次 step 使用的深度图

    @property
    def top_vert(self):
        return self.P_gel_curr[self.top_nodes]

    @property
    def top_indices(self):
        return self._top_vert_indices

    @property
    def top_force(self):
        return self.load_force[self.top_nodes]

    @property
    def contact_vert(self):
        return self.P_gel_curr[self.contact_state.contact_idx()[1]]

    @property
    def contact_force(self):
        return self.load_force[self.contact_state.contact_idx()[1]]

    @property
    def top_normals(self):
        return self._top_normals

    def read_data(self, fem_file, raw_data=None):
        if raw_data is None:
            if fem_file is None:
                raise ValueError("Either fem_file or raw_data must be provided")
            data = np.load(str(fem_file))
            # femsolver
            KF = SimpleCSR(data['KF_data'], data['KF_indices'], data['KF_indptr'], shape=data['KF_shape'])
            # MF = SimpleCSR(data['MF_data'], data['MF_indices'], data['MF_indptr'], shape=data['MF_shape'])
            MF = KF.copy()
        else:
            data = raw_data
            KF = SimpleCSR(data['KF_data'], data['KF_indices'], data['KF_indptr'], shape=data['KF_shape'])
            MF = KF.copy()

        # data
        c3d8_elements = data['elements']
        top_global_idx = data['top_nodes']  # (n_top_nodes,)  顶部节点在全局节点中的索引
        top_global_faces = data['top_indices']  # (n_top_faces, 4)  顶部面的全局节点索引
        top_local_faces = data['top_vert_indices']  # (n_top_faces, 4)  顶部面在顶部点内部的节点索引

        neighbors = make_neighbors(len(top_global_idx), top_local_faces)  # 用于平滑顶面节点数据的伞状数据
        self._mesh_smoother = get_smoother(neighbors)

        top_local_faces = np.stack(
            [top_local_faces[:, :3], top_local_faces[:, [0, 2, 3]]],
            axis = 1
        ).reshape(-1, 3).astype(dtype=np.int32)  # 转换成三角形索引

        node_xyz = data['node']  # (n_nodes, 3)  节点坐标
        node_y_min = np.min(node_xyz[:, 1])
        node_xyz[:, 1] -= node_y_min  # NOTE 将节点坐标 y 平移到 0, 方便后续处理
        mesh_shpae = data.get('mesh_shape', None)  # (行数, 列数), gelpad mesh 的形状, 如果没有则从文件名中解析
        if mesh_shpae is None:
            mesh_shpae = np.array([35, 20], np.int32)
        return MF, KF, node_xyz, top_global_idx, top_global_faces, top_local_faces, mesh_shpae

    def get_depth_to_gel(self, depth_map_size, gel_size_mm) -> Matrix4x4:
        """
        计算深度图像坐标系到 gelpad 坐标系的变换矩阵 A,  像素坐标 = A * P_gel
        """
        width, height = depth_map_size
        fx = width / gel_size_mm[0]
        fy = height / gel_size_mm[1]
        cam_to_gel = self.get_cam_to_gel()
        depth_to_cam = Matrix4x4([
            [fx, 0, 0, width / 2],
            [0, -fy, 0, height / 2],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        return depth_to_cam * cam_to_gel

    def get_cam_to_gel(self) -> Matrix4x4:
        """
        计算相机到 gelpad 的变换矩阵

        Returns:
        - Matrix4x4, xyz[mm], rpy[degree]
        """
        return Matrix4x4.fromVector6d(0, -self.gel_size_mm[1]/2, 0, 0, 180, 0)

    def get_obj_to_gel(self, base_to_obj: Matrix4x4, base_to_cam: Matrix4x4) -> Matrix4x4:
        """
        计算物体到 gelpad 的变换矩阵
        """
        obj_to_cam = base_to_obj.inverse() * base_to_cam
        obj_to_cam.moveto(*(1000 * obj_to_cam.xyz))  # m -> mm
        return obj_to_cam * self.cam_to_gel

    def reset(self):
        self.load_force[:] = 0
        self.P_gel_curr = self.P_gel_orig.copy()
        self.contact_state.flag[:] = False
        self._top_normals[:] = np.array([0, 0, 1])  # 表层节点法向量

    # def compute_wrench(self):
    #     '''
    #     反力反力矩计算
    #     '''
    #     contact_force = self.load_force.reshape((-1,3))[self.contact_nodes-1]
    #     vector_to_obj = np.array([-1,1,-1])*self.translation - np.array([0,0,self.dis_between_cam_gel*self.scale_unit]) - self.node_xyz_curr[self.contact_nodes-1]
    #     force = -np.sum(contact_force, axis = 0)
    #     torque = np.cross(vector_to_obj, contact_force) * 0.001    #N*mm -> N*m
    #     return force, torque

    def get_marker(self):
        return self.marker_interp.interp(self.P_gel_curr[self.top_nodes])
    
    def get_marker_displacement(self):
        return self.marker_interp.interp(self.P_gel_curr[self.top_nodes] - self.P_gel_orig[self.top_nodes])

    def get_mesh_data(self, enable_smooth=False):
        """
        返回绘图数据

        Parameters:
        - enable_smooth : bool, optional, default: False, 是否开启平滑

        Returns:
        - Tuple(vertices, normals), 表面 mesh 顶点坐标和法向量, 注意: 法向量是反的
        """
        if enable_smooth:
            top_vert = self._mesh_smoother(self.top_vert, 1)
            top_vert = self.fix_border(top_vert)  # NOTE: 固定边界点
            normal = compute_normals(top_vert, self.top_indices)
            normal = self._mesh_smoother(normal, 1)
            return top_vert, normal
        return self.top_vert, self._top_normals

    def get_depth(self) -> np.ndarray:
        """
        上一次 step 使用的深度图, mm
        """
        return self.depth

    def fix_border(self, vert):
        """
        固定边界点，避免边界点内移
        """
        vert = vert.reshape(self.mesh_shape[0], self.mesh_shape[1], 3)
        vert[[0,-1], :, 1] = self.top_mesh_xyz[[0,-1], :, 1]
        vert[:, [0,-1], 0] = self.top_mesh_xyz[:, [0,-1], 0]
        return vert.reshape(-1, 3)

    def _step_static(self, obj_to_gel: Matrix4x4, depth: np.ndarray):
        depth_map = depth * 1000  # m -> mm
        self.depth = depth_map

        if np.min(depth_map) > 1e-3:
            if np.any(self.contact_state.flag):
                self.reset()
            return

        cont_idx = self.contact_state.contact_idx()
        no_cont_idx = self.contact_state.no_contact_idx()

        # 计算接触顶面点对应的深度, 检查是否有脱离
        if cont_idx.shape[1] > 0:
            p_obj = self.P_obj[cont_idx[0]]  # 上一帧接触点在物体坐标系下的坐标
            p_gel = obj_to_gel.inverse() * p_obj  # 上一帧接触点物体坐标系 -> gelpad 坐标系
            top_force = self.load_force[cont_idx[1]]

            # 计算接触力与法向量的点积
            top_force_dot_normal = np.sum(top_force * self._top_normals[cont_idx[0]], axis=1)
            detached_mask = top_force_dot_normal >= 0

            self.contact_state.set(cont_idx[0, detached_mask], False)  # 更新脱离的节点的接触状态
            self.P_gel_curr[cont_idx[1, ~detached_mask]] = p_gel[~detached_mask]  # 更新保持接触的节点坐标

        # 计算未接触顶面点对应的深度, 检查是否有新接触
        if no_cont_idx.shape[1] > 0:
            p_gel = self.P_gel_curr[no_cont_idx[1]]
            p_pix = (self.depth_to_gel * p_gel).astype(np.int32)
            p_pix = np.clip(p_pix[:, :2], (0, 0), np.array([depth_map.shape[1]-1, depth_map.shape[0]-1]))
            z_gel = depth_map[p_pix[:, 1], p_pix[:, 0]].reshape(-1)
            attached_mask = z_gel < p_gel[:, 2]
            attached_idx = no_cont_idx[:, attached_mask]

            # 更新新接触点的 flag 和 P_obj
            if attached_idx.shape[1] > 0:
                self.contact_state.set(attached_idx[0], True)
                p_gel = np.hstack((p_gel[attached_mask, :2], z_gel[attached_mask, None]))

                self.P_obj[attached_idx[0]] = obj_to_gel * p_gel
                self.P_gel_curr[attached_idx[1]] = p_gel

        U, self.load_force = self.solver.solve_fric(
            constrain_idx = self.contact_state.contact_idx()[1],
            node_disp = self.P_gel_curr - self.P_gel_orig,
            P = None
        )
        self.P_gel_curr = self.P_gel_orig + U   # 更新当前时刻节点坐标
        self._top_normals = compute_normals(self.top_vert, self.top_indices)
        return self.P_gel_curr

    def _step_fric(self, obj_to_gel: Matrix4x4, depth: np.ndarray):
        #NOTE: 未完成

        # depth = (depth - 0.005) /5*2
        depth *= 0.4

        depth_map = depth * 1000  # m -> mm
        self.depth = depth_map
        # obj_to_gel = obj_to_gel
        gel_to_obj = obj_to_gel.inverse()
        if np.min(depth_map) > 1e-3:
            if np.any(self.contact_state.flag):
                self.reset()
            return

        # ====== 1: 按照粘滞点无滑动求解节点位移 U 和接触力 self.local_force
        cont_idx = self.contact_state.contact_idx()
        # 计算接触顶面点对应的深度, 检查是否有脱离
        if cont_idx.shape[1] > 0:
            # 计算接触力与法向量的点积, 判断是否脱离
            cont_force = self.load_force[cont_idx[1]]
            cont_norm = self._top_normals[cont_idx[0]]
            cont_force_norm_length = np.einsum("ij,ij->i", cont_force, cont_norm)  # shape=(ncont,)
            detached_mask = cont_force_norm_length >= 0  # 脱离点
            self.contact_state.set(cont_idx[0, detached_mask], 0)  # 更新脱离的节点的接触状态

            # 用刚体变换更新保持接触的节点坐标
            cont_idx = self.contact_state.contact_idx()
            p_obj = self.P_obj[cont_idx[0]]  # 上一帧接触点在物体坐标系下的坐标
            p_gel = gel_to_obj * p_obj  # 上一帧接触点物体坐标系 -> gelpad 坐标系
            self.P_gel_curr[cont_idx[1]] = p_gel

        # 计算未接触顶面点对应的深度, 检查是否有新接触
        no_cont_idx = self.contact_state.no_contact_idx()
        if no_cont_idx.shape[1] > 0:
            p_gel = self.P_gel_curr[no_cont_idx[1]]
            p_pix = (self.depth_to_gel * p_gel).astype(np.int32)
            p_pix = np.clip(p_pix[:, :2], (0, 0), np.array([depth_map.shape[1]-1, depth_map.shape[0]-1]))
            z_gel = depth_map[p_pix[:, 1], p_pix[:, 0]].reshape(-1)
            attached_mask = z_gel < p_gel[:, 2]
            attached_idx = no_cont_idx[:, attached_mask]

            # 更新新接触点的 flag 和 P_obj
            if attached_idx.shape[1] > 0:
                self.contact_state.set(attached_idx[0], 1)
                p_gel = np.hstack((p_gel[attached_mask, :2], z_gel[attached_mask, None]))
                self.P_obj[attached_idx[0]] = obj_to_gel * p_gel
                self.P_gel_curr[attached_idx[1], 2] = p_gel[:, 2]

        # 求解
        U, self.load_force = self.solver.solve_fric(
            constrain_idx = self.contact_state.contact_idx()[1],
            node_disp = self.P_gel_curr - self.P_gel_orig,
            P = None
        )
        self.P_gel_curr = self.P_gel_orig + U   # 更新当前时刻节点坐标
        self._top_normals = compute_normals(self.top_vert, self.top_indices)

        # ====== 2: 检查粘滞点中的滑移点, 约束滑移点的力, 和粘滞点的位移
        # 初始化摩擦力约束
        constrain_F = np.zeros_like(self.P_gel_curr)

        # 检查粘滞点的滑动状态
        cont_idx = self.contact_state.contact_idx()
        slide_num = 0
        slide_idx = None
        if cont_idx.shape[1] > 0:
            # 计算最大摩擦力和当前摩擦力
            cont_force = self.load_force[cont_idx[1]]
            cont_norm = self._top_normals[cont_idx[0]]
            cont_force_norm_len = np.einsum("ij,ij->i", cont_force, cont_norm)  # shape=(ncont,)  < 0 为粘滞
            cont_force_norm = np.einsum("i, ij->ij", cont_force_norm_len, cont_norm)
            cont_force_tangent = cont_force - cont_force_norm
            cont_force_tangent_len = np.sqrt(np.einsum("ij,ij->i", cont_force_tangent, cont_force_tangent))
            max_tangent_len = -cont_force_norm_len * self.fric_coef

            detached_mask = cont_force_norm_len >= 0
            slide_mask = (cont_force_tangent_len > max_tangent_len) & (~detached_mask)
            slide_idx = cont_idx[:, slide_mask]
            slide_num = slide_idx.shape[1]

            # self.view.vis_force.setData(self.contact_vert, self.contact_vert + cont_force_tangent*10,
            #                 color = np.zeros_like(self.contact_vert)+((0.7,0.2,0)))
            # self.view.vis_force.addData(self.contact_vert, self.contact_vert + cont_force_norm*10,
            #                             color = np.zeros_like(self.contact_vert)+((0,1,1)))

            self.contact_state.set(cont_idx[0, detached_mask], 0)  # 更新脱离的节点的接触状态

        if slide_num:
            constrain_F[slide_idx[1]] = cont_force_norm[slide_mask] + \
                cont_force_tangent[slide_mask] * (max_tangent_len[slide_mask] / cont_force_tangent_len[slide_mask])[:, None]

            constrain_idx = cont_idx[1, ~(slide_mask | detached_mask)]
            U, self.load_force = self.solver.solve_fric(constrain_idx, U, constrain_F)
            self.P_gel_curr = self.P_gel_orig + U   # 更新当前时刻节点坐标


        # ====== 3： 检查 slide 和 no_contact 中的新接触点, 约束位移
        calc_idx = self.contact_state.no_contact_idx()  # 不包含 slide 的 no_contact
        if slide_num:
            calc_idx = np.concatenate((slide_idx, calc_idx), axis=1)

        if calc_idx.shape[1] > 0:
            attached_mask = np.zeros(calc_idx.shape[1], dtype=np.bool_)
            p_gel = self.P_gel_curr[calc_idx[1]]
            p_pix = (self.depth_to_gel * p_gel).astype(np.int32)
            p_pix = np.clip(p_pix[:, :2], (0, 0), np.array([depth_map.shape[1]-1, depth_map.shape[0]-1]))
            z_gel = depth_map[p_pix[:, 1], p_pix[:, 0]].reshape(-1)
            attached_mask[:slide_num] = z_gel[:slide_num] < p_gel[:slide_num, 2] + 0.2
            attached_mask[slide_num:] = z_gel[slide_num:] < p_gel[slide_num:, 2]
            attached_idx = calc_idx[:, attached_mask]
            detached_idx = calc_idx[:, ~attached_mask]
            self.contact_state.set(attached_idx[0], 1)
            self.contact_state.set(detached_idx[0], 0)

            # 更新新接触点的 flag 和 P_obj
            if attached_idx.shape[1] > 0:
                p_gel = np.hstack((p_gel[attached_mask, :2], z_gel[attached_mask, None]))
                self.P_obj[attached_idx[0]] = obj_to_gel * p_gel
                self.P_gel_curr[attached_idx[1], 2] = p_gel[:, 2]

        # 求解
        U, self.load_force = self.solver.solve_fric(
            constrain_idx = self.contact_state.contact_idx()[1],
            node_disp = self.P_gel_curr - self.P_gel_orig,
            P = None
        )
        self.P_gel_curr = self.P_gel_orig + U   # 更新当前时刻节点坐标
        self._top_normals = compute_normals(self.top_vert, self.top_indices)

        return self.P_gel_curr

    def step(self, obj_pose: Matrix4x4, sensor_pose: Matrix4x4, depth: np.ndarray):
        obj_to_gel = self.get_obj_to_gel(obj_pose, sensor_pose)
        return self._step_fric(obj_to_gel, depth)
# ===== END FILE: xengym/fem/simulation.py =====
