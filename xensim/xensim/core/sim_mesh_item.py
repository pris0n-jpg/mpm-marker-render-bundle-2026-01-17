import cv2
import numpy as np
import OpenGL.GL as gl
from ctypes import c_void_p

from ezgl.GLGraphicsItem import GLGraphicsItem
from ezgl.items import GLMeshItem, Mesh, Shader, Camera, GLDataBlock, VAO, VBO, EBO, Texture2D

from ezgl.experimental import compute_normals
from ezgl.experimental.GLEllipseItem import surface_indices, surface_vertexes


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aXY;

out vec2 grad_uv;
out vec2 xy_tex;

uniform mat4 mvp;

void main() {
    const float PI = 3.141592653589793;

    gl_Position = mvp * vec4(aPos, 1.0);
    xy_tex = aXY;

    // aNormal 是归一化的表面法向 vec3
    float u = acos(aNormal.x);  // 极角，范围 [0, PI]
    float v = atan(aNormal.z, aNormal.y);  // 方位角，范围 [-PI, PI], 但正常都是 >0 的

    grad_uv = vec2(u / PI, v / PI);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;
in vec2 grad_uv;
in vec2 xy_tex;
uniform bool bg_flag;
uniform float scale;
uniform sampler2D grad_texture_0;
uniform sampler2D grad_texture_1;
uniform sampler2D grad_texture_2;
uniform sampler2D grad_texture_3;
uniform sampler2D grad_texture_4;
uniform sampler2D grad_texture_5;
uniform sampler2D grad_texture_6;
uniform sampler2D grad_texture_7;
uniform sampler2D bg_texture;
uniform sampler2D marker_texture;
uniform bool marker_flag;

void main() {
    float x = xy_tex.x;
    float y = xy_tex.y;
    float xx = xy_tex.x * xy_tex.x;
    float yy = xy_tex.y * xy_tex.y;
    float xy = xy_tex.x * xy_tex.y;
    vec3 p_0 = vec3(xx, yy, xy);
    vec3 p_1 = vec3(x, y, 1);
    vec3 r_0 = texture(grad_texture_0, grad_uv).rgb;
    vec3 r_1 = texture(grad_texture_1, grad_uv).rgb;
    vec3 g_0 = texture(grad_texture_2, grad_uv).rgb;
    vec3 g_1 = texture(grad_texture_3, grad_uv).rgb;
    vec3 b_0 = texture(grad_texture_4, grad_uv).rgb;
    vec3 b_1 = texture(grad_texture_5, grad_uv).rgb;
    vec3 a_0 = texture(grad_texture_6, grad_uv).rgb;
    vec3 a_1 = texture(grad_texture_7, grad_uv).rgb;

    vec4 bg_color;
    if (bg_flag) {
        bg_color = texture(bg_texture, xy_tex).rgba;
    } else {
        bg_color = vec4(0.5, 0.5, 0.5, 0.5);
    }
    
    if (marker_flag) {
        bg_color = bg_color * texture(marker_texture, vec2(x, 1-y)).rgba;
    }
    
    FragColor = vec4(
        clamp((dot(p_0, r_0) + dot(p_1, r_1)) * scale + bg_color.r, 0.0, 1.0),
        clamp((dot(p_0, g_0) + dot(p_1, g_1)) * scale + bg_color.g, 0.0, 1.0),
        clamp((dot(p_0, b_0) + dot(p_1, b_1)) * scale + bg_color.b, 0.0, 1.0),
        clamp((dot(p_0, a_0) + dot(p_1, a_1)) * scale + bg_color.a, 0.0, 1.0)
    );
}
"""



class SimMeshItem(GLGraphicsItem):

    def __init__(
        self,
        shape: tuple,
        x_range: tuple,
        y_range: tuple,
        table_path: str = None,
        zmap: np.ndarray=None,
        color_scale: float=1.,
        glOptions="opaque",
        parentItem=None
    ):
        """
        从深度图生成 Mesh 网格

        Parameters:
        - shape : tuple(nrow, ncol), 顶点行数, 列数
        - x_range : tuple, w 方向顶点坐标范围
        - y_range : tuple, h 方向顶点坐标范围
        - zmap : np.ndarray, 单通道深度图
        - glOptions : str, optional, default: "opaque"
        - parentItem : GLGraphicsItem, optional, default: None
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._table_path = table_path
        self._nrow, self._ncol = shape
        self._x_range = x_range
        self._y_range = y_range
        self._vertices_init = surface_vertexes(np.zeros((10, 10), dtype=np.float32), self._x_range, self._y_range, self._nrow, self._ncol)

        self.gl_vertices = GLDataBlock(np.float32, 3, self._vertices_init)
        self.gl_indices = GLDataBlock(np.uint32, 3, surface_indices(self._nrow, self._ncol))
        
        normals = np.zeros_like(self._vertices_init)
        normals[:, 2] = 1
        self.gl_normals = GLDataBlock(np.float32, 3, normals)
        
        xf = np.linspace(0, 1, self._ncol, dtype='f4')
        yf = np.linspace(0, 1, self._nrow, dtype='f4')
        xfyf = np.stack(np.meshgrid(xf, yf, indexing='xy'), axis=-1)

        self.gl_xfyf_tex = GLDataBlock(np.float32, 2, xfyf)
        
        self.vbo = None
        self.ebo = None
        self.textures = {}
        self.marker_texture = None
        
        # Options
        self.marker_flag = False
        self.background_flag = True
        self.color_scale = color_scale

        self.setData(zmap)
    
        
    def setData(self, zmap=None, vert_smooth=5, norm_smooth=15):
        """
        设置深度图

        Parameters:
        - zmap : np.ndarray, default: None, 单通道深度图
        - smooth : int, optional, default: 1, 平滑次数
        """
        if zmap is None:
            return

        zmap = cv2.resize(zmap, (self._ncol, self._nrow), interpolation=cv2.INTER_LINEAR)
        if vert_smooth > 1:
            zmap = cv2.GaussianBlur(zmap, (vert_smooth, vert_smooth), 0)
        vertices = self._vertices_init.copy()
        vertices[:, 2] = zmap.reshape(-1)
        new_normals = compute_normals(vertices, self.gl_indices.data)

        if norm_smooth > 1:
            new_normals = cv2.GaussianBlur(new_normals.reshape((self._nrow, self._ncol, 3)), (norm_smooth, norm_smooth), 0)
            new_normals = new_normals.reshape((-1, 3))

        self.gl_vertices.set_data(vertices)
        self.gl_normals.set_data(new_normals)
    
    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        if self.vbo is not None:
            return

        with VAO() as self.vao:
            self.vbo = VBO(
                [self.gl_vertices, self.gl_normals, self.gl_xfyf_tex],
                expandable = True,
                usage = gl.GL_STATIC_DRAW
            )
            self.vbo.setAttrPointer([0, 1, 2], attr_id=[0, 1, 2])
            self.ebo = EBO(self.gl_indices)
            self.load_textures(str(self._table_path))

    def load_textures(self, data_path: str):
        """load textures"""
        data = np.load(data_path)
        weights = data["weights"]
        bg_image = data["ref"]
        for i in range(weights.shape[2] // 3):
            self.textures[f"grad_texture_{i}"] = Texture2D(weights[:, :, i*3:(i+1)*3].transpose((1, 0, 2)), flip_y=False, 
                                                           wrap_s=gl.GL_CLAMP_TO_EDGE, wrap_t=gl.GL_CLAMP_TO_EDGE)
        self.textures["bg_texture"] = Texture2D(bg_image, flip_y=False, wrap_s=gl.GL_CLAMP_TO_EDGE, wrap_t=gl.GL_CLAMP_TO_EDGE)
    
    def set_textures(self):
        """set texture to uniforms"""
        for name, tex in self.textures.items():
            self.shader.set_uniform(name, tex.bindTexUnit(), "sampler2D")
        self.shader.set_uniform("bg_texture", self.textures["bg_texture"].bindTexUnit(), "sampler2D")

        # marker render
        if self.marker_flag and self.marker_texture is not None:
            self.shader.set_uniform("marker_flag", True, "bool")
            self.shader.set_uniform("marker_texture", self.marker_texture.bindTexUnit(), "sampler2D")
        else:
            self.shader.set_uniform("marker_flag", False, "bool")

    def updateGL(self):
        self.vbo.commit()
        self.ebo.commit()
        
    def paint(self, camera: Camera):
        self.updateGL()
        self.setupGLState()
        
        with self.shader:
            mvp = camera.get_proj_view_matrix() * self.model_matrix()
            self.shader.set_uniform("mvp", mvp.glData, "mat4")
            self.shader.set_uniform("bg_flag", self.background_flag, "bool")
            self.shader.set_uniform("scale", self.color_scale, "float")
            self.set_textures()
    
            with self.vao:
                if self.ebo.size() > 0:
                    gl.glDrawElements(gl.GL_TRIANGLES, self.ebo.size(), gl.GL_UNSIGNED_INT, c_void_p(0))
                else:
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vbo.count(0))
