import numpy as np
from xensesdk.ezgl.items import RGBCamera, gl, Shader, GLMeshItem, Texture2D


def gen_texcoords(n_row, n_col, u_range=(0, 1), v_range=(0, 1)):
    tex_u = np.linspace(*u_range, n_col)
    tex_v = np.linspace(*v_range, n_row)
    return np.stack(np.meshgrid(tex_u, tex_v), axis=-1).reshape(-1, 2)


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 mvp;

void main() {
    TexCoords = aTexCoords;
    gl_Position = mvp * vec4(aPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D marker_tex;

void main() {
    vec3 result = vec3(texture(marker_tex, TexCoords));
    FragColor = vec4(result, 1);
}
"""

class MarkerTextureCamera(RGBCamera):

    def init(self, mesh: GLMeshItem, marker_tex: Texture2D):
        self.mesh = mesh
        self.marker_tex = marker_tex
        with self.view():
            self.init_fbo()
            self.shader = Shader(vertex_shader, fragment_shader)
            self._fbo.rgb_texture.type = "tex_diffuse"

    @property
    def texture(self):
        return self._fbo.rgb_texture

    def render(self):
        if self.view() is None:
            raise ValueError("view is None")

        with self.view():
            with self._fbo:
                if not self.mesh.isInitialized:
                    self.mesh.initialize()

                gl.glDepthFunc(gl.GL_ALWAYS)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT )
                self.paint_mesh()
                gl.glDepthFunc(gl.GL_LESS)

    def get_texture_np(self):
        with self.view():
            return self._fbo.rgb_texture.getTexture()

    def paint_mesh(self):
        self.mesh.setupGLState()
        self.mesh.update_model_matrix()
        
        with self.shader:
            mvp = self.get_proj_view_matrix() * self.mesh.model_matrix()
            self.shader.set_uniform("mvp", mvp.glData, "mat4")
            self.shader.set_uniform("marker_tex", self.marker_tex.bindTexUnit(), "sampler2D")
            self.mesh._mesh.paintShadow()
            