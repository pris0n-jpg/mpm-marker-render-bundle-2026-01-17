# deepresearch bundle: mpm marker render

这个文件夹用于让 deepresearch 类 AI **只读关键代码**，快速定位：
- C(sim_mesh) 偏暗/对比差（SimMeshItem shader / bg+scale+marker multiply）
- marker appearance（random_ellipses attenuation texture）
- C 渲染管线（triplet runner / render adapter / sensor_scene 对照）
- FEM/MPM 中 advect_points（剪切导致点拉成椭圆的实现逻辑）

## Included files
- `mpm_compute_merged.py` (MPM 计算核心模块合并版：solver/contact/surface_mesh 等；便于一次性上传给 AI 阅读)
- `xengym/marker_appearance.py` (random_ellipses 生成; attenuation texture 语义)
- `example/mpm_xensim_triplet_runner.py` (A/B/C 生成 + motion_diagnostics pseudo-hole)
- `example/mpm_xensim_render_adapter.py` (C: sim_mesh renderer + sensor_scene renderer)
- `example/mpm_fem_rgb_compare.py` (MPMSensorScene: advect_points 椭圆/剪切)
- `example/mpm_xensim_baseline_lockfile.json` (复现实验输入)
- `xensim/xensim/core/sim_mesh_item.py` (SimMeshItem shader: bg + marker multiply + scale)
- `xensim/xensim/core/utils.py` (MarkerTextureCamera)

## Notes
- 这是 **copy**（快照），不修改原始文件。
- 输出产物（output/）不包含在这里。
