# Deepresearch bundle (MPM compute merged)
#
# 说明：为了减少文件数量（便于一次性上传/投喂给 deepresearch AI），把 xenmpm 里的 MPM 计算核心模块合并到单文件。
# 注意：该文件主要用于代码阅读/分析，不保证可直接运行。
# 源：F:/workspace/xenmpm
#


# ===== BEGIN FILE: xengym/mpm/__init__.py =====
"""
MPM (Material Point Method) Module
Provides VHE-MLS-MPM solver with automatic differentiation support

Autodiff support:
- DifferentiableMPMSolver: Blocked due to Taichi AD limitation on atomic ops
- ManualAdjointMPMSolver: Manual adjoint implementation that bypasses Taichi AD limitation
"""
from .config import (
    MPMConfig,
    GridConfig,
    TimeConfig,
    OgdenConfig,
    MaxwellBranchConfig,
    MaterialConfig,
    SDFConfig,
    ContactConfig,
    OutputConfig
)
from .fields import MPMFields
from .mpm_solver import MPMSolver
from .surface_mesh import TopSurfaceMeshExporter, TopSurfaceMeshFrame
from .autodiff_wrapper import DifferentiableMPMSolver
from .manual_adjoint_solver import ManualAdjointMPMSolver
from .manual_adjoint import (
    ManualAdjointFields,
    bspline_weight,
    bspline_weight_gradient,
    grid_ops_backward_kernel,
    p2g_backward_kernel,
    g2p_backward_kernel,
    update_F_backward_kernel,
    maxwell_backward_kernel,
    maxwell_G_gradient_kernel,
    bulk_viscosity_gradient_kernel,
    position_loss_backward_kernel,
    velocity_loss_backward_kernel,
    kinetic_energy_loss_backward_kernel,
    total_energy_loss_backward_kernel,
)
from .constitutive import (
    compute_ogden_stress_2terms,
    compute_ogden_stress_general,
    compute_maxwell_stress,
    compute_bulk_viscosity_stress,
    compute_maxwell_stress_no_update,
    compute_bulk_viscosity_stress_no_energy,
)
from .constitutive_gradients import (
    configure_gradient_mode,
    validate_gradient_mode,
    is_experimental_mode_enabled,
    get_scale_guards,
    compute_p_total_with_gradients,
    compute_p_total_for_diff,
    compute_g_F_numerical_p_total,
    compute_bulk_viscosity_gradient,
)
from .contact import (
    compute_contact_force,
    sdf_sphere,
    sdf_plane,
    sdf_box
)
from .decomp import (
    polar_decompose,
    safe_svd,
    eig_sym_3x3,
    make_spd,
    make_spd_ste,
    clamp_J
)
from .stability import (
    check_ogden_drucker_stability,
    check_timestep_constraints,
    validate_config
)
from .exceptions import (
    MPMError,
    ConfigurationError,
    MaterialError,
    StabilityError,
    AutodiffError,
    GradientError,
    ScaleGuardError,
    TargetNotSetError,
)

__all__ = [
    # Config
    'MPMConfig',
    'GridConfig',
    'TimeConfig',
    'OgdenConfig',
    'MaxwellBranchConfig',
    'MaterialConfig',
    'SDFConfig',
    'ContactConfig',
    'OutputConfig',
    # Core
    'MPMFields',
    'MPMSolver',
    'TopSurfaceMeshExporter',
    'TopSurfaceMeshFrame',
    'DifferentiableMPMSolver',
    # Manual Adjoint (new)
    'ManualAdjointMPMSolver',
    'ManualAdjointFields',
    'bspline_weight',
    'bspline_weight_gradient',
    'grid_ops_backward_kernel',
    'p2g_backward_kernel',
    'g2p_backward_kernel',
    'update_F_backward_kernel',
    'maxwell_backward_kernel',
    'maxwell_G_gradient_kernel',
    'bulk_viscosity_gradient_kernel',
    'position_loss_backward_kernel',
    'velocity_loss_backward_kernel',
    'kinetic_energy_loss_backward_kernel',
    'total_energy_loss_backward_kernel',
    # Constitutive
    'compute_ogden_stress_2terms',
    'compute_ogden_stress_general',
    'compute_maxwell_stress',
    'compute_bulk_viscosity_stress',
    'compute_maxwell_stress_no_update',
    'compute_bulk_viscosity_stress_no_energy',
    # Gradient Configuration
    'configure_gradient_mode',
    'validate_gradient_mode',
    'is_experimental_mode_enabled',
    'get_scale_guards',
    'compute_p_total_with_gradients',
    'compute_p_total_for_diff',
    'compute_g_F_numerical_p_total',
    'compute_bulk_viscosity_gradient',
    # Contact
    'compute_contact_force',
    'sdf_sphere',
    'sdf_plane',
    'sdf_box',
    # Decomp
    'polar_decompose',
    'safe_svd',
    'eig_sym_3x3',
    'make_spd',
    'make_spd_ste',
    'clamp_J',
    # Stability
    'check_ogden_drucker_stability',
    'check_timestep_constraints',
    'validate_config',
    # Exceptions
    'MPMError',
    'ConfigurationError',
    'MaterialError',
    'StabilityError',
    'AutodiffError',
    'GradientError',
    'ScaleGuardError',
    'TargetNotSetError',
]
# ===== END FILE: xengym/mpm/__init__.py =====


# ===== BEGIN FILE: xengym/mpm/config.py =====
"""
MPM Solver Configuration
Provides dataclass-based configuration for grid, time stepping, materials, contact, and output options.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json
import yaml
from pathlib import Path


@dataclass
class GridConfig:
    """Grid configuration for MPM solver"""
    grid_size: Tuple[int, int, int] = (64, 64, 64)  # Grid resolution
    dx: float = 0.01  # Grid spacing in meters
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Grid origin

    # Legacy boundary condition: clamp grid velocities near domain boundaries.
    # Defaults preserve previous hardcoded behavior in MPMSolver.grid_op.
    sticky_boundary: bool = True
    sticky_boundary_width: int = 3  # Width in grid cells


@dataclass
class TimeConfig:
    """Time stepping configuration"""
    dt: float = 1e-4  # Time step size
    num_steps: int = 1000  # Number of simulation steps
    substeps: int = 1  # Number of substeps per step


@dataclass
class OgdenConfig:
    """Ogden hyperelastic model parameters"""
    mu: List[float] = field(default_factory=lambda: [1e5, 1e4])  # Shear moduli
    alpha: List[float] = field(default_factory=lambda: [2.0, -2.0])  # Exponents
    kappa: float = 1e6  # Bulk modulus for volumetric response


@dataclass
class MaxwellBranchConfig:
    """Single Maxwell branch configuration"""
    G: float = 1e4  # Shear modulus
    tau: float = 0.1  # Relaxation time


@dataclass
class MaterialConfig:
    """Material configuration"""
    density: float = 1000.0  # kg/m^3

    # Ogden hyperelastic parameters
    ogden: OgdenConfig = field(default_factory=OgdenConfig)

    # Maxwell viscoelastic branches
    maxwell_branches: List[MaxwellBranchConfig] = field(default_factory=list)

    # Optional Kelvin-Voigt bulk viscosity
    enable_bulk_viscosity: bool = False
    bulk_viscosity: float = 0.0  # Pa·s


@dataclass
class SDFConfig:
    """SDF obstacle configuration

    Supported types:
    - 'plane': Infinite plane defined by point and normal
    - 'sphere': Sphere defined by center and radius
    - 'box': Axis-aligned box defined by center and half_extents
    - 'cylinder': Capped cylinder aligned with Z axis, defined by center, radius and half_height

    Parameters (as tuples):
    - plane: point=(x, y, z), normal=(nx, ny, nz)
    - sphere: center=(x, y, z), radius=r (stored as half_extents=(r, 0, 0))
    - box: center=(x, y, z), half_extents=(hx, hy, hz)
    - cylinder: center=(x, y, z), radius=r, half_height=h (stored as half_extents=(r, r, h))
    """
    sdf_type: str = 'plane'  # 'plane', 'sphere', 'box', 'cylinder'
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center/point on plane
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Normal for plane (unit vector)
    half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Half extents for box, or (radius, 0, 0) for sphere


@dataclass
class ContactConfig:
    """Contact and friction configuration"""
    enable_contact: bool = True
    contact_stiffness_normal: float = 1e5  # Normal penalty stiffness
    contact_stiffness_tangent: float = 1e4  # Tangential spring stiffness

    # Friction parameters
    mu_s: float = 0.5  # Static friction coefficient
    mu_k: float = 0.3  # Kinetic friction coefficient
    friction_transition_vel: float = 1e-3  # Velocity for tanh transition

    # Contact cleanup parameters
    K_clear: int = 10  # Hysteresis counter threshold for clearing u_t

    # SDF obstacles (default: ground plane at z=0)
    # If empty, uses default ground plane for backward compatibility
    obstacles: List[SDFConfig] = field(default_factory=list)

    # Backward compatibility
    @property
    def contact_stiffness(self) -> float:
        """Deprecated: use contact_stiffness_normal instead"""
        return self.contact_stiffness_normal


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "output"
    save_particles: bool = True
    save_energy: bool = True
    save_contact_data: bool = True
    output_interval: int = 10  # Save every N steps


@dataclass
class MPMConfig:
    """Complete MPM solver configuration"""
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MPMConfig':
        """Load configuration from dictionary"""
        grid_dict = config_dict.get('grid', {})
        # Convert lists to tuples for grid_size and origin
        if 'grid_size' in grid_dict and isinstance(grid_dict['grid_size'], list):
            grid_dict['grid_size'] = tuple(grid_dict['grid_size'])
        if 'origin' in grid_dict and isinstance(grid_dict['origin'], list):
            grid_dict['origin'] = tuple(grid_dict['origin'])
        grid = GridConfig(**grid_dict)
        time = TimeConfig(**config_dict.get('time', {}))

        # Parse material config
        mat_dict = config_dict.get('material', {})
        ogden = OgdenConfig(**mat_dict.get('ogden', {}))
        maxwell_branches = [
            MaxwellBranchConfig(**b) for b in mat_dict.get('maxwell_branches', [])
        ]
        material = MaterialConfig(
            density=mat_dict.get('density', 1000.0),
            ogden=ogden,
            maxwell_branches=maxwell_branches,
            enable_bulk_viscosity=mat_dict.get('enable_bulk_viscosity', False),
            bulk_viscosity=mat_dict.get('bulk_viscosity', 0.0)
        )

        # Parse contact config with backward compatibility
        contact_dict = config_dict.get('contact', {})
        # Handle old 'contact_stiffness' parameter
        if 'contact_stiffness' in contact_dict and 'contact_stiffness_normal' not in contact_dict:
            contact_dict['contact_stiffness_normal'] = contact_dict.pop('contact_stiffness')
            contact_dict['contact_stiffness_tangent'] = contact_dict['contact_stiffness_normal'] * 0.1

        # Parse SDF obstacles
        obstacles_list = contact_dict.pop('obstacles', [])
        obstacles = [
            SDFConfig(
                sdf_type=o.get('sdf_type', 'plane'),
                center=tuple(o.get('center', [0.0, 0.0, 0.0])),
                normal=tuple(o.get('normal', [0.0, 0.0, 1.0])),
                half_extents=tuple(o.get('half_extents', [0.0, 0.0, 0.0]))
            ) for o in obstacles_list
        ]
        contact = ContactConfig(**contact_dict, obstacles=obstacles)
        output = OutputConfig(**config_dict.get('output', {}))

        return cls(
            grid=grid,
            time=time,
            material=material,
            contact=contact,
            output=output
        )

    @classmethod
    def from_json(cls, json_path: str) -> 'MPMConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MPMConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'grid': {
                'grid_size': list(self.grid.grid_size),
                'dx': self.grid.dx,
                'origin': list(self.grid.origin),
                'sticky_boundary': bool(self.grid.sticky_boundary),
                'sticky_boundary_width': int(self.grid.sticky_boundary_width),
            },
            'time': {
                'dt': self.time.dt,
                'num_steps': self.time.num_steps,
                'substeps': self.time.substeps
            },
            'material': {
                'density': self.material.density,
                'ogden': {
                    'mu': self.material.ogden.mu,
                    'alpha': self.material.ogden.alpha,
                    'kappa': self.material.ogden.kappa
                },
                'maxwell_branches': [
                    {'G': b.G, 'tau': b.tau} for b in self.material.maxwell_branches
                ],
                'enable_bulk_viscosity': self.material.enable_bulk_viscosity,
                'bulk_viscosity': self.material.bulk_viscosity
            },
            'contact': {
                'enable_contact': self.contact.enable_contact,
                'contact_stiffness_normal': self.contact.contact_stiffness_normal,
                'contact_stiffness_tangent': self.contact.contact_stiffness_tangent,
                'mu_s': self.contact.mu_s,
                'mu_k': self.contact.mu_k,
                'friction_transition_vel': self.contact.friction_transition_vel,
                'K_clear': self.contact.K_clear,
                'obstacles': [
                    {
                        'sdf_type': o.sdf_type,
                        'center': list(o.center),
                        'normal': list(o.normal),
                        'half_extents': list(o.half_extents)
                    } for o in self.contact.obstacles
                ]
            },
            'output': {
                'output_dir': self.output.output_dir,
                'save_particles': self.output.save_particles,
                'save_energy': self.output.save_energy,
                'save_contact_data': self.output.save_contact_data,
                'output_interval': self.output.output_interval
            }
        }

    def save_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
# ===== END FILE: xengym/mpm/config.py =====


# ===== BEGIN FILE: xengym/mpm/exceptions.py =====
"""
MPM Solver Custom Exceptions

Provides a structured exception hierarchy for better error handling and debugging.
All MPM-specific exceptions inherit from MPMError for easy catching.
"""


class MPMError(RuntimeError):
    """Base exception for all MPM solver errors.

    Catch this to handle any MPM-related error generically.
    """
    pass


class ConfigurationError(MPMError):
    """Configuration or user input validation error.

    Raised when:
    - Invalid configuration values are provided
    - Required configuration is missing
    - Configuration constraints are violated
    """
    pass


class MaterialError(ConfigurationError):
    """Material parameter validation error.

    Raised when:
    - Ogden parameters violate Drucker stability
    - Maxwell branch parameters are invalid
    - Material model constraints are violated
    """
    pass


class StabilityError(ConfigurationError):
    """Numerical stability constraint violation.

    Raised when:
    - Time step exceeds CFL condition
    - Deformation gradient becomes singular
    - Energy becomes unbounded
    """
    pass


class AutodiffError(MPMError):
    """Automatic differentiation limitation or error.

    Raised when:
    - Taichi AD limitations are encountered
    - Gradient computation is not supported for a configuration
    """
    pass


class GradientError(AutodiffError):
    """Gradient computation error.

    Raised when:
    - Manual adjoint computation fails
    - Numerical gradient verification fails
    - Gradient mode configuration is invalid
    """
    pass


class ScaleGuardError(GradientError):
    """Scale guard violation in experimental gradient mode.

    Raised when:
    - Particle count exceeds experimental mode limit
    - Step count exceeds experimental mode limit
    """
    pass


class TargetNotSetError(GradientError):
    """Target data not set for loss computation.

    Raised when:
    - Target positions not set for position loss
    - Target velocities not set for velocity loss
    - Target energy not set for energy loss
    """
    pass
# ===== END FILE: xengym/mpm/exceptions.py =====


# ===== BEGIN FILE: xengym/mpm/fields.py =====
"""
Field Management for MPM Solver
Defines particle fields, grid fields, and global energy scalars.
Supports automatic differentiation via enable_grad parameter.
"""
import taichi as ti
from typing import Tuple
from .config import MPMConfig


@ti.data_oriented
class MPMFields:
    """
    Manages all fields for MPM simulation:
    - Particle fields: position, velocity, deformation gradient, APIC matrix, mass, volume, Maxwell internal variables, energy increments
    - Grid fields: mass, velocity, tangential displacement, contact mask, no-contact age
    - Global scalars: kinetic energy, elastic energy, viscous energy (step/cumulative), projection energy (step/cumulative)

    When enable_grad=True, particle state fields (x, v, F, b_bar_e) are created with needs_grad=True
    for automatic differentiation support.
    """

    def __init__(self, config: MPMConfig, n_particles: int, enable_grad: bool = False):
        """
        Initialize MPM fields

        Args:
            config: MPM configuration
            n_particles: Number of particles
            enable_grad: If True, create particle state fields with needs_grad=True for autodiff
        """
        self.config = config
        self.n_particles = n_particles
        self.n_maxwell = len(config.material.maxwell_branches)
        self.enable_grad = enable_grad

        # Grid dimensions
        self.grid_size = config.grid.grid_size
        self.dx = config.grid.dx

        # Particle fields - with optional gradient support
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles, needs_grad=enable_grad)  # Position
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles, needs_grad=enable_grad)  # Velocity
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles, needs_grad=enable_grad)  # Deformation gradient
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # APIC affine matrix (no grad needed)
        self.mass = ti.field(dtype=ti.f32, shape=n_particles)  # Mass (typically not optimized)
        self.volume = ti.field(dtype=ti.f32, shape=n_particles)  # Volume (typically not optimized)

        # Maxwell branch internal variables: b_bar_e[k] for each branch
        if self.n_maxwell > 0:
            self.b_bar_e = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(n_particles, self.n_maxwell), needs_grad=enable_grad)
        else:
            self.b_bar_e = None

        # Energy increments (particle-level)
        self.delta_E_viscous_step = ti.field(dtype=ti.f32, shape=n_particles)  # Viscous dissipation this step
        self.delta_E_proj_step = ti.field(dtype=ti.f32, shape=n_particles)  # Projection correction this step

        # Grid fields
        self.grid_m = ti.field(dtype=ti.f32, shape=self.grid_size)  # Grid mass
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_size)  # Grid velocity
        self.grid_ut = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_size)  # Tangential displacement (persistent)
        self.grid_contact_mask = ti.field(dtype=ti.i32, shape=self.grid_size)  # Contact flag
        self.grid_nocontact_age = ti.field(dtype=ti.i32, shape=self.grid_size)  # No-contact age counter

        # Global energy scalars
        self.E_kin = ti.field(dtype=ti.f32, shape=())  # Kinetic energy
        self.E_elastic = ti.field(dtype=ti.f32, shape=())  # Elastic energy
        self.E_viscous_step = ti.field(dtype=ti.f32, shape=())  # Viscous dissipation this step
        self.E_viscous_cum = ti.field(dtype=ti.f32, shape=())  # Cumulative viscous dissipation
        self.E_proj_step = ti.field(dtype=ti.f32, shape=())  # Projection correction this step
        self.E_proj_cum = ti.field(dtype=ti.f32, shape=())  # Cumulative projection correction

    @ti.kernel
    def clear_grid(self):
        """Clear grid fields (except grid_ut and grid_nocontact_age which are persistent)"""
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector([0.0, 0.0, 0.0])
            # Clear contact mask (will be set in grid_op if in contact)
            self.grid_contact_mask[I] = 0
            # Note: grid_ut and grid_nocontact_age are NOT cleared here (persistent across steps)

    @ti.kernel
    def clear_particle_energy_increments(self):
        """Clear particle-level energy increments"""
        for p in range(self.n_particles):
            self.delta_E_viscous_step[p] = 0.0
            self.delta_E_proj_step[p] = 0.0

    @ti.kernel
    def clear_global_energy_step(self):
        """Clear step-level global energy accumulators"""
        self.E_viscous_step[None] = 0.0
        self.E_proj_step[None] = 0.0

    def initialize_particles(self, positions, velocities=None, volumes=None, density=None):
        """
        Initialize particle data

        Args:
            positions: (n_particles, 3) array of positions
            velocities: (n_particles, 3) array of velocities (optional)
            volumes: (n_particles,) array of volumes (optional)
            density: Material density (optional, uses config if not provided)
        """
        if density is None:
            density = self.config.material.density

        self.x.from_numpy(positions)

        if velocities is not None:
            self.v.from_numpy(velocities)
        else:
            self.v.fill(0.0)

        # Initialize deformation gradient to identity
        @ti.kernel
        def init_F():
            for p in range(self.n_particles):
                self.F[p] = ti.Matrix.identity(ti.f32, 3)
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)

        init_F()

        # Initialize volumes and masses
        if volumes is not None:
            self.volume.from_numpy(volumes)
        else:
            # Default: uniform volume based on grid spacing
            vol = (self.dx * 0.5) ** 3
            self.volume.fill(vol)

        @ti.kernel
        def init_mass():
            for p in range(self.n_particles):
                self.mass[p] = self.volume[p] * density

        init_mass()

        # Initialize Maxwell internal variables to identity
        if self.b_bar_e is not None:
            @ti.kernel
            def init_maxwell():
                for p in range(self.n_particles):
                    for k in ti.static(range(self.n_maxwell)):
                        self.b_bar_e[p, k] = ti.Matrix.identity(ti.f32, 3)

            init_maxwell()

    def get_particle_data(self):
        """Get particle data as numpy arrays"""
        return {
            'x': self.x.to_numpy(),
            'v': self.v.to_numpy(),
            'F': self.F.to_numpy(),
            'mass': self.mass.to_numpy(),
            'volume': self.volume.to_numpy()
        }

    def get_energy_data(self):
        """Get energy data as dictionary"""
        return {
            'E_kin': self.E_kin[None],
            'E_elastic': self.E_elastic[None],
            'E_viscous_step': self.E_viscous_step[None],
            'E_viscous_cum': self.E_viscous_cum[None],
            'E_proj_step': self.E_proj_step[None],
            'E_proj_cum': self.E_proj_cum[None]
        }

    def reset_gradients(self):
        """Reset gradient fields to zero. Only effective when enable_grad=True."""
        if not self.enable_grad:
            return

        # Reset particle field gradients using fill
        self.x.grad.fill(0.0)
        self.v.grad.fill(0.0)
        self.F.grad.fill(0.0)

        if self.b_bar_e is not None:
            self.b_bar_e.grad.fill(0.0)
# ===== END FILE: xengym/mpm/fields.py =====


# ===== BEGIN FILE: xengym/mpm/decomp.py =====
"""
Linear Algebra Decomposition Utilities
Provides wrappers for polar decomposition and symmetric eigenvalue decomposition.
Uses Taichi built-in functions by default, with slots for custom SafeSVD if needed.
Includes SPD projection with Straight-Through Estimator (STE) for autodiff support.
"""
import taichi as ti


@ti.func
def polar_decompose(F: ti.template()) -> ti.template():
    """
    Polar decomposition: F = R @ S
    Returns rotation matrix R and symmetric stretch matrix S

    Args:
        F: 3x3 deformation gradient matrix

    Returns:
        R: 3x3 rotation matrix
        S: 3x3 symmetric stretch matrix
    """
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    S = V @ ti.Matrix([[sig[0, 0], 0, 0],
                       [0, sig[1, 1], 0],
                       [0, 0, sig[2, 2]]]) @ V.transpose()
    return R, S


@ti.func
def safe_svd(F: ti.template(), eps: ti.f32 = 1e-8) -> ti.template():
    """
    Safe SVD with gradient handling for near-zero singular values

    Args:
        F: 3x3 matrix
        eps: Small epsilon for numerical stability

    Returns:
        U: 3x3 left singular vectors
        sig: 3x1 singular values
        V: 3x3 right singular vectors
    """
    # For now, use Taichi's built-in SVD
    # Can be replaced with custom implementation if needed for better autodiff stability
    U, sig, V = ti.svd(F)

    # Clamp singular values to avoid numerical issues
    for i in ti.static(range(3)):
        sig[i, i] = ti.max(sig[i, i], eps)

    return U, sig, V


@ti.func
def eig_sym_3x3(A: ti.template()) -> ti.template():
    """
    Symmetric eigenvalue decomposition for 3x3 matrix: A = Q @ Lambda @ Q^T

    Uses SVD as a more robust alternative to sym_eig for numerical stability.
    For symmetric positive semi-definite matrices: A = U Σ V^T where U = V.

    Args:
        A: 3x3 symmetric matrix (will be symmetrized if not perfectly symmetric)

    Returns:
        eigenvalues: 3x1 vector of eigenvalues
        eigenvectors: 3x3 matrix of eigenvectors (columns)
    """
    # Ensure matrix is symmetric
    A_sym = 0.5 * (A + A.transpose())

    # Use SVD for robust eigenvalue decomposition of symmetric matrix
    # For symmetric A: A = U Σ V^T, and U = V (up to sign)
    U, sig, V = ti.svd(A_sym)

    # Extract eigenvalues from singular values (diagonal of sig)
    eigenvalues = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])

    # For symmetric matrices, eigenvectors are the columns of U (or V)
    eigenvectors = U

    return eigenvalues, eigenvectors


@ti.func
def make_spd(A: ti.template(), eps: ti.f32 = 1e-8) -> ti.template():
    """
    Project matrix to symmetric positive definite (SPD) space

    Args:
        A: 3x3 matrix
        eps: Minimum eigenvalue threshold

    Returns:
        A_spd: 3x3 SPD matrix
    """
    # Symmetrize
    A_sym = 0.5 * (A + A.transpose())

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eig_sym_3x3(A_sym)

    # Clamp eigenvalues to be positive
    for i in ti.static(range(3)):
        eigenvalues[i] = ti.max(eigenvalues[i], eps)

    # Reconstruct SPD matrix
    Lambda = ti.Matrix([[eigenvalues[0], 0, 0],
                        [0, eigenvalues[1], 0],
                        [0, 0, eigenvalues[2]]])
    A_spd = eigenvectors @ Lambda @ eigenvectors.transpose()

    return A_spd


@ti.func
def make_spd_ste(A: ti.template(), eps: ti.f32 = 1e-8) -> ti.template():
    """
    Project matrix to SPD space with Straight-Through Estimator (STE) for autodiff.

    The STE allows gradients to flow through the non-differentiable SPD projection
    by treating the projection as identity in the backward pass.

    Forward: A_spd = make_spd(A)
    Backward: grad_A = grad_A_spd (straight-through)

    Implementation: A_spd_ste = stop_grad(A_spd) + A - stop_grad(A)

    Args:
        A: 3x3 matrix (the relaxed internal variable b_bar_e_relaxed)
        eps: Minimum eigenvalue threshold

    Returns:
        A_spd: 3x3 SPD matrix with STE gradient
    """
    # Compute actual SPD projection
    A_spd = make_spd(A, eps)

    # STE: gradient flows through A as if projection were identity
    # Forward: returns A_spd
    # Backward: grad flows to A directly
    A_spd_ste = ti.stop_grad(A_spd) + A - ti.stop_grad(A)

    return A_spd_ste


@ti.func
def clamp_J(F: ti.template(), J_min: ti.f32 = 0.5, J_max: ti.f32 = 2.0) -> ti.template():
    """
    Clamp Jacobian (determinant) of deformation gradient to avoid extreme compression/expansion

    Args:
        F: 3x3 deformation gradient
        J_min: Minimum allowed Jacobian
        J_max: Maximum allowed Jacobian

    Returns:
        F_clamped: 3x3 deformation gradient with clamped Jacobian
    """
    J = F.determinant()
    J_clamped = ti.max(ti.min(J, J_max), J_min)

    F_clamped = F
    if ti.abs(J) > 1e-10:
        scale = ti.pow(J_clamped / J, 1.0 / 3.0)
        F_clamped = scale * F

    return F_clamped
# ===== END FILE: xengym/mpm/decomp.py =====


# ===== BEGIN FILE: xengym/mpm/contact.py =====
"""
Contact and Friction Module
Implements SDF penalty-based contact with regularized elastoplastic friction
"""
import taichi as ti


@ti.func
def sdf_sphere(x: ti.template(), center: ti.template(), radius: ti.f32) -> ti.f32:
    """
    Signed distance function for sphere

    Args:
        x: Query point
        center: Sphere center
        radius: Sphere radius

    Returns:
        Signed distance (negative inside, positive outside)
    """
    return (x - center).norm() - radius


@ti.func
def sdf_plane(x: ti.template(), point: ti.template(), normal: ti.template()) -> ti.f32:
    """
    Signed distance function for plane

    Args:
        x: Query point
        point: Point on plane
        normal: Plane normal (unit vector)

    Returns:
        Signed distance (negative below, positive above)
    """
    return (x - point).dot(normal)


@ti.func
def sdf_box(x: ti.template(), center: ti.template(), half_extents: ti.template()) -> ti.f32:
    """
    Signed distance function for box

    Args:
        x: Query point
        center: Box center
        half_extents: Half extents in each dimension

    Returns:
        Signed distance
    """
    q = ti.abs(x - center) - half_extents
    return ti.max(q, 0.0).norm() + ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)


@ti.func
def sdf_capped_cylinder(
    x: ti.template(),
    center: ti.template(),
    radius: ti.f32,
    half_height: ti.f32,
) -> ti.f32:
    """
    Signed distance function for a capped cylinder aligned with the Z axis.

    Args:
        x: Query point
        center: Cylinder center
        radius: Cylinder radius
        half_height: Half height along Z (cap-to-cap height = 2 * half_height)

    Returns:
        Signed distance (negative inside, positive outside)
    """
    p = x - center
    d_xy = ti.sqrt(p[0] * p[0] + p[1] * p[1]) - radius
    d_z = ti.abs(p[2]) - half_height

    ax = ti.max(d_xy, 0.0)
    az = ti.max(d_z, 0.0)
    outside_dist = ti.sqrt(ax * ax + az * az)
    inside_dist = ti.min(ti.max(d_xy, d_z), 0.0)
    return inside_dist + outside_dist


@ti.func
def evaluate_sdf(
    x: ti.template(),
    sdf_type: ti.i32,
    center: ti.template(),
    normal: ti.template(),
    half_extents: ti.template()
) -> ti.f32:
    """
    Evaluate SDF for configurable obstacle type

    Args:
        x: Query point
        sdf_type: 0=plane, 1=sphere, 2=box, 3=cylinder
        center: Center/point on plane
        normal: Normal for plane (ignored for sphere/box/cylinder)
        half_extents: Half extents for box; (radius, 0, 0) for sphere;
            (radius, radius, half_height) for cylinder

    Returns:
        Signed distance (negative = penetration)
    """
    phi = 0.0
    if sdf_type == 0:  # Plane
        phi = sdf_plane(x, center, normal)
    elif sdf_type == 1:  # Sphere
        phi = sdf_sphere(x, center, half_extents[0])
    elif sdf_type == 3:  # Cylinder (capped, Z-axis)
        phi = sdf_capped_cylinder(x, center, half_extents[0], half_extents[2])
    else:  # Box (sdf_type == 2)
        phi = sdf_box(x, center, half_extents)
    return phi


@ti.func
def compute_sdf_normal(
    x: ti.template(),
    sdf_type: ti.i32,
    center: ti.template(),
    normal_plane: ti.template(),
    half_extents: ti.template()
) -> ti.template():
    """
    Compute outward normal for SDF obstacle

    Args:
        x: Query point
        sdf_type: 0=plane, 1=sphere, 2=box, 3=cylinder
        center: Center/point on plane
        normal_plane: Normal for plane
        half_extents: Half extents for box; (radius, 0, 0) for sphere;
            (radius, radius, half_height) for cylinder

    Returns:
        Outward normal vector (unit)
    """
    n = ti.Vector([0.0, 0.0, 1.0])

    if sdf_type == 0:  # Plane
        n = normal_plane
    elif sdf_type == 1:  # Sphere
        diff = x - center
        dist = diff.norm()
        if dist > 1e-8:
            n = diff / dist
        else:
            n = ti.Vector([0.0, 0.0, 1.0])
    elif sdf_type == 3:  # Cylinder (capped, Z-axis)
        p = x - center
        radius = half_extents[0]
        half_height = half_extents[2]

        d_xy = ti.sqrt(p[0] * p[0] + p[1] * p[1])
        qx = d_xy - radius
        qz = ti.abs(p[2]) - half_height

        ax = ti.max(qx, 0.0)
        az = ti.max(qz, 0.0)
        outside = (qx > 0.0) | (qz > 0.0)

        if outside:
            l = ti.sqrt(ax * ax + az * az)
            if l > 1e-8:
                side_w = ax / l
                cap_w = az / l

                nx = 0.0
                ny = 0.0
                nz = 0.0
                if d_xy > 1e-8:
                    nx = (p[0] / d_xy) * side_w
                    ny = (p[1] / d_xy) * side_w
                else:
                    nx = side_w
                    ny = 0.0

                nz = cap_w * ti.select(p[2] >= 0.0, 1.0, -1.0)
                n = ti.Vector([nx, ny, nz])
            else:
                if d_xy > 1e-8:
                    n = ti.Vector([p[0] / d_xy, p[1] / d_xy, 0.0])
                else:
                    n = ti.Vector([0.0, 0.0, 1.0])
        else:
            if qx > qz:
                if d_xy > 1e-8:
                    n = ti.Vector([p[0] / d_xy, p[1] / d_xy, 0.0])
                else:
                    n = ti.Vector([1.0, 0.0, 0.0])
            else:
                n = ti.Vector([0.0, 0.0, ti.select(p[2] >= 0.0, 1.0, -1.0)])
    else:  # Box (sdf_type == 2)
        # For box, compute normal from closest face
        q = x - center
        abs_q = ti.abs(q) - half_extents

        # Find which face is closest
        max_comp = ti.max(abs_q[0], ti.max(abs_q[1], abs_q[2]))
        n = ti.Vector([0.0, 0.0, 0.0])

        if abs_q[0] >= abs_q[1] and abs_q[0] >= abs_q[2]:
            n[0] = ti.select(q[0] >= 0, 1.0, -1.0)
        elif abs_q[1] >= abs_q[0] and abs_q[1] >= abs_q[2]:
            n[1] = ti.select(q[1] >= 0, 1.0, -1.0)
        else:
            n[2] = ti.select(q[2] >= 0, 1.0, -1.0)

    return n


@ti.func
def compute_contact_force(
    phi: ti.f32,
    v_rel: ti.template(),
    normal: ti.template(),
    u_t: ti.template(),
    dt: ti.f32,
    k_normal: ti.f32,
    k_tangent: ti.f32,
    mu_s: ti.f32,
    mu_k: ti.f32,
    v_transition: ti.f32
) -> ti.template():
    """
    Compute contact force with regularized elastoplastic friction

    Args:
        phi: Signed distance (negative = penetration)
        v_rel: Relative velocity
        normal: Contact normal (pointing outward from obstacle)
        u_t: Tangential displacement (elastic spring)
        dt: Time step
        k_normal: Normal contact stiffness
        k_tangent: Tangential spring stiffness
        mu_s: Static friction coefficient
        mu_k: Kinetic friction coefficient
        v_transition: Velocity for tanh transition

    Returns:
        f_contact: Contact force
        u_t_new: Updated tangential displacement
        is_contact: Contact flag (1 if in contact, 0 otherwise)
    """
    f_contact = ti.Vector([0.0, 0.0, 0.0])
    u_t_new = u_t
    is_contact = 0

    if phi < 0.0:  # Penetration
        is_contact = 1

        # Normal force (penalty method)
        f_n = -k_normal * phi * normal

        # Tangential velocity
        v_n = v_rel.dot(normal) * normal
        v_t = v_rel - v_n

        # Update tangential displacement (elastic spring)
        u_t_trial = u_t + v_t * dt

        # Compute tangential force (using tangential stiffness)
        f_t_trial = -k_tangent * u_t_trial

        # Friction limit
        f_n_mag = f_n.norm()
        f_t_mag = f_t_trial.norm()

        # Regularized friction coefficient (tanh transition from static to kinetic)
        v_t_mag = v_t.norm()
        mu_eff = mu_k + (mu_s - mu_k) * ti.tanh(v_transition / (v_t_mag + 1e-8))

        f_t_max = mu_eff * f_n_mag

        # Initialize f_t
        f_t = ti.Vector([0.0, 0.0, 0.0])

        # Elastoplastic friction: if |f_t| > f_t_max, slide
        if f_t_mag > f_t_max:
            # Sliding: limit tangential force and update u_t
            if f_t_mag > 1e-10:
                f_t = f_t_trial * (f_t_max / f_t_mag)
                u_t_new = -f_t / k_tangent
            else:
                f_t = ti.Vector([0.0, 0.0, 0.0])
                u_t_new = ti.Vector([0.0, 0.0, 0.0])
        else:
            # Sticking: use trial force
            f_t = f_t_trial
            u_t_new = u_t_trial

        f_contact = f_n + f_t

    return f_contact, u_t_new, is_contact


@ti.func
def update_contact_age(
    is_contact: ti.i32,
    age: ti.i32,
    K_clear: ti.i32
) -> ti.template():
    """
    Update no-contact age counter for hysteresis-based cleanup

    Args:
        is_contact: Contact flag (1 if in contact, 0 otherwise)
        age: Current no-contact age
        K_clear: Threshold for clearing tangential displacement

    Returns:
        age_new: Updated age
        should_clear: Flag indicating whether to clear u_t
    """
    age_new = age
    should_clear = 0

    if is_contact == 1:
        # In contact: reset age
        age_new = 0
    else:
        # Not in contact: increment age
        age_new = age + 1

        # Clear u_t if age exceeds threshold
        if age_new >= K_clear:
            should_clear = 1
            age_new = 0  # Reset age after clearing

    return age_new, should_clear
# ===== END FILE: xengym/mpm/contact.py =====


# ===== BEGIN FILE: xengym/mpm/constitutive.py =====
"""
Constitutive Models for VHE-MPM
Implements Ogden hyperelasticity + generalized Maxwell viscoelasticity + optional Kelvin-Voigt bulk viscosity
"""
import taichi as ti
from .decomp import eig_sym_3x3, make_spd, clamp_J


@ti.func
def compute_ogden_stress_general(F: ti.template(), mu: ti.template(), alpha: ti.template(), n_terms: ti.i32, kappa: ti.f32) -> ti.template():
    """
    Compute Ogden hyperelastic stress with arbitrary number of terms

    Args:
        F: 3x3 deformation gradient
        mu: Taichi field of shear moduli
        alpha: Taichi field of exponents
        n_terms: Number of Ogden terms
        kappa: Bulk modulus

    Returns:
        P: 3x3 first Piola-Kirchhoff stress
        psi: Elastic energy density
    """
    # Clamp Jacobian
    F_clamped = clamp_J(F, 0.5, 2.0)
    J = F_clamped.determinant()

    # Right Cauchy-Green tensor (ensure symmetric due to numerical precision)
    C = F_clamped.transpose() @ F_clamped
    C = 0.5 * (C + C.transpose())  # Explicit symmetrization

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eig_sym_3x3(C)

    # Principal stretches
    lambda_vec = ti.Vector([ti.sqrt(ti.max(eigenvalues[0], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[1], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[2], 1e-8))])

    # Deviatoric part
    J_pow = ti.pow(J, -1.0/3.0)
    lambda_bar = lambda_vec * J_pow

    # Compute deviatoric stress and energy
    psi_dev = 0.0
    S_dev_principal = ti.Vector([0.0, 0.0, 0.0])

    for k in ti.static(range(4)):  # Max 4 terms
        if k < n_terms:
            mu_k = mu[k]
            alpha_k = alpha[k]

            for i in ti.static(range(3)):
                psi_dev += mu_k / alpha_k * (ti.pow(lambda_bar[i], alpha_k) - 1.0)
                S_dev_principal[i] += mu_k * ti.pow(lambda_bar[i], alpha_k - 1.0)

    # Apply deviatoric projection
    trace_S = S_dev_principal[0] + S_dev_principal[1] + S_dev_principal[2]
    for i in ti.static(range(3)):
        S_dev_principal[i] = J_pow * (S_dev_principal[i] - trace_S / 3.0)

    # Reconstruct deviatoric 2nd PK stress
    S_dev = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                S_dev[i, j] += S_dev_principal[k] * eigenvectors[i, k] * eigenvectors[j, k]

    # Volumetric part
    psi_vol = 0.5 * kappa * (J - 1.0) ** 2
    S_vol = kappa * (J - 1.0) * J * F_clamped.inverse().transpose() @ F_clamped.inverse()

    # Total stress
    S = S_dev + S_vol
    P = F_clamped @ S
    psi = psi_dev + psi_vol

    return P, psi


@ti.func
def compute_ogden_stress_2terms(F: ti.template(), mu0: ti.f32, alpha0: ti.f32, mu1: ti.f32, alpha1: ti.f32, kappa: ti.f32) -> ti.template():
    """
    Compute Ogden hyperelastic stress with 2 terms (deviatoric + volumetric)

    Ogden model: W = sum_k mu_k/alpha_k * (lambda_1^alpha_k + lambda_2^alpha_k + lambda_3^alpha_k - 3) + kappa/2 * (J-1)^2

    Args:
        F: 3x3 deformation gradient
        mu0, alpha0: First Ogden term
        mu1, alpha1: Second Ogden term
        kappa: Bulk modulus

    Returns:
        P: 3x3 first Piola-Kirchhoff stress
        psi: Elastic energy density
    """
    # Clamp Jacobian to avoid extreme deformation
    F_clamped = clamp_J(F, 0.5, 2.0)
    J = F_clamped.determinant()

    # Right Cauchy-Green tensor (ensure symmetric due to numerical precision)
    C = F_clamped.transpose() @ F_clamped
    C = 0.5 * (C + C.transpose())  # Explicit symmetrization

    # Eigenvalue decomposition of C
    eigenvalues, eigenvectors = eig_sym_3x3(C)

    # Principal stretches (lambda_i = sqrt(eigenvalue_i))
    lambda_vec = ti.Vector([ti.sqrt(ti.max(eigenvalues[0], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[1], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[2], 1e-8))])

    # Deviatoric part: J^(-1/3) * lambda_i
    J_pow = ti.pow(J, -1.0/3.0)
    lambda_bar = lambda_vec * J_pow

    # Compute deviatoric stress and energy
    psi_dev = 0.0
    S_dev_principal = ti.Vector([0.0, 0.0, 0.0])

    # Term 0
    for i in ti.static(range(3)):
        psi_dev += mu0 / alpha0 * (ti.pow(lambda_bar[i], alpha0) - 1.0)
        S_dev_principal[i] += mu0 * ti.pow(lambda_bar[i], alpha0 - 1.0)

    # Term 1
    for i in ti.static(range(3)):
        psi_dev += mu1 / alpha1 * (ti.pow(lambda_bar[i], alpha1) - 1.0)
        S_dev_principal[i] += mu1 * ti.pow(lambda_bar[i], alpha1 - 1.0)

    # Apply deviatoric projection: S_dev = J^(-1/3) * (S_principal - 1/3 * tr(S_principal) * I)
    trace_S = S_dev_principal[0] + S_dev_principal[1] + S_dev_principal[2]
    for i in ti.static(range(3)):
        S_dev_principal[i] = J_pow * (S_dev_principal[i] - trace_S / 3.0)

    # Reconstruct deviatoric 2nd PK stress in original basis
    S_dev = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                S_dev[i, j] += S_dev_principal[k] * eigenvectors[i, k] * eigenvectors[j, k]

    # Volumetric part: kappa * (J - 1) * J
    psi_vol = 0.5 * kappa * (J - 1.0) ** 2
    S_vol = kappa * (J - 1.0) * J * F_clamped.inverse().transpose() @ F_clamped.inverse()

    # Total 2nd PK stress
    S = S_dev + S_vol

    # Convert to 1st PK stress: P = F @ S
    P = F_clamped @ S

    # Total energy density
    psi = psi_dev + psi_vol

    return P, psi


@ti.func
def update_maxwell_branch(
    F: ti.template(),
    b_bar_e_old: ti.template(),
    dt: ti.f32,
    G: ti.f32,
    tau: ti.f32
) -> ti.template():
    """
    Update single Maxwell branch with upper-convected derivative + relaxation + SPD projection

    Args:
        F: 3x3 deformation gradient
        b_bar_e_old: 3x3 elastic left Cauchy-Green tensor (isochoric) from previous step
        dt: Time step
        G: Shear modulus of this branch
        tau: Relaxation time

    Returns:
        b_bar_e_new: Updated elastic left Cauchy-Green tensor
        tau_maxwell: Cauchy stress contribution from this branch
        delta_E_proj: Energy correction due to SPD projection
    """
    J = F.determinant()
    F_bar = ti.pow(J, -1.0/3.0) * F  # Isochoric deformation gradient

    # Velocity gradient (approximated from F)
    # For explicit MPM, we use: L ≈ (F - F_old) / dt / F_old ≈ (I - F_old^-1 @ F) / dt
    # Simplified: use F_bar directly for upper-convected update
    F_bar_inv = F_bar.inverse()

    # Upper-convected derivative: b_bar_e_dot = L @ b_bar_e + b_bar_e @ L^T
    # Approximation: b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar^T (push-forward)
    b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar.transpose()

    # Relaxation: b_bar_e_new = b_bar_e_trial * exp(-dt/tau)
    relax_factor = ti.exp(-dt / tau)
    b_bar_e_relaxed = relax_factor * b_bar_e_trial + (1.0 - relax_factor) * ti.Matrix.identity(ti.f32, 3)

    # SPD projection (ensure positive definiteness and isochoric constraint)
    b_bar_e_new = make_spd(b_bar_e_relaxed, 1e-8)

    # Enforce isochoric constraint: det(b_bar_e) = 1
    det_b = b_bar_e_new.determinant()
    if det_b > 1e-10:
        scale = ti.pow(det_b, -1.0/3.0)
        b_bar_e_new = scale * b_bar_e_new

    # Compute energy correction due to projection
    # ΔE_proj ≈ G/2 * ||b_bar_e_new - b_bar_e_relaxed||^2 (Frobenius norm)
    diff = b_bar_e_new - b_bar_e_relaxed
    delta_E_proj = 0.5 * G * (diff[0,0]**2 + diff[0,1]**2 + diff[0,2]**2 +
                               diff[1,0]**2 + diff[1,1]**2 + diff[1,2]**2 +
                               diff[2,0]**2 + diff[2,1]**2 + diff[2,2]**2)

    # Compute Cauchy stress: tau = G * dev(b_bar_e)
    trace_b = b_bar_e_new[0,0] + b_bar_e_new[1,1] + b_bar_e_new[2,2]
    tau_maxwell = G * (b_bar_e_new - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))

    return b_bar_e_new, tau_maxwell, delta_E_proj


@ti.func
def compute_maxwell_stress(
    F: ti.template(),
    b_bar_e: ti.template(),
    dt: ti.f32,
    maxwell_G: ti.template(),
    maxwell_tau: ti.template(),
    n_maxwell: ti.i32
) -> ti.template():
    """
    Compute total Maxwell viscoelastic stress and update internal variables

    Args:
        F: 3x3 deformation gradient
        b_bar_e: (n_maxwell, 3, 3) elastic left Cauchy-Green tensors
        dt: Time step
        maxwell_G: List of shear moduli
        maxwell_tau: List of relaxation times
        n_maxwell: Number of Maxwell branches

    Returns:
        tau_total: Total Cauchy stress from all Maxwell branches
        b_bar_e_new: Updated internal variables
        delta_E_proj_total: Total energy correction
    """
    J = F.determinant()
    tau_total = ti.Matrix.zero(ti.f32, 3, 3)
    delta_E_proj_total = 0.0

    b_bar_e_new = ti.Matrix.zero(ti.f32, 3, 3)  # Placeholder, will be updated in loop

    for k in ti.static(range(n_maxwell)):
        b_bar_e_k_new, tau_k, delta_E_k = update_maxwell_branch(
            F, b_bar_e[k], dt, maxwell_G[k], maxwell_tau[k]
        )
        b_bar_e[k] = b_bar_e_k_new
        tau_total += tau_k
        delta_E_proj_total += delta_E_k

    # Convert Cauchy stress to 1st PK stress: P = J * tau * F^-T
    P_maxwell = J * tau_total @ F.inverse().transpose()

    return P_maxwell, delta_E_proj_total


@ti.func
def compute_maxwell_stress_no_update(
    F: ti.template(),
    b_bar_e: ti.template(),
    G: ti.f32,
    tau: ti.f32
) -> ti.template():
    """
    Compute Maxwell branch stress WITHOUT updating internal variables.
    Used for numerical differentiation in gradient computation.

    Args:
        F: 3x3 deformation gradient
        b_bar_e: 3x3 elastic left Cauchy-Green tensor (current state, not modified)
        G: Shear modulus of this branch
        tau: Relaxation time (not used here, kept for API consistency)

    Returns:
        P_maxwell: 1st PK stress contribution from this branch
    """
    J = F.determinant()

    # Compute Cauchy stress: tau = G * dev(b_bar_e)
    trace_b = b_bar_e[0, 0] + b_bar_e[1, 1] + b_bar_e[2, 2]
    tau_maxwell = G * (b_bar_e - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))

    # Convert Cauchy stress to 1st PK stress: P = J * tau * F^-T
    P_maxwell = J * tau_maxwell @ F.inverse().transpose()

    return P_maxwell


@ti.func
def compute_bulk_viscosity_stress_no_energy(
    F: ti.template(),
    F_old: ti.template(),
    dt: ti.f32,
    eta_bulk: ti.f32
) -> ti.template():
    """
    Compute Kelvin-Voigt bulk viscosity stress WITHOUT energy calculation.
    Used for numerical differentiation in gradient computation.

    Args:
        F: Current deformation gradient
        F_old: Previous deformation gradient
        dt: Time step
        eta_bulk: Bulk viscosity coefficient

    Returns:
        P_visc: 1st PK stress from bulk viscosity
    """
    J = F.determinant()
    J_old = F_old.determinant()

    # Volumetric strain rate: J_dot / J ≈ (J - J_old) / (dt * J)
    J_dot = (J - J_old) / dt
    vol_strain_rate = J_dot / J

    # Bulk viscosity stress (Cauchy): sigma_visc = eta_bulk * (J_dot / J) * I
    sigma_visc = eta_bulk * vol_strain_rate * ti.Matrix.identity(ti.f32, 3)

    # Convert to 1st PK stress: P = J * sigma * F^-T
    P_visc = J * sigma_visc @ F.inverse().transpose()

    return P_visc


@ti.func
def compute_bulk_viscosity_stress(F: ti.template(), F_old: ti.template(), dt: ti.f32, eta_bulk: ti.f32) -> ti.template():
    """
    Compute Kelvin-Voigt bulk viscosity stress

    Args:
        F: Current deformation gradient
        F_old: Previous deformation gradient
        dt: Time step
        eta_bulk: Bulk viscosity coefficient

    Returns:
        P_visc: 1st PK stress from bulk viscosity
        delta_E_visc: Viscous dissipation
    """
    J = F.determinant()
    J_old = F_old.determinant()

    # Volumetric strain rate: J_dot / J ≈ (J - J_old) / (dt * J)
    J_dot = (J - J_old) / dt
    vol_strain_rate = J_dot / J

    # Bulk viscosity stress (Cauchy): sigma_visc = eta_bulk * (J_dot / J) * I
    sigma_visc = eta_bulk * vol_strain_rate * ti.Matrix.identity(ti.f32, 3)

    # Convert to 1st PK stress: P = J * sigma * F^-T
    P_visc = J * sigma_visc @ F.inverse().transpose()

    # Viscous dissipation: delta_E = eta_bulk * (J_dot / J)^2 * V * dt
    delta_E_visc = eta_bulk * vol_strain_rate ** 2

    return P_visc, delta_E_visc
# ===== END FILE: xengym/mpm/constitutive.py =====


# ===== BEGIN FILE: xengym/mpm/surface_mesh.py =====
"""
Top-surface mesh export utilities for the MPM solver.

This module provides a small, optional exporter that converts particle positions into a
regular-grid surface mesh (vertices + normals) in **millimeters**.

Design notes:
- Pure NumPy post-processing (no Taichi kernels), so it is only paid for when explicitly used.
- The mesh includes **tangential (x/y) displacement** by averaging per-cell surface particle motion.
- The default output coordinate frame matches the legacy MPM height-field renderer:
  x is centered around 0 in [-gel_w/2, +gel_w/2], y is in [0, gel_h], z is displacement from the initial
  top surface (mm, usually <= 0 under indentation when using edge-referenced baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

ArrayF32 = NDArray[np.float32]


def _box_blur_2d(values: NDArray[np.floating], iterations: int) -> ArrayF32:
    if values.ndim != 2:
        raise ValueError("values must be (H,W)")
    result = values.astype(np.float32, copy=True)
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")
        result = (
            padded[0:-2, 0:-2]
            + padded[0:-2, 1:-1]
            + padded[0:-2, 2:]
            + padded[1:-1, 0:-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, 0:-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / np.float32(9.0)
    return result


def _box_blur_2d_xy(values: NDArray[np.floating], iterations: int) -> ArrayF32:
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("values must be (H,W,2)")
    result = values.astype(np.float32, copy=True)
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        result = (
            padded[0:-2, 0:-2]
            + padded[0:-2, 1:-1]
            + padded[0:-2, 2:]
            + padded[1:-1, 0:-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, 0:-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / np.float32(9.0)
    return result


def _fill_height_holes(height_mm: ArrayF32, valid_mask: NDArray[np.bool_], max_iterations: int) -> ArrayF32:
    """
    Diffusion fill for height map holes.

    Any remaining holes after max_iterations are set to 0 (flat).
    """
    if height_mm.ndim != 2:
        raise ValueError("height_mm must be (H,W)")
    if valid_mask.shape != height_mm.shape:
        raise ValueError("valid_mask shape must match height_mm")

    result = height_mm.astype(np.float32, copy=True)
    filled = valid_mask.copy()

    h, w = result.shape
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for _ in range(max(int(max_iterations), 0)):
        if bool(filled.all()):
            break

        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2]
            + padded_filled[0:-2, 1:-1]
            + padded_filled[0:-2, 2:]
            + padded_filled[1:-1, 0:-2]
            + padded_filled[1:-1, 2:]
            + padded_filled[2:, 0:-2]
            + padded_filled[2:, 1:-1]
            + padded_filled[2:, 2:]
        )

        can_fill = (~filled) & (neighbor_count > 0)
        if not bool(can_fill.any()):
            break

        padded_h = np.pad(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), ((1, 1), (1, 1)), mode="constant")
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros((h, w), dtype=np.float32)
        neighbor_weight = np.zeros((h, w), dtype=np.float32)
        for di, dj in neighbor_offsets:
            i_slice = slice(1 + di, 1 + di + h)
            j_slice = slice(1 + dj, 1 + dj + w)
            w_ij = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_h[i_slice, j_slice] * w_ij
            neighbor_weight += w_ij

        fill_mask = can_fill & (neighbor_weight > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_weight[fill_mask]
        filled[fill_mask] = True

    result[~filled] = 0.0
    return result


def _fill_uv_holes(uv_mm: ArrayF32, valid_mask: NDArray[np.bool_], max_iterations: int) -> ArrayF32:
    """Diffusion fill for (H,W,2) uv displacement holes."""
    if uv_mm.ndim != 3 or uv_mm.shape[-1] != 2:
        raise ValueError("uv_mm must be (H,W,2)")
    if valid_mask.shape != uv_mm.shape[:2]:
        raise ValueError("valid_mask shape must match uv_mm[:,:,0]")

    result = uv_mm.astype(np.float32, copy=True)
    filled = valid_mask.copy()

    h, w = valid_mask.shape
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for _ in range(max(int(max_iterations), 0)):
        if bool(filled.all()):
            break

        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2]
            + padded_filled[0:-2, 1:-1]
            + padded_filled[0:-2, 2:]
            + padded_filled[1:-1, 0:-2]
            + padded_filled[1:-1, 2:]
            + padded_filled[2:, 0:-2]
            + padded_filled[2:, 1:-1]
            + padded_filled[2:, 2:]
        )

        can_fill = (~filled) & (neighbor_count > 0)
        if not bool(can_fill.any()):
            break

        padded_uv = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0)
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros((h, w, 2), dtype=np.float32)
        neighbor_weight = np.zeros((h, w), dtype=np.float32)
        for di, dj in neighbor_offsets:
            i_slice = slice(1 + di, 1 + di + h)
            j_slice = slice(1 + dj, 1 + dj + w)
            w_ij = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_uv[i_slice, j_slice] * w_ij[..., None]
            neighbor_weight += w_ij

        fill_mask = can_fill & (neighbor_weight > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_weight[fill_mask][..., None]
        filled[fill_mask] = True

    result[~filled] = 0.0
    return result


def _compute_vertex_normals_mm(vertices_mm: ArrayF32) -> ArrayF32:
    """Compute per-vertex normals for a regular grid mesh."""
    if vertices_mm.ndim != 3 or vertices_mm.shape[-1] != 3:
        raise ValueError("vertices_mm must be (H,W,3)")

    v = vertices_mm.astype(np.float32, copy=False)
    h, w, _ = v.shape

    dx = np.empty((h, w, 3), dtype=np.float32)
    dy = np.empty((h, w, 3), dtype=np.float32)

    dx[:, 1:-1] = v[:, 2:] - v[:, :-2]
    dx[:, 0] = v[:, 1] - v[:, 0]
    dx[:, -1] = v[:, -1] - v[:, -2]

    dy[1:-1] = v[2:] - v[:-2]
    dy[0] = v[1] - v[0]
    dy[-1] = v[-1] - v[-2]

    n = np.cross(dx, dy).astype(np.float32, copy=False)
    norm = np.linalg.norm(n, axis=-1, keepdims=True).astype(np.float32, copy=False)
    n = n / np.maximum(norm, np.float32(1e-8))
    return n.astype(np.float32, copy=False)


@dataclass(frozen=True)
class TopSurfaceMeshFrame:
    """A single exported top-surface mesh frame (all units in millimeters)."""

    vertices_mm: ArrayF32  # (H,W,3)
    normals: ArrayF32  # (H,W,3), unit length
    height_mm: ArrayF32  # (H,W), z displacement (mm)
    uv_disp_mm: ArrayF32  # (H,W,2), tangential displacement (mm)


class TopSurfaceMeshExporter:
    """
    Export a regular-grid top-surface mesh (vertices + normals) from MPM particle positions.

    The exporter uses a 4-neighbor splat to estimate a per-cell top-surface displacement field,
    then derives an (x,y) tangential displacement field from particles near that local top surface.

    Output:
    - vertices_mm: (H,W,3) float32 in mm
    - normals: (H,W,3) float32 unit vectors
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        initial_positions_m: NDArray[np.floating],
        *,
        dx_m: float,
        particles_per_cell: float,
        initial_top_z_m: Optional[float] = None,
        reference_edge: bool = True,
        fill_holes: bool = True,
        fill_iters: int = 10,
        smooth_iters: int = 0,
    ) -> None:
        n_row, n_col = (int(grid_shape[0]), int(grid_shape[1]))
        if n_row <= 0 or n_col <= 0:
            raise ValueError("grid_shape must be positive")

        init = np.asarray(initial_positions_m, dtype=np.float32)
        if init.ndim != 2 or init.shape[1] != 3:
            raise ValueError("initial_positions_m must be (N,3)")
        if init.shape[0] == 0:
            raise ValueError("initial_positions_m must be non-empty")

        self.grid_shape = (n_row, n_col)
        self.reference_edge = bool(reference_edge)
        self.fill_holes = bool(fill_holes)
        self.fill_iters = int(fill_iters)
        self.smooth_iters = int(smooth_iters)

        self.dx_m = float(dx_m)
        self.particles_per_cell = float(particles_per_cell)
        if not np.isfinite(self.dx_m) or self.dx_m <= 0:
            raise ValueError("dx_m must be finite and > 0")
        if not np.isfinite(self.particles_per_cell) or self.particles_per_cell <= 0:
            raise ValueError("particles_per_cell must be finite and > 0")

        init_mm = init * np.float32(1000.0)
        x_min, x_max = float(init_mm[:, 0].min()), float(init_mm[:, 0].max())
        y_min, y_max = float(init_mm[:, 1].min()), float(init_mm[:, 1].max())

        self.gel_w_mm = np.float32(max(x_max - x_min, 1e-6))
        self.gel_h_mm = np.float32(max(y_max - y_min, 1e-6))
        self._x_center_mm = np.float32((x_min + x_max) / 2.0)
        self._y_min_mm = np.float32(y_min)

        if initial_top_z_m is None:
            initial_top_z_m = float(np.max(init[:, 2]))
        self.initial_top_z_m = float(initial_top_z_m)
        self._z_top_init_mm = np.float32(self.initial_top_z_m * 1000.0)

        cell_w = np.float32(self.gel_w_mm / np.float32(n_col))
        cell_h = np.float32(self.gel_h_mm / np.float32(n_row))
        self._cell_w = cell_w
        self._cell_h = cell_h

        # Cell-center coordinates in the exported mesh plane.
        x_centers = (np.arange(n_col, dtype=np.float32) + np.float32(0.5)) * cell_w - np.float32(self.gel_w_mm / 2.0)
        y_centers = (np.arange(n_row, dtype=np.float32) + np.float32(0.5)) * cell_h
        xx, yy = np.meshgrid(x_centers, y_centers)
        self._grid_x_mm = xx.astype(np.float32, copy=False)
        self._grid_y_mm = yy.astype(np.float32, copy=False)

        # Cache initial positions in the same "sensor" frame for displacement.
        init_sensor = init_mm.copy()
        init_sensor[:, 0] -= self._x_center_mm
        init_sensor[:, 1] -= self._y_min_mm
        self._init_sensor_mm = init_sensor.astype(np.float32, copy=False)

        particle_spacing_m = self.dx_m / max(self.particles_per_cell, 1.0)
        self._surface_band_mm = np.float32(2.0 * particle_spacing_m * 1000.0)

    def extract(self, positions_m: NDArray[np.floating]) -> TopSurfaceMeshFrame:
        pos = np.asarray(positions_m, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions_m must be (N,3)")
        if pos.shape[0] != self._init_sensor_mm.shape[0]:
            raise ValueError("positions_m must have the same N as initial_positions_m")

        n_row, n_col = self.grid_shape

        pos_mm = pos * np.float32(1000.0)
        pos_sensor = pos_mm.copy()
        pos_sensor[:, 0] -= self._x_center_mm
        pos_sensor[:, 1] -= self._y_min_mm

        x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
        y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
        z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)

        z_disp = (z_mm - self._z_top_init_mm).astype(np.float32, copy=False)

        col_f = (x_mm + np.float32(self.gel_w_mm / 2.0)) / self._cell_w - np.float32(0.5)
        row_f = y_mm / self._cell_h - np.float32(0.5)
        col0 = np.floor(col_f).astype(np.int32, copy=False)
        row0 = np.floor(row_f).astype(np.int32, copy=False)

        # Height field: max z displacement per cell.
        height_mm = np.full((n_row, n_col), -np.inf, dtype=np.float32)
        for di in (0, 1):
            rr = row0 + di
            for dj in (0, 1):
                cc = col0 + dj
                m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                if bool(m.any()):
                    np.maximum.at(height_mm, (rr[m], cc[m]), z_disp[m])
        height_mm[~np.isfinite(height_mm)] = np.nan
        valid = np.isfinite(height_mm)

        reference_z = np.float32(0.0)
        if self.reference_edge and bool(valid.any()):
            edge_margin = max(3, n_row // 20)
            edge_mask = np.zeros_like(valid, dtype=bool)
            edge_mask[:edge_margin, :] = True
            edge_mask[-edge_margin:, :] = True
            if edge_margin * 2 < n_row and edge_margin * 2 < n_col:
                edge_mask[edge_margin:-edge_margin, :edge_margin] = True
                edge_mask[edge_margin:-edge_margin, -edge_margin:] = True

            edge_values = height_mm[edge_mask & valid]
            edge_values = edge_values[np.isfinite(edge_values) & (edge_values > -10)]
            if edge_values.size > 0:
                reference_z = np.float32(np.median(edge_values))
                height_mm = height_mm - reference_z

        # Ensure the exported height field is finite everywhere.
        if self.fill_holes:
            height_mm = _fill_height_holes(height_mm, np.isfinite(height_mm), max_iterations=self.fill_iters)
        else:
            height_mm = np.where(np.isfinite(height_mm), height_mm, 0.0).astype(np.float32, copy=False)

        if self.smooth_iters > 0:
            height_mm = _box_blur_2d(height_mm, iterations=self.smooth_iters)

        # Tangential displacement: average (dx, dy) of particles near the local top surface.
        disp_x = (pos_sensor[:, 0] - self._init_sensor_mm[:, 0]).astype(np.float32, copy=False)
        disp_y = (pos_sensor[:, 1] - self._init_sensor_mm[:, 1]).astype(np.float32, copy=False)
        z_disp_ref = (z_disp - reference_z).astype(np.float32, copy=False)

        uv_sum = np.zeros((n_row, n_col, 2), dtype=np.float32)
        uv_cnt = np.zeros((n_row, n_col), dtype=np.int32)
        for di in (0, 1):
            rr = row0 + di
            for dj in (0, 1):
                cc = col0 + dj
                m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                if not bool(m.any()):
                    continue
                rr_m = rr[m]
                cc_m = cc[m]
                ref = height_mm[rr_m, cc_m].astype(np.float32, copy=False)
                top = z_disp_ref[m] >= (ref - self._surface_band_mm)
                if not bool(top.any()):
                    continue
                rr_t = rr_m[top]
                cc_t = cc_m[top]
                np.add.at(uv_sum[..., 0], (rr_t, cc_t), disp_x[m][top])
                np.add.at(uv_sum[..., 1], (rr_t, cc_t), disp_y[m][top])
                np.add.at(uv_cnt, (rr_t, cc_t), 1)

        uv_mm = np.zeros((n_row, n_col, 2), dtype=np.float32)
        nonzero = uv_cnt > 0
        uv_mm[nonzero] = uv_sum[nonzero] / uv_cnt[nonzero][..., None]
        if self.fill_holes:
            uv_mm = _fill_uv_holes(uv_mm, nonzero, max_iterations=self.fill_iters)
        if self.smooth_iters > 0:
            uv_mm = _box_blur_2d_xy(uv_mm, iterations=self.smooth_iters)

        vertices_mm = np.empty((n_row, n_col, 3), dtype=np.float32)
        vertices_mm[..., 0] = self._grid_x_mm + uv_mm[..., 0]
        vertices_mm[..., 1] = self._grid_y_mm + uv_mm[..., 1]
        vertices_mm[..., 2] = height_mm

        normals = _compute_vertex_normals_mm(vertices_mm)
        return TopSurfaceMeshFrame(
            vertices_mm=vertices_mm.astype(np.float32, copy=False),
            normals=normals.astype(np.float32, copy=False),
            height_mm=height_mm.astype(np.float32, copy=False),
            uv_disp_mm=uv_mm.astype(np.float32, copy=False),
        )
# ===== END FILE: xengym/mpm/surface_mesh.py =====


# ===== BEGIN FILE: xengym/mpm/mpm_solver.py =====
"""
MPM Solver Main Flow
Implements the complete MLS-MPM/APIC solver with VHE constitutive model
Supports automatic differentiation via enable_grad parameter
"""
from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING

import taichi as ti
import numpy as np
from numpy.typing import NDArray

from .config import MPMConfig
from .fields import MPMFields
from .constitutive import compute_ogden_stress_general, compute_ogden_stress_2terms, compute_maxwell_stress, compute_bulk_viscosity_stress
from .contact import compute_contact_force, update_contact_age, sdf_plane, evaluate_sdf, compute_sdf_normal
from .decomp import make_spd, make_spd_ste
from .exceptions import ConfigurationError, MaterialError


@ti.data_oriented
class MPMSolver:
    """
    3D Explicit MLS-MPM/APIC Solver with VHE constitutive model
    Supports automatic differentiation when enable_grad=True
    """

    def __init__(self, config: MPMConfig, n_particles: int, enable_grad: bool = False, use_spd_ste: bool = True):
        """
        Initialize MPM solver

        Args:
            config: MPM configuration
            n_particles: Number of particles
            enable_grad: If True, enable automatic differentiation support
            use_spd_ste: If True, use Straight-Through Estimator for SPD projection in AD mode
        """
        self.config = config
        self.n_particles = n_particles
        self.enable_grad = enable_grad
        self.use_spd_ste = use_spd_ste
        self.fields = MPMFields(config, n_particles, enable_grad=enable_grad)
        self.current_step = 0

        # Loss field for autodiff (always create, only needs_grad when enable_grad=True)
        self.loss_field = ti.field(dtype=ti.f32, shape=(), needs_grad=enable_grad)

        # Material parameters (convert to Taichi fields for kernel access)
        # Support up to 4 Ogden terms
        self.n_ogden = min(len(config.material.ogden.mu), 4)
        if self.n_ogden == 0:
            raise MaterialError("At least one Ogden term is required")
        if len(config.material.ogden.mu) != len(config.material.ogden.alpha):
            raise MaterialError("Ogden mu and alpha must have the same length")

        # Ogden parameters with optional gradient support
        self.ogden_mu = ti.field(dtype=ti.f32, shape=4, needs_grad=enable_grad)
        self.ogden_alpha = ti.field(dtype=ti.f32, shape=4, needs_grad=enable_grad)
        # Fill with actual values
        for i in range(self.n_ogden):
            self.ogden_mu[i] = config.material.ogden.mu[i]
            self.ogden_alpha[i] = config.material.ogden.alpha[i]
        # Fill remaining with zeros (won't be used)
        for i in range(self.n_ogden, 4):
            self.ogden_mu[i] = 0.0
            self.ogden_alpha[i] = 1.0

        self.ogden_kappa = config.material.ogden.kappa

        # Maxwell parameters with optional gradient support
        self.n_maxwell = len(config.material.maxwell_branches)
        if self.n_maxwell > 0:
            self.maxwell_G = ti.field(dtype=ti.f32, shape=self.n_maxwell, needs_grad=enable_grad)
            self.maxwell_tau = ti.field(dtype=ti.f32, shape=self.n_maxwell, needs_grad=enable_grad)
            G_list = [b.G for b in config.material.maxwell_branches]
            tau_list = [b.tau for b in config.material.maxwell_branches]
            self.maxwell_G.from_numpy(np.array(G_list, dtype=np.float32))
            self.maxwell_tau.from_numpy(np.array(tau_list, dtype=np.float32))

        # Contact parameters
        self.enable_contact = config.contact.enable_contact
        self.contact_stiffness_normal = config.contact.contact_stiffness_normal
        self.contact_stiffness_tangent = config.contact.contact_stiffness_tangent
        self.mu_s = config.contact.mu_s
        self.mu_k = config.contact.mu_k
        self.v_transition = config.contact.friction_transition_vel
        self.K_clear = config.contact.K_clear

        # SDF obstacles (default: ground plane at z=0 for backward compatibility)
        obstacles = config.contact.obstacles
        if len(obstacles) == 0:
            # Default ground plane
            self.n_obstacles = 1
            self.obstacle_types = ti.field(dtype=ti.i32, shape=1)
            self.obstacle_centers = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.obstacle_normals = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.obstacle_half_extents = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.obstacle_types[0] = 0  # plane
            self.obstacle_centers[0] = ti.Vector([0.0, 0.0, 0.0])
            self.obstacle_normals[0] = ti.Vector([0.0, 0.0, 1.0])
            self.obstacle_half_extents[0] = ti.Vector([0.0, 0.0, 0.0])
        else:
            self.n_obstacles = len(obstacles)
            self.obstacle_types = ti.field(dtype=ti.i32, shape=self.n_obstacles)
            self.obstacle_centers = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
            self.obstacle_normals = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
            self.obstacle_half_extents = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
            type_map = {'plane': 0, 'sphere': 1, 'box': 2, 'cylinder': 3}
            for i, obs in enumerate(obstacles):
                self.obstacle_types[i] = type_map.get(obs.sdf_type, 0)
                self.obstacle_centers[i] = ti.Vector(list(obs.center))
                self.obstacle_normals[i] = ti.Vector(list(obs.normal))
                self.obstacle_half_extents[i] = ti.Vector(list(obs.half_extents))

        # Obstacle kinematics (for moving obstacles / friction in relative frame)
        self.obstacle_centers_prev = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
        self.obstacle_velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
        for i in range(self.n_obstacles):
            self.obstacle_centers_prev[i] = self.obstacle_centers[i]
            self.obstacle_velocities[i] = ti.Vector([0.0, 0.0, 0.0])

        # Grid boundary condition (legacy "sticky walls" at domain boundaries)
        self.sticky_boundary = bool(config.grid.sticky_boundary)
        self.sticky_boundary_width = int(config.grid.sticky_boundary_width)
        if self.sticky_boundary_width < 0:
            raise ConfigurationError("grid.sticky_boundary_width must be >= 0")

        # Time stepping
        self.dt = config.time.dt
        self.dx = config.grid.dx
        self.inv_dx = 1.0 / self.dx

        # Gravity
        self.gravity = ti.Vector([0.0, 0.0, -9.81])

    def initialize_particles(
        self,
        positions: NDArray[np.float32],
        velocities: Optional[NDArray[np.float32]] = None,
        volumes: Optional[NDArray[np.float32]] = None
    ) -> None:
        """Initialize particle data"""
        self.fields.initialize_particles(positions, velocities, volumes)

    @ti.kernel
    def p2g(self):
        """Particle to grid transfer (P2G)"""
        for p in range(self.n_particles):
            # Particle state
            x_p = self.fields.x[p]
            v_p = self.fields.v[p]
            F_p = self.fields.F[p]
            C_p = self.fields.C[p]
            m_p = self.fields.mass[p]
            V_p = self.fields.volume[p]

            # Compute stress (using general Ogden model)
            P_elastic, psi_elastic = compute_ogden_stress_general(
                F_p,
                self.ogden_mu,
                self.ogden_alpha,
                self.n_ogden,
                self.ogden_kappa
            )

            # Add Maxwell stress if enabled
            P_total = P_elastic
            if ti.static(self.n_maxwell > 0):
                # Compute Maxwell stress from internal variables
                J = F_p.determinant()
                tau_maxwell_total = ti.Matrix.zero(ti.f32, 3, 3)

                for k in ti.static(range(self.n_maxwell)):
                    b_bar_e_k = self.fields.b_bar_e[p, k]
                    G_k = self.maxwell_G[k]

                    # Cauchy stress: tau = G * dev(b_bar_e)
                    trace_b = b_bar_e_k[0,0] + b_bar_e_k[1,1] + b_bar_e_k[2,2]
                    tau_k = G_k * (b_bar_e_k - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))
                    tau_maxwell_total += tau_k

                # Convert Cauchy stress to 1st PK stress: P = J * tau * F^-T
                P_maxwell = J * tau_maxwell_total @ F_p.inverse().transpose()
                P_total = P_elastic + P_maxwell

            # Add bulk viscosity stress if enabled
            if ti.static(self.config.material.enable_bulk_viscosity):
                # Approximate velocity gradient from C
                L = self.fields.C[p]
                trace_L = L[0,0] + L[1,1] + L[2,2]

                # Bulk viscosity stress (Cauchy): sigma_visc = eta_bulk * tr(D) * I
                # where D = (L + L^T) / 2 is the rate of deformation
                eta_bulk = self.config.material.bulk_viscosity
                sigma_visc = eta_bulk * trace_L * ti.Matrix.identity(ti.f32, 3)

                # Convert to 1st PK stress
                J = F_p.determinant()
                P_visc = J * sigma_visc @ F_p.inverse().transpose()
                P_total += P_visc

            # Affine momentum
            affine = P_total @ F_p.transpose() * V_p + m_p * C_p

            # Grid node base index
            base = ti.cast(x_p * self.inv_dx - 0.5, ti.i32)

            # Quadratic B-spline weights
            fx = x_p * self.inv_dx - ti.cast(base, ti.f32)

            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

            # Scatter to grid
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]

                grid_idx = base + offset

                # Check bounds
                if 0 <= grid_idx[0] < self.fields.grid_size[0] and \
                   0 <= grid_idx[1] < self.fields.grid_size[1] and \
                   0 <= grid_idx[2] < self.fields.grid_size[2]:

                    # Mass
                    self.fields.grid_m[grid_idx] += weight * m_p

                    # Momentum (with affine contribution)
                    self.fields.grid_v[grid_idx] += weight * (m_p * v_p + affine @ dpos)

    @ti.kernel
    def grid_op(self):
        """Grid operations: apply forces, boundary conditions, and contact"""
        for I in ti.grouped(self.fields.grid_m):
            if self.fields.grid_m[I] > 1e-10:
                # Normalize momentum to get velocity
                self.fields.grid_v[I] /= self.fields.grid_m[I]

                # Apply gravity
                self.fields.grid_v[I] += self.dt * self.gravity

                # Boundary conditions (sticky walls)
                if ti.static(self.sticky_boundary):
                    w = ti.static(self.sticky_boundary_width)
                    for d in ti.static(range(3)):
                        if I[d] < w or I[d] >= self.fields.grid_size[d] - w:
                            self.fields.grid_v[I][d] = 0.0

                # Contact with SDF obstacles (configurable: plane/sphere/box/cylinder)
                if ti.static(self.enable_contact):
                    grid_x = ti.cast(I, ti.f32) * self.dx
                    any_contact = 0

                    # Iterate over all obstacles
                    for obs_idx in range(self.n_obstacles):
                        obs_type = self.obstacle_types[obs_idx]
                        obs_center = self.obstacle_centers[obs_idx]       
                        obs_normal = self.obstacle_normals[obs_idx]       
                        obs_half_ext = self.obstacle_half_extents[obs_idx] 
                        obs_vel = self.obstacle_velocities[obs_idx]

                        # Evaluate SDF for this obstacle
                        phi = evaluate_sdf(grid_x, obs_type, obs_center, obs_normal, obs_half_ext)

                        if phi < 0.0:
                            # In contact with this obstacle
                            normal = compute_sdf_normal(grid_x, obs_type, obs_center, obs_normal, obs_half_ext)
                            v_rel = self.fields.grid_v[I] - obs_vel

                            # Compute contact force with friction
                            f_contact, u_t_new, is_contact = compute_contact_force(
                                phi, v_rel, normal,
                                self.fields.grid_ut[I],
                                self.dt,
                                self.contact_stiffness_normal,
                                self.contact_stiffness_tangent,
                                self.mu_s, self.mu_k,
                                self.v_transition
                            )

                            # Apply contact force (impulse)
                            if self.fields.grid_m[I] > 1e-10:
                                self.fields.grid_v[I] += self.dt * f_contact / self.fields.grid_m[I]

                            # Update tangential displacement
                            self.fields.grid_ut[I] = u_t_new

                            # Mark as in contact
                            any_contact = 1

                    self.fields.grid_contact_mask[I] = any_contact

    @ti.kernel
    def g2p(self):
        """Grid to particle transfer (G2P)"""
        for p in range(self.n_particles):
            x_p = self.fields.x[p]

            # Grid node base index
            base = ti.cast(x_p * self.inv_dx - 0.5, ti.i32)
            fx = x_p * self.inv_dx - ti.cast(base, ti.f32)

            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

            # Gather from grid
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]

                grid_idx = base + offset

                if 0 <= grid_idx[0] < self.fields.grid_size[0] and \
                   0 <= grid_idx[1] < self.fields.grid_size[1] and \
                   0 <= grid_idx[2] < self.fields.grid_size[2]:

                    grid_v = self.fields.grid_v[grid_idx]
                    new_v += weight * grid_v
                    new_C += 4.0 * self.inv_dx * weight * grid_v.outer_product(dpos)

            # Update particle velocity and position
            self.fields.v[p] = new_v
            self.fields.x[p] += self.dt * new_v

            # Update APIC matrix
            self.fields.C[p] = new_C

    @ti.kernel
    def update_F_and_internal(self):
        """Update deformation gradient and internal variables"""
        for p in range(self.n_particles):
            # Save old F for viscous dissipation calculation
            F_old = self.fields.F[p]

            # Update deformation gradient: F_new = (I + dt * C) @ F_old
            self.fields.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * self.fields.C[p]) @ F_old

            # Clamp F to avoid extreme deformation
            J = self.fields.F[p].determinant()
            if J < 0.5:
                self.fields.F[p] *= ti.pow(0.5 / J, 1.0/3.0)
            elif J > 2.0:
                self.fields.F[p] *= ti.pow(2.0 / J, 1.0/3.0)

            # Update Maxwell internal variables and compute energy corrections
            if ti.static(self.n_maxwell > 0):
                F_new = self.fields.F[p]
                J_new = F_new.determinant()
                F_bar = ti.pow(J_new, -1.0/3.0) * F_new

                delta_E_proj_p = 0.0
                delta_E_visc_maxwell = 0.0

                for k in ti.static(range(self.n_maxwell)):
                    # Get old internal variable
                    b_bar_e_old = self.fields.b_bar_e[p, k]

                    # Upper-convected update
                    b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar.transpose()

                    # Relaxation
                    G_k = self.maxwell_G[k]
                    tau_k = self.maxwell_tau[k]
                    relax_factor = ti.exp(-self.dt / tau_k)
                    b_bar_e_relaxed = relax_factor * b_bar_e_trial + (1.0 - relax_factor) * ti.Matrix.identity(ti.f32, 3)

                    # Compute Maxwell viscous dissipation (before projection)
                    # Dissipation = G_k / tau_k * ||b_bar_e_trial - I||^2 * dt / 2
                    diff_relax = b_bar_e_trial - ti.Matrix.identity(ti.f32, 3)
                    delta_E_visc_k = 0.5 * G_k / tau_k * (
                        diff_relax[0,0]**2 + diff_relax[0,1]**2 + diff_relax[0,2]**2 +
                        diff_relax[1,0]**2 + diff_relax[1,1]**2 + diff_relax[1,2]**2 +
                        diff_relax[2,0]**2 + diff_relax[2,1]**2 + diff_relax[2,2]**2
                    ) * self.dt
                    delta_E_visc_maxwell += delta_E_visc_k

                    # SPD projection - use STE in autodiff mode if enabled
                    if ti.static(self.enable_grad and self.use_spd_ste):
                        b_bar_e_new = make_spd_ste(b_bar_e_relaxed, 1e-8)
                    else:
                        b_bar_e_new = make_spd(b_bar_e_relaxed, 1e-8)

                    # Enforce isochoric constraint
                    det_b = b_bar_e_new.determinant()
                    if det_b > 1e-10:
                        scale = ti.pow(det_b, -1.0/3.0)
                        b_bar_e_new = scale * b_bar_e_new

                    # Compute projection energy correction
                    diff_proj = b_bar_e_new - b_bar_e_relaxed
                    delta_E_proj_k = 0.5 * G_k * (
                        diff_proj[0,0]**2 + diff_proj[0,1]**2 + diff_proj[0,2]**2 +
                        diff_proj[1,0]**2 + diff_proj[1,1]**2 + diff_proj[1,2]**2 +
                        diff_proj[2,0]**2 + diff_proj[2,1]**2 + diff_proj[2,2]**2
                    )
                    delta_E_proj_p += delta_E_proj_k

                    # Update internal variable
                    self.fields.b_bar_e[p, k] = b_bar_e_new

                # Store energy corrections
                self.fields.delta_E_proj_step[p] = delta_E_proj_p
                self.fields.delta_E_viscous_step[p] += delta_E_visc_maxwell

            # Compute bulk viscosity dissipation if enabled
            if ti.static(self.config.material.enable_bulk_viscosity):
                J_old = F_old.determinant()
                J_new = self.fields.F[p].determinant()
                J_dot = (J_new - J_old) / self.dt
                vol_strain_rate = J_dot / J_new
                eta_bulk = self.config.material.bulk_viscosity
                delta_E_visc_bulk = eta_bulk * vol_strain_rate ** 2 * self.dt
                # Accumulate (not overwrite) bulk viscosity dissipation
                self.fields.delta_E_viscous_step[p] += delta_E_visc_bulk

    @ti.kernel
    def clear_energy_fields(self):
        """Clear global energy accumulators (separate kernel for autodiff compatibility)"""
        self.fields.E_kin[None] = 0.0
        self.fields.E_elastic[None] = 0.0
        self.fields.E_viscous_step[None] = 0.0
        self.fields.E_proj_step[None] = 0.0

    @ti.kernel
    def reduce_energies(self):
        """Reduce particle-level energies to global scalars"""
        # Accumulate from particles
        for p in range(self.n_particles):
            # Kinetic energy
            v_p = self.fields.v[p]
            m_p = self.fields.mass[p]
            self.fields.E_kin[None] += 0.5 * m_p * v_p.dot(v_p)

            # Elastic energy
            F_p = self.fields.F[p]
            V_p = self.fields.volume[p]
            _, psi = compute_ogden_stress_general(
                F_p,
                self.ogden_mu,
                self.ogden_alpha,
                self.n_ogden,
                self.ogden_kappa
            )
            self.fields.E_elastic[None] += psi * V_p

            # Viscous and projection energies
            self.fields.E_viscous_step[None] += self.fields.delta_E_viscous_step[p] * V_p
            self.fields.E_proj_step[None] += self.fields.delta_E_proj_step[p] * V_p

        # Update cumulative energies
        self.fields.E_viscous_cum[None] += self.fields.E_viscous_step[None]
        self.fields.E_proj_cum[None] += self.fields.E_proj_step[None]

    @ti.kernel
    def cleanup_ut(self):
        """Cleanup tangential displacement based on hysteresis counter"""
        for I in ti.grouped(self.fields.grid_ut):
            age_new, should_clear = update_contact_age(
                self.fields.grid_contact_mask[I],
                self.fields.grid_nocontact_age[I],
                self.K_clear
            )
            self.fields.grid_nocontact_age[I] = age_new

            if should_clear == 1:
                self.fields.grid_ut[I] = ti.Vector([0.0, 0.0, 0.0])       

    @ti.kernel
    def update_obstacle_velocities(self):
        """Update obstacle velocities from center delta (for moving obstacle friction)."""
        for i in range(self.n_obstacles):
            cur = self.obstacle_centers[i]
            prev = self.obstacle_centers_prev[i]
            self.obstacle_velocities[i] = (cur - prev) / self.dt
            self.obstacle_centers_prev[i] = cur

    def step(self) -> None:
        """Execute one simulation step"""
        self.fields.clear_grid()
        self.fields.clear_particle_energy_increments()
        self.fields.clear_global_energy_step()

        self.update_obstacle_velocities()
        self.p2g()
        self.grid_op()
        self.g2p()
        self.update_F_and_internal()  # Now handles both STE/non-STE via ti.static
        self.clear_energy_fields()  # Clear energy accumulators (autodiff-safe)
        self.reduce_energies()  # Accumulate energies from particles
        self.cleanup_ut()

        self.current_step += 1

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run simulation for specified number of steps"""
        if num_steps is None:
            num_steps = self.config.time.num_steps

        for _ in range(num_steps):
            self.step()

    def reset_loss(self) -> None:
        """Reset loss field to zero. Call before starting a new autodiff pass."""
        self.loss_field[None] = 0.0

    def reset_gradients(self) -> None:
        """Reset all gradient fields to zero. Call before starting a new autodiff pass."""
        if not self.enable_grad:
            return

        # Reset loss gradient
        self.loss_field.grad[None] = 0.0

        # Reset material parameter gradients
        for i in range(4):
            self.ogden_mu.grad[i] = 0.0
            self.ogden_alpha.grad[i] = 0.0

        if self.n_maxwell > 0:
            for k in range(self.n_maxwell):
                self.maxwell_G.grad[k] = 0.0
                self.maxwell_tau.grad[k] = 0.0

        # Reset particle field gradients
        self.fields.reset_gradients()

    def get_particle_data(self) -> Dict[str, NDArray[np.float32]]:
        """Get current particle data"""
        return self.fields.get_particle_data()

    def get_energy_data(self) -> Dict[str, float]:
        """Get current energy data"""
        return self.fields.get_energy_data()
# ===== END FILE: xengym/mpm/mpm_solver.py =====
