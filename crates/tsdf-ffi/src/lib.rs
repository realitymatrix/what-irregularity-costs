//! C ABI bindings to every TSDF fusion arm.
//!
//! Every arm in this project exposes the same `extern "C"` surface, modelled on
//! the existing hash TSDF in `openstrate-reconstruct-rs/vendor/libinfer/src/`
//! (`tsdf_hash.h`). That header is already a pure C ABI with no Rust types, no
//! ARCore state and no TensorRT, so the golden reference needs a binding here
//! rather than a reimplementation, and stays bit-comparable.
//!
//! Arms:
//!   A0  reference  the existing Rust/CUDA hash TSDF. Golden, not a language arm.
//!   A1  open3d     Open3D VoxelBlockGrid behind a C shim.
//!   A2  cpu        our own C++17/20 CPU TSDF.
//!   A3  cuda       our own CUDA C++ TSDF.
//!   A4  rust       our own Rust CUDA TSDF via cuda-oxide.
//!   A5  triton     Triton kernels, AOT-compiled to cubin, launched from Rust.
//!
//! The point of one ABI is that the harness holds `Box<dyn TsdfBackend>` and
//! cannot accidentally measure a different code path per arm.

use std::ffi::c_void;

/// Which implementation to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arm {
    /// A0: the existing Rust/CUDA implementation. Correctness reference.
    Reference,
    /// A1: Open3D VoxelBlockGrid.
    Open3d,
    /// A2: our own C++ CPU implementation.
    Cpu,
    /// A3: our own CUDA C++ implementation.
    Cuda,
    /// A4: our own Rust CUDA implementation (cuda-oxide).
    RustCuda,
    /// A5: Triton kernels launched from Rust.
    Triton,
}

impl Arm {
    pub const ALL: [Arm; 6] = [
        Arm::Reference,
        Arm::Open3d,
        Arm::Cpu,
        Arm::Cuda,
        Arm::RustCuda,
        Arm::Triton,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Arm::Reference => "reference",
            Arm::Open3d => "open3d",
            Arm::Cpu => "cpu",
            Arm::Cuda => "cuda",
            Arm::RustCuda => "rust-cuda",
            Arm::Triton => "triton",
        }
    }

    /// True for arms that execute on the GPU. The CPU arm is expected to be
    /// orders of magnitude slower and is reported on its own axis rather than
    /// alongside the GPU arms, so plots are not dominated by it.
    pub fn is_gpu(self) -> bool {
        !matches!(self, Arm::Cpu)
    }
}

/// Construction parameters, mirroring `tsdf_hash_create`.
#[derive(Debug, Clone, Copy)]
pub struct VolumeConfig {
    pub voxel_size_m: f32,
    pub pool_capacity_blocks: i32,
    pub committed_cap_points: i32,
    /// TSDF truncation half-width. <= 0 selects 4 x voxel_size, matching the
    /// reference's `tsdf_hash_set_mesh_mode` contract.
    pub trunc_m: f32,
}

impl Default for VolumeConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.01,
            pool_capacity_blocks: 1 << 20,
            committed_cap_points: 1 << 22,
            trunc_m: -1.0,
        }
    }
}

/// One batch of world-space points to integrate.
///
/// World-space, deliberately: the reference API takes world points, so pose
/// composition never enters the volume and cannot differ between arms.
#[derive(Debug, Clone, Copy)]
pub struct PointBatch {
    /// Device pointer to 3 * n_points f32, interleaved xyz.
    pub d_positions: u64,
    /// Device pointer to 3 * n_points u8 rgb, or 0 for neutral grey.
    pub d_colors: u64,
    /// Device pointer to n_points f32 weights, or 0 for uniform 1.0.
    pub d_weights: u64,
    pub n_points: i32,
    /// Monotonic chunk id, used for LRU eviction stamping.
    pub chunk_id: i32,
    /// Camera origin for the radius gate.
    pub cam: [f32; 3],
    /// Far-field cutoff in metres. <= 0 disables.
    ///
    /// Not cosmetic: at the TartanAir baseline (fx*B = 80), 1 px of disparity
    /// error costs 31 cm at 5 m, so beyond a few metres depth error exceeds any
    /// sensible voxel and the points only burn pool slots. See docs/DATASET.md.
    pub radius_m: f32,
}

/// A triangle soup plus per-vertex colour, as returned by every arm.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    /// 6 floats per vertex: position xyz then normal xyz.
    pub posnor: Vec<f32>,
    /// 3 bytes per vertex.
    pub rgb: Vec<u8>,
    pub n_vertices: usize,
}

impl Mesh {
    pub fn n_triangles(&self) -> usize {
        self.n_vertices / 3
    }

    pub fn positions(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        (0..self.n_vertices).map(move |i| {
            [
                self.posnor[i * 6],
                self.posnor[i * 6 + 1],
                self.posnor[i * 6 + 2],
            ]
        })
    }
}

/// The one interface every arm implements.
///
/// Intentionally narrow. Anything an arm needs beyond this (allocation policy,
/// eviction, streaming) is an implementation detail, because the moment arms
/// differ in what the harness calls, the comparison stops being apples to
/// apples.
pub trait TsdfBackend {
    fn arm(&self) -> Arm;

    /// Integrate one batch. Points are already on device.
    fn integrate(&mut self, batch: &PointBatch) -> Result<(), TsdfError>;

    /// Marching cubes over the current volume.
    fn extract_mesh(&mut self, min_weight: f32, iso: f32) -> Result<Mesh, TsdfError>;

    /// Allocated block count. Diagnostic, and a cheap cross-arm invariant: two
    /// arms that disagree here have diverged before any mesh is compared.
    fn block_count(&self) -> i32;

    /// Points dropped because the block pool was exhausted. Must be reported,
    /// never silently ignored: a saturating pool looks like a quality
    /// regression rather than a capacity problem.
    fn drop_count(&self) -> u64;
}

#[derive(Debug, thiserror::Error)]
pub enum TsdfError {
    #[error("{arm} backend not yet implemented")]
    NotImplemented { arm: &'static str },
    #[error("{arm}: {op} failed with code {code}")]
    Ffi {
        arm: &'static str,
        op: &'static str,
        code: i32,
    },
    #[error("{arm}: output buffer too small, needed more than {cap} vertices")]
    BufferTooSmall { arm: &'static str, cap: i32 },
    #[error("{arm}: null handle returned from create")]
    NullHandle { arm: &'static str },
}

// ---------------------------------------------------------------------------
// A0: the golden reference, bound to the existing C ABI verbatim.
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct TsdfHash {
    _private: [u8; 0],
}

// Signatures copied from vendor/libinfer/src/tsdf_hash.h. Kept verbatim so the
// reference is bit-comparable rather than merely similar.
extern "C" {
    pub fn tsdf_hash_create(
        voxel_size_m: f32,
        pool_capacity_blocks: i32,
        committed_cap_points: i32,
    ) -> *mut TsdfHash;

    pub fn tsdf_hash_destroy(h: *mut TsdfHash);

    pub fn tsdf_hash_set_mesh_mode(h: *mut TsdfHash, on: i32, trunc_m: f32) -> i32;

    pub fn tsdf_hash_add_points_chunk_device(
        h: *mut TsdfHash,
        d_positions: u64,
        d_colors: u64,
        d_weights: u64,
        n_points: i32,
        chunk_id: i32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_m: f32,
    ) -> i32;

    pub fn tsdf_hash_extract_mesh(
        h: *mut TsdfHash,
        buffer_posnor: *mut f32,
        buffer_rgb: *mut u8,
        buffer_block: *mut i32,
        vert_cap: i32,
        min_weight: f32,
        iso: f32,
        dirty_only: i32,
        current_chunk: i32,
    ) -> i32;

    pub fn tsdf_hash_block_count(h: *mut TsdfHash) -> i32;
    pub fn tsdf_hash_drop_count(h: *mut TsdfHash) -> u64;
}

/// A0. Owns the C handle and frees it on drop.
pub struct ReferenceBackend {
    handle: *mut TsdfHash,
    current_chunk: i32,
    vert_cap: i32,
}

impl ReferenceBackend {
    pub fn new(cfg: &VolumeConfig) -> Result<Self, TsdfError> {
        // SAFETY: plain scalars in, opaque handle out. Null signals allocation
        // failure, which is checked below.
        let handle = unsafe {
            tsdf_hash_create(
                cfg.voxel_size_m,
                cfg.pool_capacity_blocks,
                cfg.committed_cap_points,
            )
        };
        if handle.is_null() {
            return Err(TsdfError::NullHandle { arm: "reference" });
        }
        // Mesh mode is required: without it, integrate does centroid binning
        // instead of projective TSDF splatting and marching cubes has no
        // signed field to walk.
        let rc = unsafe { tsdf_hash_set_mesh_mode(handle, 1, cfg.trunc_m) };
        if rc != 0 {
            unsafe { tsdf_hash_destroy(handle) };
            return Err(TsdfError::Ffi {
                arm: "reference",
                op: "set_mesh_mode",
                code: rc,
            });
        }
        Ok(Self {
            handle,
            current_chunk: 0,
            vert_cap: 1 << 22,
        })
    }
}

impl Drop for ReferenceBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle came from tsdf_hash_create and is freed once.
            unsafe { tsdf_hash_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl TsdfBackend for ReferenceBackend {
    fn arm(&self) -> Arm {
        Arm::Reference
    }

    fn integrate(&mut self, b: &PointBatch) -> Result<(), TsdfError> {
        self.current_chunk = b.chunk_id;
        // SAFETY: device pointers are caller-owned and must cover n_points as
        // documented on PointBatch; 0 is the documented null for colors/weights.
        let rc = unsafe {
            tsdf_hash_add_points_chunk_device(
                self.handle,
                b.d_positions,
                b.d_colors,
                b.d_weights,
                b.n_points,
                b.chunk_id,
                b.cam[0],
                b.cam[1],
                b.cam[2],
                b.radius_m,
            )
        };
        if rc != 0 {
            return Err(TsdfError::Ffi {
                arm: "reference",
                op: "add_points_chunk_device",
                code: rc,
            });
        }
        Ok(())
    }

    fn extract_mesh(&mut self, min_weight: f32, iso: f32) -> Result<Mesh, TsdfError> {
        let cap = self.vert_cap as usize;
        let mut posnor = vec![0.0f32; cap * 6];
        let mut rgb = vec![0u8; cap * 3];
        let mut block = vec![0i32; cap];

        // SAFETY: buffers are sized to vert_cap as the C API requires. A
        // negative return means overflow, not a partial write.
        let n = unsafe {
            tsdf_hash_extract_mesh(
                self.handle,
                posnor.as_mut_ptr(),
                rgb.as_mut_ptr(),
                block.as_mut_ptr(),
                self.vert_cap,
                min_weight,
                iso,
                0, // dirty_only = 0: full re-mesh, so arms are comparable
                self.current_chunk,
            )
        };
        if n < 0 {
            return Err(TsdfError::BufferTooSmall {
                arm: "reference",
                cap: self.vert_cap,
            });
        }
        let n = n as usize;
        posnor.truncate(n * 6);
        rgb.truncate(n * 3);
        Ok(Mesh {
            posnor,
            rgb,
            n_vertices: n,
        })
    }

    fn block_count(&self) -> i32 {
        // SAFETY: read-only query on a live handle.
        unsafe { tsdf_hash_block_count(self.handle) }
    }

    fn drop_count(&self) -> u64 {
        // SAFETY: read-only query on a live handle.
        unsafe { tsdf_hash_drop_count(self.handle) }
    }
}

/// Suppress the unused-import warning until the arms that need raw pointers land.
const _: Option<*mut c_void> = None;
