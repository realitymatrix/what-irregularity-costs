//! The one interface every TSDF fusion arm implements.
//!
//! Nothing here is copied from another codebase. Prior implementations were
//! studied for algorithm and API shape only; every line is written for this
//! project.
//!
//! Arms:
//!   A1  open3d        Open3D VoxelBlockGrid behind a C shim. Third-party baseline.
//!   A2  cpu           our C++17/20 CPU TSDF.
//!   A3  cuda          our CUDA C++ TSDF.
//!   A4  rust-cuda     our Rust CUDA TSDF via cuda-oxide.
//!   A5a triton-shared Triton per-block update, insertion shared with A3.
//!   A5b triton-full   Triton does insertion too.
//!
//! # Correctness without a code-derived oracle
//!
//! There is deliberately no "reference implementation" arm. Certifying ports
//! against the implementation they were ported from is weakly circular: a
//! shared misunderstanding of the algorithm passes the gate. Instead the gate
//! triangulates across three independent sources of truth, in decreasing order
//! of authority (see the `tsdf-harness` crate):
//!
//!   1. **Analytic.** Synthetic scenes whose surface is known in closed form
//!      (plane, sphere). Independent of every implementation, including ours.
//!   2. **Third-party.** Open3D's `VoxelBlockGrid` on identical input. An
//!      independent implementation by other authors.
//!   3. **Cross-arm.** All arms must agree with each other. Catches per-arm
//!      bugs, but cannot catch a mistake common to all of them, which is why it
//!      ranks last.
//!
//! Tier 1 is what makes this stronger than a single-oracle design: no
//! implementation, ours or anyone else's, gets to define what "correct" means.

/// Which implementation to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arm {
    /// A1: Open3D VoxelBlockGrid. Third-party baseline and cross-check.
    Open3d,
    /// A2: our C++ CPU implementation.
    Cpu,
    /// A3: our CUDA C++ implementation.
    Cuda,
    /// A4: our Rust CUDA implementation (cuda-oxide).
    RustCuda,
    /// A5a: Triton per-block update, hash insertion shared with the CUDA arm.
    /// The clean update-only comparison against A3.
    TritonShared,
    /// A5b: Triton does the irregular hash insertion too. Isolates the
    /// control-flow tax: no per-lane early exit and no mask on `atomic_cas`.
    TritonFull,
}

impl Arm {
    pub const ALL: [Arm; 6] = [
        Arm::Open3d,
        Arm::Cpu,
        Arm::Cuda,
        Arm::RustCuda,
        Arm::TritonShared,
        Arm::TritonFull,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Arm::Open3d => "open3d",
            Arm::Cpu => "cpu",
            Arm::Cuda => "cuda",
            Arm::RustCuda => "rust-cuda",
            Arm::TritonShared => "triton-shared",
            Arm::TritonFull => "triton-full",
        }
    }

    /// True for arms that execute on the GPU. The CPU arm is expected to be
    /// orders of magnitude slower and is reported on its own axis rather than
    /// alongside the GPU arms, so plots are not dominated by it.
    pub fn is_gpu(self) -> bool {
        !matches!(self, Arm::Cpu)
    }

    /// True for arms authored in this project. Open3D is not, so it is a
    /// baseline rather than a subject of the language comparison.
    pub fn is_ours(self) -> bool {
        !matches!(self, Arm::Open3d)
    }
}

/// Construction parameters.
#[derive(Debug, Clone, Copy)]
pub struct VolumeConfig {
    pub voxel_size_m: f32,
    /// Voxels per side of a block. 8 gives 512 voxels per block, small enough
    /// that the hash table stays cheap while still amortising each lookup over
    /// a useful number of voxels.
    pub block_dim: i32,
    pub pool_capacity_blocks: i32,
    /// TSDF truncation half-width in metres. <= 0 selects 4 x voxel_size.
    pub trunc_m: f32,
}

impl VolumeConfig {
    /// Device bytes the block pool will consume.
    ///
    /// Worth computing before allocating: an oversized pool fails as a bare
    /// out-of-memory at construction, with nothing naming the pool as the
    /// cause. At 8^3 blocks across 8 f32/u32 fields that is 16 KiB per block,
    /// so a nominal-looking 1<<20 blocks is 16 GiB.
    pub fn pool_bytes(&self) -> u64 {
        const FIELDS: u64 = 8;
        let bd = self.block_dim as u64;
        self.pool_capacity_blocks as u64 * bd * bd * bd * FIELDS * 4
    }

    pub fn trunc_or_default(&self) -> f32 {
        if self.trunc_m > 0.0 {
            self.trunc_m
        } else {
            4.0 * self.voxel_size_m
        }
    }
}

impl Default for VolumeConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.01,
            block_dim: 8,
            // 65_536 blocks = 1 GiB at the default block_dim, which covers a
            // room-scale scan with headroom. Scale up deliberately, after
            // checking `pool_bytes()` against the device.
            pool_capacity_blocks: 1 << 16,
            trunc_m: -1.0,
        }
    }
}

/// One batch of world-space points to integrate.
///
/// World-space, deliberately: pose composition happens once, in the loader, so
/// it cannot differ between arms.
#[derive(Debug, Clone, Copy)]
pub struct PointBatch {
    /// Device pointer to 3 * n_points f32, interleaved xyz.
    pub d_positions: u64,
    /// Device pointer to 3 * n_points u8 rgb, or 0 for neutral grey.
    pub d_colors: u64,
    /// Device pointer to n_points f32 weights, or 0 for uniform 1.0.
    pub d_weights: u64,
    pub n_points: i32,
    /// Monotonic chunk id, for LRU eviction stamping.
    pub chunk_id: i32,
    /// Camera origin. Required for projective TSDF: the sign of the distance
    /// depends on which side of the surface a sample lies, which is only
    /// defined relative to a viewpoint.
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

    /// Positions only, which is what the mesh metrics consume.
    pub fn positions(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.n_vertices * 3);
        for i in 0..self.n_vertices {
            v.extend_from_slice(&self.posnor[i * 6..i * 6 + 3]);
        }
        v
    }
}

/// The one interface every arm implements.
///
/// Intentionally narrow. Allocation policy, eviction and streaming are
/// implementation details, because the moment arms differ in what the harness
/// calls, the comparison stops being apples to apples.
pub trait TsdfBackend {
    fn arm(&self) -> Arm;

    /// Integrate one batch. Points are already on device.
    fn integrate(&mut self, batch: &PointBatch) -> Result<(), TsdfError>;

    /// Marching cubes over the current volume.
    fn extract_mesh(&mut self, min_weight: f32, iso: f32) -> Result<Mesh, TsdfError>;

    /// Allocated block count. A cheap cross-arm invariant: arms that disagree
    /// here have diverged before any mesh is compared.
    fn block_count(&self) -> i32;

    /// Points dropped because the block pool was exhausted. Must be reported,
    /// never silently ignored: a saturating pool presents as a quality
    /// regression when it is a capacity problem.
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
    #[error("{arm}: allocation failed")]
    Alloc { arm: &'static str },
}
