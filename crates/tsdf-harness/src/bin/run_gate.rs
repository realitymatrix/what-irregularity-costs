//! Run the correctness gate against arm A0 on real TartanAir data.
//!
//! This is the first end-to-end path: TartanAir depth -> world points ->
//! device -> A0 integrate -> marching cubes -> mesh agreement.
//!
//! Right now A0 is the only wired arm, so the gate runs A0 against A0 across
//! two independent volumes. That is not a tautology, it checks three things
//! that must hold before any other arm can be trusted against it:
//!
//!   * **Determinism.** Two volumes fed identical points in identical order
//!     must produce identical meshes. A reference that varies run to run cannot
//!     be an oracle, and atomics-heavy integrate is exactly where nondeterminism
//!     would come from (float atomicAdd is not associative).
//!   * **The gate's plumbing works on real geometry**, not just the analytic
//!     spheres in `verify_metrics`. Real meshes are open, non-manifold triangle
//!     soups with far more vertices.
//!   * **The pool does not silently saturate.** `drop_count` must be zero, or
//!     the mesh is missing geometry for a capacity reason that would otherwise
//!     read as a quality regression.
//!
//! Usage: run_gate <trajectory_dir> [n_frames]

use std::path::PathBuf;

use cuda_min::{CudaContext, DeviceBuffer};
use mesh_metrics::{gate, Tolerance, TriMesh};
use tartanair::{backproject, Trajectory};
use tsdf_ffi::{PointBatch, ReferenceBackend, TsdfBackend, VolumeConfig};

/// Far-field cutoff. At this baseline (fx*B = 80) 1 px of disparity error costs
/// 31 cm at 5 m, so points beyond a few metres carry error exceeding any
/// sensible voxel and only consume block-pool slots.
const RADIUS_M: f32 = 4.0;
const VOXEL_M: f32 = 0.02;

fn build_volume(
    traj: &Trajectory,
    n_frames: usize,
    cfg: &VolumeConfig,
) -> Result<(ReferenceBackend, usize), Box<dyn std::error::Error>> {
    let mut vol = ReferenceBackend::new(cfg)?;
    let mut total_pts = 0usize;

    for i in 0..n_frames {
        let depth = traj.depth(i)?;
        let pose = traj.poses_left[i];
        let pc = backproject(&depth, None, &pose, RADIUS_M);
        if pc.n == 0 {
            continue;
        }
        total_pts += pc.n;

        // Device upload. The reference takes device pointers, which is also the
        // path the depth stage will use later (libinfer infer_device_io), so no
        // host round-trip is measured.
        let d_pos = DeviceBuffer::from_slice(&pc.xyz)?;
        let cam = [pose.t[0] as f32, pose.t[1] as f32, pose.t[2] as f32];

        vol.integrate(&PointBatch {
            d_positions: d_pos.ptr(),
            d_colors: 0,
            d_weights: 0,
            n_points: pc.n as i32,
            chunk_id: i as i32,
            cam,
            radius_m: RADIUS_M,
        })?;
    }
    Ok((vol, total_pts))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: run_gate <trajectory_dir> [n_frames]")?;
    let n_frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("=== correctness gate: arm A0 (reference) on real data ===");
    let _ctx = CudaContext::new(0)?;
    let traj = Trajectory::open(&dir, "front")?;
    let n_frames = n_frames.min(traj.n_frames);
    println!("  {}", traj.root.display());
    println!("  {n_frames} frames, voxel {VOXEL_M} m, radius gate {RADIUS_M} m\n");

    let cfg = VolumeConfig {
        voxel_size_m: VOXEL_M,
        ..Default::default()
    };

    // Two independent volumes over identical input.
    let (mut a, n_pts) = build_volume(&traj, n_frames, &cfg)?;
    let (mut b, _) = build_volume(&traj, n_frames, &cfg)?;

    let mesh_a = a.extract_mesh(1.0, 0.0)?;
    let mesh_b = b.extract_mesh(1.0, 0.0)?;

    println!("  points integrated  {n_pts}");
    println!("  blocks allocated   {} / {}", a.block_count(), b.block_count());
    println!("  points dropped     {} / {}", a.drop_count(), b.drop_count());
    println!(
        "  mesh               {} verts, {} tris\n",
        mesh_a.n_vertices,
        mesh_a.n_triangles()
    );

    let mut ok = true;
    let mut check = |label: &str, pass: bool, detail: String| {
        ok &= pass;
        println!("  [{}] {label}  {detail}", if pass { "PASS" } else { "FAIL" });
    };

    check(
        "produced geometry",
        mesh_a.n_vertices > 0,
        format!("{} vertices", mesh_a.n_vertices),
    );
    check(
        "block pool not saturated",
        a.drop_count() == 0,
        format!("{} points dropped", a.drop_count()),
    );
    check(
        "block count deterministic",
        a.block_count() == b.block_count(),
        format!("{} vs {}", a.block_count(), b.block_count()),
    );

    // Determinism, characterised rather than assumed.
    //
    // Marching cubes claims output slots with an atomicAdd on a shared counter,
    // so the *order* of the triangle soup varies between runs even when the
    // geometry does not. Comparing the arrays element-wise therefore reports a
    // difference that is not a geometric one. Distinguish the two: sort both
    // vertex sets and compare bitwise. Equal sorted sets mean pure reordering;
    // unequal means the values themselves moved, which would implicate
    // non-associative float atomicAdd in the integrate path and would force the
    // gate onto tolerance rather than exact agreement.
    let same_len = mesh_a.posnor.len() == mesh_b.posnor.len();
    let in_order = same_len
        && mesh_a
            .posnor
            .iter()
            .zip(&mesh_b.posnor)
            .all(|(x, y)| x.to_bits() == y.to_bits());

    let sorted_equal = same_len && {
        let key = |m: &tsdf_ffi::Mesh| {
            let mut v: Vec<[u32; 3]> = (0..m.n_vertices)
                .map(|i| {
                    [
                        m.posnor[i * 6].to_bits(),
                        m.posnor[i * 6 + 1].to_bits(),
                        m.posnor[i * 6 + 2].to_bits(),
                    ]
                })
                .collect();
            v.sort_unstable();
            v
        };
        key(&mesh_a) == key(&mesh_b)
    };

    check(
        "vertex count deterministic",
        same_len,
        format!("{} vs {}", mesh_a.n_vertices, mesh_b.n_vertices),
    );
    check(
        "vertex VALUES deterministic (order-independent)",
        sorted_equal,
        if sorted_equal {
            format!(
                "identical vertex set; emission order {}",
                if in_order { "also stable" } else { "varies (atomicAdd slot claim)" }
            )
        } else {
            "values differ, not just order: gate must use tolerance".to_string()
        },
    );

    // The metric on real geometry.
    let ta = TriMesh { xyz: &positions(&mesh_a) };
    let tb = TriMesh { xyz: &positions(&mesh_b) };
    let tol = Tolerance::for_voxel(VOXEL_M as f64);
    let g = gate(&ta, &tb, tol, 50_000, VOXEL_M as f64 * 2.0);

    println!(
        "\n  mean surface {:.9} m | hausdorff {:.9} m | p99 {:.9} m",
        g.agreement.mean_surface, g.agreement.hausdorff, g.agreement.hausdorff_p99
    );
    check(
        "gate passes A0 vs A0",
        g.passed,
        if g.passed {
            format!(
                "tolerance mean {:.4} m, p99 {:.4} m",
                tol.mean_surface_m, tol.hausdorff_p99_m
            )
        } else {
            format!("{:?}", g.failures)
        },
    );

    println!("\n=== {} ===", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

/// Strip normals: the mesh is (pos3, nor3) interleaved, the metric wants xyz.
fn positions(m: &tsdf_ffi::Mesh) -> Vec<f32> {
    let mut v = Vec::with_capacity(m.n_vertices * 3);
    for i in 0..m.n_vertices {
        v.extend_from_slice(&m.posnor[i * 6..i * 6 + 3]);
    }
    v
}
