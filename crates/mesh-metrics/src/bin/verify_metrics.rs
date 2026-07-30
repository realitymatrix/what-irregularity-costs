//! Cross-validate the mesh metrics against Open3D, and check the gate bites.
//!
//! Two things have to be true for the gate to be worth anything:
//!   1. The distances agree with an independent implementation (Open3D).
//!   2. The gate FAILS on a mesh that is wrong by more than tolerance.
//!
//! (2) matters as much as (1). A gate that always passes is worse than no gate,
//! because it produces false confidence in every timing number downstream.
//!
//! Emits PLY files and a small JSON of its own results so
//! `tools/verify_metrics_open3d.py` can compare against Open3D on identical
//! input.
//!
//! Usage: verify_metrics <outdir>

use std::path::PathBuf;

use mesh_metrics::{compare, gate, write_ply, Tolerance, TriMesh};

/// Unit sphere as a triangle soup, via icosphere-free lat/long tessellation.
/// Deterministic, and analytically known, so displacing it by `d` gives an
/// exact expected surface distance of `d`.
fn sphere(radius: f64, n_lat: usize, n_lon: usize, offset: [f64; 3]) -> Vec<f32> {
    let mut v = Vec::new();
    let p = |lat: usize, lon: usize| -> [f64; 3] {
        let theta = std::f64::consts::PI * lat as f64 / n_lat as f64;
        let phi = 2.0 * std::f64::consts::PI * lon as f64 / n_lon as f64;
        [
            offset[0] + radius * theta.sin() * phi.cos(),
            offset[1] + radius * theta.sin() * phi.sin(),
            offset[2] + radius * theta.cos(),
        ]
    };
    let push = |a: [f64; 3], b: [f64; 3], c: [f64; 3], v: &mut Vec<f32>| {
        for q in [a, b, c] {
            v.push(q[0] as f32);
            v.push(q[1] as f32);
            v.push(q[2] as f32);
        }
    };
    for lat in 0..n_lat {
        for lon in 0..n_lon {
            let (a, b, c, d) = (
                p(lat, lon),
                p(lat + 1, lon),
                p(lat + 1, lon + 1),
                p(lat, lon + 1),
            );
            push(a, b, c, &mut v);
            push(a, c, d, &mut v);
        }
    }
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/metrics"));
    std::fs::create_dir_all(&out)?;

    println!("=== mesh-metrics validation ===");
    let n = 60_000;
    let cell = 0.02;

    // Analytic case: two concentric spheres differing by exactly 0.05 in
    // radius. Every point on one is 0.05 from the other, so mean surface
    // distance must be 0.05 and Hausdorff must be 0.05. This is the check that
    // the metric measures what it claims, independent of Open3D.
    let a = sphere(1.00, 64, 128, [0.0, 0.0, 0.0]);
    let b = sphere(1.05, 64, 128, [0.0, 0.0, 0.0]);
    let (ma, mb) = (TriMesh { xyz: &a }, TriMesh { xyz: &b });
    let ag = compare(&ma, &mb, n, cell, 0xC0FFEE);

    let mut ok = true;
    let mut check = |label: &str, pass: bool, detail: String| {
        ok &= pass;
        println!("  [{}] {label}  {detail}", if pass { "PASS" } else { "FAIL" });
    };

    check(
        "analytic: concentric spheres dr=0.05",
        (ag.mean_surface - 0.05).abs() < 1e-3,
        format!(
            "mean {:.6} (exact 0.05), hausdorff {:.6}, p99 {:.6}",
            ag.mean_surface, ag.hausdorff, ag.hausdorff_p99
        ),
    );

    // Identical meshes must agree to ~sampling resolution, not exactly: the two
    // sides are sampled with different seeds on purpose, so a nonzero floor is
    // expected and is the metric's noise floor.
    let ag_same = compare(&ma, &ma, n, cell, 0xC0FFEE);
    check(
        "identical meshes -> near zero",
        ag_same.mean_surface < 2e-3,
        format!(
            "mean {:.6}, hausdorff {:.6} (sampling floor)",
            ag_same.mean_surface, ag_same.hausdorff
        ),
    );

    // The gate must PASS on identical geometry ...
    let tol = Tolerance::for_voxel(0.01);
    let g_same = gate(&ma, &ma, tol, n, cell);
    check(
        "gate passes identical mesh",
        g_same.passed,
        format!("mean {:.6} m", g_same.agreement.mean_surface),
    );

    // ... and FAIL on geometry wrong by 5x the voxel. A gate that cannot fail
    // is worse than no gate.
    let g_diff = gate(&ma, &mb, tol, n, cell);
    check(
        "gate fails 0.05 m displacement at 0.01 m voxel",
        !g_diff.passed,
        format!("{} failure(s): {:?}", g_diff.failures.len(), g_diff.failures),
    );

    // A subtle case: a small offset just over tolerance. Catches a gate whose
    // thresholds are so loose that only gross errors trip it.
    let c = sphere(1.00, 64, 128, [0.012, 0.0, 0.0]);
    let mc = TriMesh { xyz: &c };
    let g_small = gate(&ma, &mc, tol, n, cell);
    check(
        "gate fails 12 mm translation at 0.01 m voxel",
        !g_small.passed,
        format!("mean {:.6} m, p99 {:.6} m", g_small.agreement.mean_surface, g_small.agreement.hausdorff_p99),
    );

    // Emit inputs and our numbers for the Open3D cross-check.
    write_ply(&out.join("sphere_a.ply"), &ma)?;
    write_ply(&out.join("sphere_b.ply"), &mb)?;
    std::fs::write(
        out.join("rust_metrics.json"),
        format!(
            "{{\n  \"mean_surface\": {},\n  \"hausdorff\": {},\n  \"hausdorff_p99\": {},\n  \"a_to_b_mean\": {},\n  \"b_to_a_mean\": {},\n  \"n_samples\": {}\n}}\n",
            ag.mean_surface, ag.hausdorff, ag.hausdorff_p99, ag.a_to_b.mean, ag.b_to_a.mean, n
        ),
    )?;
    println!("\n  wrote PLYs + rust_metrics.json to {}", out.display());
    println!("  cross-check with: python tools/verify_metrics_open3d.py {}", out.display());

    println!("\n=== {} ===", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
