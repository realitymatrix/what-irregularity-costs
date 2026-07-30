//! Cross-validate the Rust TartanAir loader against the Python reference.
//!
//! The reference values below were produced by cv2 + scipy on the same frame
//! (see tools/tartanair.py). They are hard-coded on purpose: the point is to
//! catch a divergence between the two implementations, so recomputing them here
//! with the same code under test would defeat it.
//!
//! Depth decoding in particular fails silently when the byte order is wrong.
//! Measured, by decoding in file order `[0,1,2,3]` instead of `[2,1,0,3]`: the
//! values land at 2.0000055 m, which is plausibly scaled, AND the median
//! gradient is 0.0015 m, which passes the smoothness check. Only the exact
//! reference comparison catches that permutation. Hence hard-coded values here
//! rather than self-consistency checks.
//!
//! Usage: verify_loader <trajectory_dir>

use std::path::PathBuf;
use tartanair::{backproject, Trajectory, BASELINE_M, CX, CY, FX, HEIGHT, WIDTH};

/// From cv2 `.view("<f4")` on frame 0 of RetroOffice/Data_easy/P000.
const REF_DEPTH_00: f32 = 2.359375;
const REF_DEPTH_320_320: f32 = 3.71875;

/// From the Python backprojection: `R(q) @ NED_R_cam @ [x,y,Z] + t`.
const REF_WORLD: &[(usize, usize, [f64; 3])] = &[
    (0, 0, [3.455864, 4.104830, -0.946016]),
    (320, 320, [0.755493, 4.454894, 1.413359]),
    (639, 639, [1.224518, 1.418045, 2.145439]),
    (100, 500, [2.711212, 3.520484, 2.597685]),
];
const REF_CENTROID: [f64; 3] = [1.748159, 3.203392, 1.127373];
const REF_N_POINTS: usize = 409600;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).map(PathBuf::from).ok_or(
        "usage: verify_loader <trajectory_dir>",
    )?;

    println!("=== TartanAir loader cross-validation (Rust vs Python reference) ===");
    let traj = Trajectory::open(&dir, "front")?;
    println!("  {} ({} frames)\n", traj.root.display(), traj.n_frames);

    let mut ok = true;
    let mut check = |label: &str, pass: bool, detail: String| {
        ok &= pass;
        println!("  [{}] {label}  {detail}", if pass { "PASS" } else { "FAIL" });
    };

    // -- depth decode ------------------------------------------------------
    let depth = traj.depth(0)?;
    check(
        "depth size",
        depth.data.len() == WIDTH * HEIGHT,
        format!("{} values", depth.data.len()),
    );
    let d00 = depth.at(0, 0);
    let dcc = depth.at(320, 320);
    check(
        "depth[0,0] exact",
        d00 == REF_DEPTH_00,
        format!("{d00} (ref {REF_DEPTH_00})"),
    );
    check(
        "depth[320,320] exact",
        dcc == REF_DEPTH_320_320,
        format!("{dcc} (ref {REF_DEPTH_320_320})"),
    );
    let grad = depth.median_abs_gradient();
    check(
        "depth is smooth (not byte-permuted)",
        depth.verify_smooth(),
        format!("median |dZ/dx| = {grad:.4} m"),
    );

    // -- intrinsics --------------------------------------------------------
    check(
        "intrinsics",
        FX == 320.0 && CX == 320.0 && CY == 320.0,
        format!("fx={FX} cx={CX} cy={CY}"),
    );

    // -- rectification -----------------------------------------------------
    let mut max_lat = 0.0f64;
    let mut bmin = f64::MAX;
    let mut bmax = f64::MIN;
    for i in 0..traj.n_frames.min(traj.poses_left.len()) {
        let b = traj.baseline_in_cam(i);
        bmin = bmin.min(b[0]);
        bmax = bmax.max(b[0]);
        max_lat = max_lat.max(b[1].abs()).max(b[2].abs());
    }
    check(
        "baseline on camera x",
        (bmin - BASELINE_M).abs() < 1e-6 && (bmax - BASELINE_M).abs() < 1e-6,
        format!("[{bmin:.9}, {bmax:.9}] m"),
    );
    check(
        "no vertical/forward offset (rectified)",
        max_lat < 1e-5,
        format!("max |t_y|,|t_z| = {max_lat:.2e} m"),
    );

    // -- backprojection ----------------------------------------------------
    let pose = traj.poses_left[0];
    for &(u, v, want) in REF_WORLD {
        let z = depth.at(u, v) as f64;
        let r = pose.cam_to_world();
        let x = (u as f64 - CX) * z / FX;
        let y = (v as f64 - CY) * z / FX;
        let got = [
            r[0][0] * x + r[0][1] * y + r[0][2] * z + pose.t[0],
            r[1][0] * x + r[1][1] * y + r[1][2] * z + pose.t[1],
            r[2][0] * x + r[2][1] * y + r[2][2] * z + pose.t[2],
        ];
        let err = (0..3).map(|i| (got[i] - want[i]).abs()).fold(0.0, f64::max);
        check(
            &format!("world px({u},{v})"),
            err < 1e-5,
            format!("[{:.6}, {:.6}, {:.6}] err {err:.2e}", got[0], got[1], got[2]),
        );
    }

    // Full-frame centroid: catches an error that a few sampled pixels would miss.
    let pc = backproject(&depth, None, &pose, 0.0);
    let mut c = [0.0f64; 3];
    for p in pc.xyz.chunks_exact(3) {
        for i in 0..3 {
            c[i] += p[i] as f64;
        }
    }
    for v in c.iter_mut() {
        *v /= pc.n as f64;
    }
    let cerr = (0..3).map(|i| (c[i] - REF_CENTROID[i]).abs()).fold(0.0, f64::max);
    check(
        "point count",
        pc.n == REF_N_POINTS,
        format!("{} (ref {REF_N_POINTS})", pc.n),
    );
    check(
        "full-frame centroid",
        cerr < 1e-4,
        format!("[{:.6}, {:.6}, {:.6}] err {cerr:.2e}", c[0], c[1], c[2]),
    );

    // -- range gate --------------------------------------------------------
    let gated = backproject(&depth, None, &pose, 3.0);
    check(
        "range gate drops far field",
        gated.n < pc.n && gated.n > 0,
        format!("{} of {} points within 3 m", gated.n, pc.n),
    );

    println!("\n=== {} ===", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
