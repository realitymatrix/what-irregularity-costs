//! Mesh agreement metrics for the correctness gate.
//!
//! Every arm must produce a mesh agreeing with the golden reference before any
//! timing is trusted. A faster backend that drifts is not a result.
//!
//! # Why this is implemented here rather than called out to Open3D
//!
//! The roadmap's plan of record was to use Open3D for mesh comparison. This
//! deviates, for two reasons:
//!
//! * Linking a static Open3D (1.4 GB, ~30 third-party archives, CUDA runtime,
//!   libstdc++) into the Rust harness makes the gate the most fragile component
//!   in the project, when it is the one component every other arm depends on.
//! * The gate runs on every arm on every change, so it wants to be fast and
//!   self-contained.
//!
//! The correctness argument is preserved two ways: an analytic case with a
//! known answer (concentric spheres, where every surface point is exactly `dr`
//! from the other surface), and cross-validation against Open3D's
//! `RaycastingScene.compute_distance` on identical inputs
//! (`tools/verify_metrics_open3d.py`). Same discipline as the TartanAir loader
//! and the Triton ABI: implement locally, then check against an external
//! reference. Open3D remains a first-class dependency as arm A1.
//!
//! # Method
//!
//! Arms emit triangle soups. Comparing vertex sets directly would be sensitive
//! to tessellation differences that are not geometric disagreements, so each
//! mesh is area-weighted sampled and each sample is measured to the *other
//! mesh's surface* (point-to-triangle), symmetrically.
//!
//! Point-to-triangle rather than point-to-point, and the distinction decides
//! whether the gate works at all. An earlier point-to-point version imposed a
//! floor of roughly half the sample spacing: 60k samples over a 12.6 m^2 sphere
//! reported **7 mm between a mesh and itself**, against a 1 mm tolerance for a
//! 1 cm voxel. The metric could not resolve what it existed to gate. Measuring
//! to the surface removes the floor; identical meshes now give exactly 0.0.
//!
//! Sampling is seeded and deterministic: a gate that gives a different answer
//! per run cannot gate anything.

use std::collections::HashMap;

/// A triangle soup: 3 consecutive vertices per triangle.
pub struct TriMesh<'a> {
    /// Interleaved xyz, `3 * n_vertices` floats.
    pub xyz: &'a [f32],
}

impl<'a> TriMesh<'a> {
    pub fn n_vertices(&self) -> usize {
        self.xyz.len() / 3
    }

    pub fn n_triangles(&self) -> usize {
        self.n_vertices() / 3
    }

    fn vertex(&self, i: usize) -> [f64; 3] {
        [
            self.xyz[i * 3] as f64,
            self.xyz[i * 3 + 1] as f64,
            self.xyz[i * 3 + 2] as f64,
        ]
    }

    fn triangle(&self, t: usize) -> [[f64; 3]; 3] {
        [
            self.vertex(t * 3),
            self.vertex(t * 3 + 1),
            self.vertex(t * 3 + 2),
        ]
    }
}

/// Deterministic RNG. `Math.random`-style nondeterminism would make the gate
/// unreproducible, so this is a seeded xorshift rather than a thread RNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Area-weighted uniform sampling over the mesh surface.
///
/// Area-weighted, not per-triangle-uniform: a mesh with many tiny triangles in
/// one region would otherwise oversample it and the metric would report
/// tessellation differences as geometric error.
pub fn sample_surface(mesh: &TriMesh, n_samples: usize, seed: u64) -> Vec<[f64; 3]> {
    let nt = mesh.n_triangles();
    if nt == 0 || n_samples == 0 {
        return Vec::new();
    }

    // Cumulative area distribution.
    let mut cum = Vec::with_capacity(nt);
    let mut total = 0.0f64;
    for t in 0..nt {
        let [a, b, c] = mesh.triangle(t);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cr = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        total += 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        cum.push(total);
    }
    if total <= 0.0 {
        // Degenerate (zero-area) mesh: fall back to vertices so the caller gets
        // a meaningful comparison instead of an empty set.
        return (0..mesh.n_vertices()).map(|i| mesh.vertex(i)).collect();
    }

    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let r = rng.next_f64() * total;
        let t = cum.partition_point(|&c| c < r).min(nt - 1);
        let [a, b, c] = mesh.triangle(t);
        // Uniform barycentric sample.
        let (mut u, mut v) = (rng.next_f64(), rng.next_f64());
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;
        out.push([
            a[0] * w + b[0] * u + c[0] * v,
            a[1] * w + b[1] * u + c[1] * v,
            a[2] * w + b[2] * u + c[2] * v,
        ]);
    }
    out
}

/// Squared distance from a point to a triangle.
///
/// Point-to-triangle, not point-to-point, and that distinction is load-bearing.
/// Comparing two independent surface samplings by nearest *point* imposes a
/// floor of roughly half the sample spacing: measured here, 60k samples over a
/// 12.6 m^2 sphere gave a 7 mm floor between a mesh and *itself*, which exceeds
/// the 1 mm tolerance a 1 cm voxel demands. The metric could not have resolved
/// the agreement it exists to gate. Distance to the triangle removes the floor:
/// identical meshes give exactly zero at any sample count.
fn point_triangle_dist2(p: [f64; 3], tri: &[[f64; 3]; 3]) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let (a, b, c) = (tri[0], tri[1], tri[2]);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap); // vertex A
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp); // vertex B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = [a[0] + ab[0] * v, a[1] + ab[1] * v, a[2] + ab[2] * v];
        let d = sub(p, q);
        return dot(d, d); // edge AB
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp); // vertex C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = [a[0] + ac[0] * w, a[1] + ac[1] * w, a[2] + ac[2] * w];
        let d = sub(p, q);
        return dot(d, d); // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = [
            b[0] + (c[0] - b[0]) * w,
            b[1] + (c[1] - b[1]) * w,
            b[2] + (c[2] - b[2]) * w,
        ];
        let d = sub(p, q);
        return dot(d, d); // edge BC
    }
    // Interior: project onto the plane via barycentric coordinates.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    let d = sub(p, q);
    dot(d, d)
}

/// Uniform spatial hash over triangles, for point-to-surface queries.
///
/// Each triangle is binned into every cell its AABB touches, so a query only
/// has to expand shells until the best hit is provably closer than the shell
/// boundary.
struct TriGrid {
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
    inv_cell: f64,
    tris: Vec<[[f64; 3]; 3]>,
}

impl TriGrid {
    fn build(tris: Vec<[[f64; 3]; 3]>, cell: f64) -> Self {
        let inv_cell = 1.0 / cell;
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, t) in tris.iter().enumerate() {
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            for v in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            let (kl, kh) = (key(&lo, inv_cell), key(&hi, inv_cell));
            for x in kl.0..=kh.0 {
                for y in kl.1..=kh.1 {
                    for z in kl.2..=kh.2 {
                        cells.entry((x, y, z)).or_default().push(i);
                    }
                }
            }
        }
        Self {
            cells,
            inv_cell,
            tris,
        }
    }

    fn nearest_sq(&self, q: &[f64; 3]) -> f64 {
        let (kx, ky, kz) = key(q, self.inv_cell);
        let cell = 1.0 / self.inv_cell;
        let mut best = f64::INFINITY;
        let mut r = 0i64;
        loop {
            for dx in -r..=r {
                for dy in -r..=r {
                    let range: Vec<i64> = if dx.abs() == r || dy.abs() == r {
                        (-r..=r).collect()
                    } else {
                        vec![-r, r]
                    };
                    for dz in range {
                        if let Some(ids) = self.cells.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &i in ids {
                                let d = point_triangle_dist2(*q, &self.tris[i]);
                                if d < best {
                                    best = d;
                                }
                            }
                        }
                    }
                }
            }
            // Stop once the best hit is provably closer than anything outside
            // the shells scanned. Deliberately NOT conditioned on improving
            // during this shell: a shell can contain no closer triangle while a
            // later one still could, so keying the exit on improvement fails to
            // terminate. That bug cost a hung run before it was caught.
            if best.is_finite() && best.sqrt() <= (r as f64) * cell {
                break;
            }
            r += 1;
            if r > 64 {
                break;
            }
        }
        best
    }
}

fn key(p: &[f64; 3], inv_cell: f64) -> (i64, i64, i64) {
    (
        (p[0] * inv_cell).floor() as i64,
        (p[1] * inv_cell).floor() as i64,
        (p[2] * inv_cell).floor() as i64,
    )
}

/// Distances from one point set to its nearest neighbours in another.
#[derive(Debug, Clone)]
pub struct DistanceStats {
    pub mean: f64,
    pub rms: f64,
    pub p95: f64,
    pub p99: f64,
    /// Max distance, i.e. the one-sided Hausdorff.
    pub max: f64,
    pub n: usize,
}

fn stats(mut d: Vec<f64>) -> DistanceStats {
    if d.is_empty() {
        return DistanceStats {
            mean: 0.0,
            rms: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
            n: 0,
        };
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n as f64;
    let rms = (d.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    DistanceStats {
        mean,
        rms,
        p95: d[((n as f64 * 0.95) as usize).min(n - 1)],
        p99: d[((n as f64 * 0.99) as usize).min(n - 1)],
        max: d[n - 1],
        n,
    }
}

/// Symmetric agreement between two meshes.
#[derive(Debug, Clone)]
pub struct MeshAgreement {
    pub a_to_b: DistanceStats,
    pub b_to_a: DistanceStats,
    /// Symmetric mean surface-to-surface distance.
    pub mean_surface: f64,
    /// Symmetric Hausdorff, i.e. `max(max(a->b), max(b->a))`.
    pub hausdorff: f64,
    /// Symmetric 99th percentile. Reported alongside Hausdorff because a true
    /// max is decided by a single vertex and is therefore easy to trip on an
    /// otherwise-correct arm; p99 is the robust version.
    pub hausdorff_p99: f64,
}

/// Compare two meshes by area-weighted surface sampling.
///
/// `cell` sizes the spatial hash and should be near the voxel size; it affects
/// speed only, never the result.
pub fn compare(a: &TriMesh, b: &TriMesh, n_samples: usize, cell: f64, seed: u64) -> MeshAgreement {
    let sa = sample_surface(a, n_samples, seed);
    let sb = sample_surface(b, n_samples, seed ^ 0x9E3779B97F4A7C15);

    let tris = |m: &TriMesh| (0..m.n_triangles()).map(|t| m.triangle(t)).collect::<Vec<_>>();
    let ga = TriGrid::build(tris(a), cell);
    let gb = TriGrid::build(tris(b), cell);

    // Points sampled on A, measured to B's *surface*, and vice versa.
    let a_to_b = stats(sa.iter().map(|p| gb.nearest_sq(p).sqrt()).collect());
    let b_to_a = stats(sb.iter().map(|p| ga.nearest_sq(p).sqrt()).collect());

    MeshAgreement {
        mean_surface: 0.5 * (a_to_b.mean + b_to_a.mean),
        hausdorff: a_to_b.max.max(b_to_a.max),
        hausdorff_p99: a_to_b.p99.max(b_to_a.p99),
        a_to_b,
        b_to_a,
    }
}

/// Pass/fail thresholds for the correctness gate.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub mean_surface_m: f64,
    pub hausdorff_p99_m: f64,
    /// Relative vertex-count difference allowed. Marching cubes is
    /// deterministic given identical input, so a large divergence here means
    /// the arms saw different volumes, not merely different tessellation.
    pub vertex_count_rel: f64,
}

impl Tolerance {
    /// Default for a 1 cm voxel. Sub-voxel agreement in the mean, and a p99
    /// within one voxel: two implementations of the same algorithm should not
    /// disagree by more than the discretisation itself.
    pub fn for_voxel(voxel_m: f64) -> Self {
        Self {
            mean_surface_m: 0.1 * voxel_m,
            hausdorff_p99_m: voxel_m,
            vertex_count_rel: 0.02,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateResult {
    pub agreement: MeshAgreement,
    pub n_vertices_a: usize,
    pub n_vertices_b: usize,
    pub vertex_count_rel: f64,
    pub passed: bool,
    pub failures: Vec<String>,
}

pub fn gate(a: &TriMesh, b: &TriMesh, tol: Tolerance, n_samples: usize, cell: f64) -> GateResult {
    let agreement = compare(a, b, n_samples, cell, 0xC0FFEE);
    let (na, nb) = (a.n_vertices(), b.n_vertices());
    let rel = if na.max(nb) == 0 {
        0.0
    } else {
        (na as f64 - nb as f64).abs() / na.max(nb) as f64
    };

    let mut failures = Vec::new();
    if agreement.mean_surface > tol.mean_surface_m {
        failures.push(format!(
            "mean surface distance {:.6} m > {:.6} m",
            agreement.mean_surface, tol.mean_surface_m
        ));
    }
    if agreement.hausdorff_p99 > tol.hausdorff_p99_m {
        failures.push(format!(
            "hausdorff p99 {:.6} m > {:.6} m",
            agreement.hausdorff_p99, tol.hausdorff_p99_m
        ));
    }
    if rel > tol.vertex_count_rel {
        failures.push(format!(
            "vertex count differs by {:.2}% > {:.2}%",
            rel * 100.0,
            tol.vertex_count_rel * 100.0
        ));
    }

    GateResult {
        passed: failures.is_empty(),
        agreement,
        n_vertices_a: na,
        n_vertices_b: nb,
        vertex_count_rel: rel,
        failures,
    }
}

/// Write a triangle soup as a binary-free ASCII PLY.
///
/// Exists so the gate's inputs can be handed to Open3D for cross-validation,
/// and so a failing arm's output can be inspected rather than only reported.
pub fn write_ply(path: &std::path::Path, mesh: &TriMesh) -> std::io::Result<()> {
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    let nv = mesh.n_vertices();
    let nt = mesh.n_triangles();
    writeln!(w, "ply")?;
    writeln!(w, "format ascii 1.0")?;
    writeln!(w, "element vertex {nv}")?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    writeln!(w, "element face {nt}")?;
    writeln!(w, "property list uchar int vertex_indices")?;
    writeln!(w, "end_header")?;
    for i in 0..nv {
        let v = mesh.vertex(i);
        writeln!(w, "{} {} {}", v[0], v[1], v[2])?;
    }
    for t in 0..nt {
        writeln!(w, "3 {} {} {}", t * 3, t * 3 + 1, t * 3 + 2)?;
    }
    Ok(())
}
