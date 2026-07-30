//! TartanAir V2 loader: rectified stereo, ground-truth depth, poses, and
//! backprojection to world-space points.
//!
//! World-space is the interface deliberately: every TSDF arm takes world points
//! (`tsdf_hash_add_points_chunk_device`), so pose composition happens exactly
//! once, here, and cannot differ between arms.
//!
//! # The constants are measured, not documented
//!
//! TartanAir's V1 documentation is wrong about V2 in ways that corrupt results
//! **silently rather than loudly**. Every value below was measured from the
//! data (see `tools/tartanair.py` and `docs/DATASET.md`):
//!
//! * Resolution is 640x640, not the documented 640x480, so `cy = 320` not 240.
//!   The documented value biases every backprojected point vertically.
//! * Depth is float32 bit-packed into a 4-channel PNG, not 16-bit NPY.
//!
//! # The depth byte order
//!
//! The float's little-endian bytes are **not** in PNG channel order. Verified
//! bitwise over a whole frame: `f32 LE = [file[2], file[1], file[0], file[3]]`.
//!
//! That is because the data was written with OpenCV, which treats a 4-channel
//! image as BGRA and therefore swaps R and B on the way to the file's RGBA.
//!
//! Getting this wrong is dangerous because the failure is quiet. Measured, by
//! deliberately decoding in file order `[0,1,2,3]` instead:
//!
//! | Check | Wrong byte order |
//! |---|---|
//! | Value range | plausible: 2.0000055 m at (0,0) |
//! | Median gradient | 0.0015 m, i.e. *passes* the smoothness check |
//! | Exact value vs reference | 2.0000055 vs 2.359375, fails |
//!
//! So neither a range assertion nor `verify_smooth` is sufficient on its own:
//! that permutation yields depth that is both plausibly scaled and smooth. Only
//! comparison against externally computed reference values catches it, which is
//! what `verify_loader` does. `verify_smooth` still earns its place because it
//! catches the *other* permutations, which do produce high-frequency noise.

use std::path::{Path, PathBuf};

// --- Measured format constants. Do not "correct" these from the docs. ------
pub const WIDTH: usize = 640;
pub const HEIGHT: usize = 640; // docs say 480
pub const FX: f64 = 320.0;
pub const FY: f64 = 320.0;
pub const CX: f64 = 320.0;
pub const CY: f64 = 320.0; // docs say 240
/// Verified 0.250000011 m, std 1.3e-7, over 129 frames.
pub const BASELINE_M: f64 = 0.25;
/// `Z = FX_TIMES_BASELINE / disparity`.
pub const FX_TIMES_BASELINE: f64 = FX * BASELINE_M; // 80.0

/// Rotation taking TartanAir's NED body frame to the camera frame.
/// From the upstream `tartanair` package (`customizer.py`):
/// `camera_pose[0:3,0:3] = R(quat) @ NED_R_cam`.
const NED_R_CAM: [[f64; 3]; 3] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

#[derive(Debug, thiserror::Error)]
pub enum TaError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("png decode error on {path}: {msg}")]
    Png { path: String, msg: String },
    #[error("{path}: expected {expect}, got {got}")]
    Format {
        path: String,
        expect: String,
        got: String,
    },
    #[error("{path}: malformed pose line {line}")]
    Pose { path: String, line: usize },
}

/// Ground-truth depth: planar Z in metres, row-major, `WIDTH * HEIGHT`.
///
/// Planar Z, not ray length. Disparity is defined against planar Z, so applying
/// the upstream package's `depth_to_dist()` before disparity work injects a
/// radially varying error that looks like lens distortion.
pub struct Depth {
    pub data: Vec<f32>,
}

impl Depth {
    pub fn load(path: &Path) -> Result<Self, TaError> {
        let (buf, info) = decode_png(path)?;
        if info.2 != 4 {
            return Err(TaError::Format {
                path: path.display().to_string(),
                expect: "4-channel PNG".into(),
                got: format!("{}-channel", info.2),
            });
        }
        if info.0 != WIDTH || info.1 != HEIGHT {
            return Err(TaError::Format {
                path: path.display().to_string(),
                expect: format!("{WIDTH}x{HEIGHT}"),
                got: format!("{}x{}", info.0, info.1),
            });
        }
        let n = WIDTH * HEIGHT;
        let mut data = Vec::with_capacity(n);
        for px in buf.chunks_exact(4) {
            // See the module docs: the float's LE bytes are file[2,1,0,3].
            data.push(f32::from_le_bytes([px[2], px[1], px[0], px[3]]));
        }
        Ok(Self { data })
    }

    pub fn at(&self, u: usize, v: usize) -> f32 {
        self.data[v * WIDTH + u]
    }

    /// Median absolute horizontal gradient, in metres.
    ///
    /// Catches byte permutations that scatter the exponent across pixels,
    /// which a value-range assertion would not: real depth is piecewise smooth
    /// (~0.002 m here), badly permuted depth is high-frequency noise.
    ///
    /// NOT sufficient on its own. The specific permutation `[0,1,2,3]` yields
    /// depth that is both plausibly scaled and smooth (0.0015 m), and is caught
    /// only by comparing exact values against a reference. See the module docs.
    pub fn median_abs_gradient(&self) -> f32 {
        let mut g = Vec::with_capacity(WIDTH * HEIGHT);
        for v in 0..HEIGHT {
            for u in 1..WIDTH {
                g.push((self.at(u, v) - self.at(u - 1, v)).abs());
            }
        }
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        g[g.len() / 2]
    }

    pub fn verify_smooth(&self) -> bool {
        self.median_abs_gradient() < 0.05
    }
}

/// RGB image, row-major, 3 bytes per pixel.
pub struct Rgb {
    pub data: Vec<u8>,
}

impl Rgb {
    pub fn load(path: &Path) -> Result<Self, TaError> {
        let (buf, info) = decode_png(path)?;
        let (w, h, ch) = info;
        if w != WIDTH || h != HEIGHT {
            return Err(TaError::Format {
                path: path.display().to_string(),
                expect: format!("{WIDTH}x{HEIGHT}"),
                got: format!("{w}x{h}"),
            });
        }
        let data = if ch == 3 {
            buf
        } else {
            buf.chunks_exact(ch)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect()
        };
        Ok(Self { data })
    }
}

/// One pose: translation and unit quaternion, NED (x forward, y right, z down).
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub t: [f64; 3],
    /// `qx, qy, qz, qw`, matching the file column order.
    pub q: [f64; 4],
}

impl Pose {
    /// Camera-to-world rotation: `R(q) @ NED_R_cam`.
    pub fn cam_to_world(&self) -> [[f64; 3]; 3] {
        let r = quat_to_mat(self.q);
        let mut out = [[0.0f64; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| r[i][k] * NED_R_CAM[k][j]).sum();
            }
        }
        out
    }
}

/// `qx, qy, qz, qw` to a 3x3 rotation matrix.
fn quat_to_mat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let n = (x * x + y * y + z * z + w * w).sqrt();
    let (x, y, z, w) = (x / n, y / n, z / n, w / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// A trajectory directory, e.g. `<env>/Data_easy/P000`.
pub struct Trajectory {
    pub root: PathBuf,
    cam: String,
    pub poses_left: Vec<Pose>,
    pub poses_right: Vec<Pose>,
    pub n_frames: usize,
}

impl Trajectory {
    pub fn open(root: impl Into<PathBuf>, cam: &str) -> Result<Self, TaError> {
        let root = root.into();
        let poses_left = load_poses(&root.join(format!("pose_lcam_{cam}.txt")))?;
        let poses_right = load_poses(&root.join(format!("pose_rcam_{cam}.txt")))?;
        let dir = root.join(format!("image_lcam_{cam}"));
        let n_frames = std::fs::read_dir(&dir)
            .map_err(|e| TaError::Io {
                path: dir.display().to_string(),
                source: e,
            })?
            .filter(|e| {
                e.as_ref()
                    .map(|e| e.path().extension().is_some_and(|x| x == "png"))
                    .unwrap_or(false)
            })
            .count();
        Ok(Self {
            root,
            cam: cam.to_string(),
            poses_left,
            poses_right,
            n_frames,
        })
    }

    pub fn depth_path(&self, i: usize) -> PathBuf {
        self.root
            .join(format!("depth_lcam_{}", self.cam))
            .join(format!("{:06}_lcam_{}_depth.png", i, self.cam))
    }

    pub fn rgb_path(&self, i: usize) -> PathBuf {
        self.root
            .join(format!("image_lcam_{}", self.cam))
            .join(format!("{:06}_lcam_{}.png", i, self.cam))
    }

    pub fn depth(&self, i: usize) -> Result<Depth, TaError> {
        Depth::load(&self.depth_path(i))
    }

    pub fn rgb(&self, i: usize) -> Result<Rgb, TaError> {
        Rgb::load(&self.rgb_path(i))
    }

    /// Stereo baseline in the left-camera frame, per frame.
    ///
    /// A rectified pair must show pure translation along camera x with no
    /// relative rotation. Used by the verifier rather than assumed.
    pub fn baseline_in_cam(&self, i: usize) -> [f64; 3] {
        let l = self.poses_left[i];
        let r = self.poses_right[i];
        let rot = l.cam_to_world();
        let d = [
            r.t[0] - l.t[0],
            r.t[1] - l.t[1],
            r.t[2] - l.t[2],
        ];
        // R^T * d, i.e. world delta expressed in the left camera frame.
        [
            rot[0][0] * d[0] + rot[1][0] * d[1] + rot[2][0] * d[2],
            rot[0][1] * d[0] + rot[1][1] * d[1] + rot[2][1] * d[2],
            rot[0][2] * d[0] + rot[1][2] * d[1] + rot[2][2] * d[2],
        ]
    }
}

/// Backprojection output: interleaved xyz world positions plus rgb.
pub struct PointCloud {
    pub xyz: Vec<f32>,
    pub rgb: Vec<u8>,
    pub n: usize,
}

/// Backproject one frame to world-space points.
///
/// `max_range_m` drops far-field points before they reach the volume. Not
/// cosmetic: at this baseline `fx*B = 80`, so 1 px of disparity error costs
/// 31 cm at 5 m. Beyond a few metres, depth error exceeds any sensible voxel
/// and the points only consume block-pool slots. See `docs/DATASET.md`.
pub fn backproject(
    depth: &Depth,
    rgb: Option<&Rgb>,
    pose: &Pose,
    max_range_m: f32,
) -> PointCloud {
    let r = pose.cam_to_world();
    let t = pose.t;
    let mut xyz = Vec::with_capacity(WIDTH * HEIGHT * 3);
    let mut out_rgb = Vec::with_capacity(WIDTH * HEIGHT * 3);

    for v in 0..HEIGHT {
        for u in 0..WIDTH {
            let z = depth.at(u, v);
            if !z.is_finite() || z <= 0.0 || (max_range_m > 0.0 && z > max_range_m) {
                continue;
            }
            let zd = z as f64;
            let x = (u as f64 - CX) * zd / FX;
            let y = (v as f64 - CY) * zd / FY;
            for i in 0..3 {
                let w = r[i][0] * x + r[i][1] * y + r[i][2] * zd + t[i];
                xyz.push(w as f32);
            }
            if let Some(c) = rgb {
                let o = (v * WIDTH + u) * 3;
                out_rgb.extend_from_slice(&c.data[o..o + 3]);
            }
        }
    }
    let n = xyz.len() / 3;
    PointCloud {
        xyz,
        rgb: out_rgb,
        n,
    }
}

// --- helpers ---------------------------------------------------------------

/// Image dimensions and sample count: `(width, height, channels)`.
type ImageInfo = (usize, usize, usize);

/// Returns raw 8-bit samples in file channel order, plus dimensions.
fn decode_png(path: &Path) -> Result<(Vec<u8>, ImageInfo), TaError> {
    let file = std::fs::File::open(path).map_err(|e| TaError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().map_err(|e| TaError::Png {
        path: path.display().to_string(),
        msg: e.to_string(),
    })?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| TaError::Png {
        path: path.display().to_string(),
        msg: e.to_string(),
    })?;
    buf.truncate(info.buffer_size());
    let channels = info.color_type.samples();
    if info.bit_depth != png::BitDepth::Eight {
        return Err(TaError::Format {
            path: path.display().to_string(),
            expect: "8-bit samples".into(),
            got: format!("{:?}", info.bit_depth),
        });
    }
    Ok((buf, (info.width as usize, info.height as usize, channels)))
}

fn load_poses(path: &Path) -> Result<Vec<Pose>, TaError> {
    let text = std::fs::read_to_string(path).map_err(|e| TaError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len() != 7 {
            return Err(TaError::Pose {
                path: path.display().to_string(),
                line: i + 1,
            });
        }
        out.push(Pose {
            t: [v[0], v[1], v[2]],
            q: [v[3], v[4], v[5], v[6]],
        });
    }
    Ok(out)
}
