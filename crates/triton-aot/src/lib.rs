//! Load and launch ahead-of-time compiled Triton kernels from Rust.
//!
//! The Triton fusion arm must run under the same driver as every other arm, or
//! the benchmark measures language runtime instead of kernel time. Triton
//! exposes its compiled cubin, so `tools/triton_aot.py` emits `.cubin` plus a
//! manifest at build time and this crate launches it through the CUDA driver
//! API. Python is never in the measured path.
//!
//! ## The launch ABI
//!
//! Undocumented; read out of `triton/backends/nvidia/driver.py` in Triton 3.6.0
//! and validated end to end by `validate_abi`:
//!
//! * Kernel params are the non-constexpr arguments in declaration order,
//!   followed by `global_scratch` then `profile_scratch` (driver.py:261-262).
//!   Both are `CUdeviceptr`, null when their sizes are zero. **Omitting them
//!   does not fail the launch**, it reads garbage from the param space.
//! * `blockDimX = 32 * num_warps` (driver.py:330). This is NOT the `BLOCK`
//!   constexpr. `BLOCK` is a tile size; confusing them launches the wrong
//!   shape and silently computes the wrong answer.
//! * `shared` from the manifest is dynamic shared memory.
//!
//! Every one of those fails *silently* rather than loudly, which is why
//! `validate_abi` checks numerical output against Python-launched reference
//! values rather than merely checking that the launch returned success.
//!
//! Raw driver FFI rather than a crate: the surface needed is small, and the
//! project already depends on the CUDA toolchain.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::path::Path;

pub type CUdeviceptr = u64;
pub type CUresult = c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUctxOpaque {
    _p: [u8; 0],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmodOpaque {
    _p: [u8; 0],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUfuncOpaque {
    _p: [u8; 0],
}

pub type CUcontext = *mut CUctxOpaque;
pub type CUmodule = *mut CUmodOpaque;
pub type CUfunction = *mut CUfuncOpaque;
pub type CUstream = *mut c_void;

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: c_int) -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(module: CUmodule) -> CUresult;
    fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_bytes: c_uint,
        stream: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuMemsetD8_v2(dst: CUdeviceptr, uc: u8, n: usize) -> CUresult;
    fn cuGetErrorName(error: CUresult, pstr: *mut *const c_char) -> CUresult;
}

#[derive(Debug, thiserror::Error)]
pub enum CuError {
    #[error("CUDA driver call {op} failed: {name} ({code})")]
    Driver {
        op: &'static str,
        name: String,
        code: c_int,
    },
    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("manifest parse error in {path}: {msg}")]
    Manifest { path: String, msg: String },
}

fn check(op: &'static str, code: CUresult) -> Result<(), CuError> {
    if code == 0 {
        return Ok(());
    }
    let mut p: *const c_char = std::ptr::null();
    // SAFETY: cuGetErrorName writes a pointer to a static string, or leaves it
    // null for an unknown code, which is handled below.
    let name = unsafe {
        if cuGetErrorName(code, &mut p) == 0 && !p.is_null() {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            "UNKNOWN".to_string()
        }
    };
    Err(CuError::Driver { op, name, code })
}

/// Owns a CUDA context. Everything else here borrows it.
pub struct CudaContext {
    ctx: CUcontext,
}

impl CudaContext {
    pub fn new(device_ordinal: i32) -> Result<Self, CuError> {
        // SAFETY: standard driver init sequence; all out-params are owned locals.
        unsafe {
            check("cuInit", cuInit(0))?;
            let mut dev: c_int = 0;
            check("cuDeviceGet", cuDeviceGet(&mut dev, device_ordinal))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check("cuCtxCreate", cuCtxCreate_v2(&mut ctx, 0, dev))?;
            Ok(Self { ctx })
        }
    }

    pub fn synchronize(&self) -> Result<(), CuError> {
        // SAFETY: context is current for this thread.
        unsafe { check("cuCtxSynchronize", cuCtxSynchronize()) }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: created by cuCtxCreate, destroyed once.
            unsafe { cuCtxDestroy_v2(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// An owned device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn zeroed(bytes: usize) -> Result<Self, CuError> {
        // SAFETY: out-param is a local; the allocation is owned by this struct.
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            check("cuMemAlloc", cuMemAlloc_v2(&mut ptr, bytes))?;
            check("cuMemsetD8", cuMemsetD8_v2(ptr, 0, bytes))?;
            Ok(Self { ptr, bytes })
        }
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self, CuError> {
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `data` is valid for `bytes` and the allocation is exactly that big.
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            check("cuMemAlloc", cuMemAlloc_v2(&mut ptr, bytes))?;
            check(
                "cuMemcpyHtoD",
                cuMemcpyHtoD_v2(ptr, data.as_ptr() as *const c_void, bytes),
            )?;
            Ok(Self { ptr, bytes })
        }
    }

    pub fn to_vec<T: Copy + Default>(&self) -> Result<Vec<T>, CuError> {
        let n = self.bytes / std::mem::size_of::<T>();
        let mut out = vec![T::default(); n];
        // SAFETY: `out` holds exactly self.bytes and the allocation is live.
        unsafe {
            check(
                "cuMemcpyDtoH",
                cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.ptr, self.bytes),
            )?;
        }
        Ok(out)
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: allocated by cuMemAlloc, freed once.
            unsafe { cuMemFree_v2(self.ptr) };
            self.ptr = 0;
        }
    }
}

/// The subset of the AOT manifest the launcher needs.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub name: String,
    pub num_warps: u32,
    pub block_dim_x: u32,
    pub shared_bytes: u32,
    pub global_scratch_size: usize,
    pub profile_scratch_size: usize,
    pub runtime_args: Vec<String>,
}

impl Manifest {
    /// Minimal field extraction. Deliberately dependency-free: pulling in a
    /// JSON crate for six fields is not worth the build-time cost, and the
    /// manifest is generated by our own tool so its shape is fixed.
    pub fn from_json(path: &Path) -> Result<Self, CuError> {
        let text = std::fs::read_to_string(path).map_err(|e| CuError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        let p = path.display().to_string();
        let err = |msg: String| CuError::Manifest {
            path: p.clone(),
            msg,
        };

        let str_field = |key: &str| -> Result<String, CuError> {
            let pat = format!("\"{key}\":");
            let i = text.find(&pat).ok_or_else(|| err(format!("missing {key}")))?;
            let rest = &text[i + pat.len()..];
            let a = rest.find('"').ok_or_else(|| err(format!("bad {key}")))?;
            let b = rest[a + 1..]
                .find('"')
                .ok_or_else(|| err(format!("bad {key}")))?;
            Ok(rest[a + 1..a + 1 + b].to_string())
        };
        let num_field = |key: &str| -> Result<u64, CuError> {
            let pat = format!("\"{key}\":");
            let i = text.find(&pat).ok_or_else(|| err(format!("missing {key}")))?;
            let rest = text[i + pat.len()..].trim_start();
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end]
                .parse::<u64>()
                .map_err(|_| err(format!("bad number for {key}")))
        };

        // Argument names, in declaration order. Order is the ABI.
        let mut runtime_args = Vec::new();
        if let Some(i) = text.find("\"runtime_args\":") {
            let seg = &text[i..];
            let end = seg.find(']').unwrap_or(seg.len());
            for chunk in seg[..end].split("\"name\":").skip(1) {
                let rest = chunk.trim_start();
                if let Some(a) = rest.find('"') {
                    if let Some(b) = rest[a + 1..].find('"') {
                        runtime_args.push(rest[a + 1..a + 1 + b].to_string());
                    }
                }
            }
        }

        Ok(Manifest {
            name: str_field("name")?,
            num_warps: num_field("num_warps")? as u32,
            block_dim_x: num_field("block_dim_x")? as u32,
            shared_bytes: num_field("shared_bytes")? as u32,
            global_scratch_size: num_field("global_scratch_size")? as usize,
            profile_scratch_size: num_field("profile_scratch_size")? as usize,
            runtime_args,
        })
    }
}

/// One AOT-compiled Triton kernel, ready to launch.
pub struct TritonKernel {
    module: CUmodule,
    func: CUfunction,
    pub manifest: Manifest,
    // Scratch buffers the launcher passes as the two trailing params. Held so
    // their lifetime covers every launch.
    _global_scratch: Option<DeviceBuffer>,
    _profile_scratch: Option<DeviceBuffer>,
    global_scratch_ptr: CUdeviceptr,
    profile_scratch_ptr: CUdeviceptr,
}

impl TritonKernel {
    /// Load `<dir>/<name>.cubin` with `<dir>/<name>.json`.
    pub fn load(dir: &Path, name: &str) -> Result<Self, CuError> {
        let manifest = Manifest::from_json(&dir.join(format!("{name}.json")))?;
        let cubin_path = dir.join(format!("{name}.cubin"));
        let cubin = std::fs::read(&cubin_path).map_err(|e| CuError::Io {
            path: cubin_path.display().to_string(),
            source: e,
        })?;

        // SAFETY: `cubin` outlives the call; cuModuleLoadData copies the image.
        let module = unsafe {
            let mut m: CUmodule = std::ptr::null_mut();
            check(
                "cuModuleLoadData",
                cuModuleLoadData(&mut m, cubin.as_ptr() as *const c_void),
            )?;
            m
        };

        let cname = CString::new(manifest.name.as_str()).unwrap();
        // SAFETY: module is live; the name comes from the manifest the same
        // build step emitted, so it matches the cubin's entry symbol.
        let func = unsafe {
            let mut f: CUfunction = std::ptr::null_mut();
            check(
                "cuModuleGetFunction",
                cuModuleGetFunction(&mut f, module, cname.as_ptr()),
            )?;
            f
        };

        // Allocate the trailing scratch buffers only when non-zero. A null
        // pointer is what the Python launcher passes in that case.
        let g = if manifest.global_scratch_size > 0 {
            Some(DeviceBuffer::zeroed(manifest.global_scratch_size)?)
        } else {
            None
        };
        let p = if manifest.profile_scratch_size > 0 {
            Some(DeviceBuffer::zeroed(manifest.profile_scratch_size)?)
        } else {
            None
        };
        let gp = g.as_ref().map(|b| b.ptr()).unwrap_or(0);
        let pp = p.as_ref().map(|b| b.ptr()).unwrap_or(0);

        Ok(Self {
            module,
            func,
            manifest,
            _global_scratch: g,
            _profile_scratch: p,
            global_scratch_ptr: gp,
            profile_scratch_ptr: pp,
        })
    }

    /// Launch with `args` as the runtime (non-constexpr) parameters.
    ///
    /// The two trailing scratch pointers are appended here, so callers pass
    /// only the kernel's own arguments and cannot forget them.
    ///
    /// # Safety
    /// Each entry of `args` must point to a live value whose type matches the
    /// kernel's declared parameter type, and device pointers must address
    /// allocations large enough for the kernel's accesses. Neither is checked.
    pub unsafe fn launch(&self, grid_x: u32, args: &mut [*mut c_void]) -> Result<(), CuError> {
        let mut params: Vec<*mut c_void> = Vec::with_capacity(args.len() + 2);
        params.extend_from_slice(args);
        // driver.py:261-262. Order matters and omission is silent.
        params.push(&self.global_scratch_ptr as *const _ as *mut c_void);
        params.push(&self.profile_scratch_ptr as *const _ as *mut c_void);

        check(
            "cuLaunchKernel",
            cuLaunchKernel(
                self.func,
                grid_x,
                1,
                1,
                self.manifest.block_dim_x, // 32 * num_warps, NOT the BLOCK constexpr
                1,
                1,
                self.manifest.shared_bytes,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
        )
    }
}

impl Drop for TritonKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: loaded by cuModuleLoadData, unloaded once.
            unsafe { cuModuleUnload(self.module) };
            self.module = std::ptr::null_mut();
        }
    }
}
