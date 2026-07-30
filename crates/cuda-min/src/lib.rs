//! Minimal CUDA driver-API bindings shared by every arm.
//!
//! Raw FFI rather than a crate: the surface needed is small, the project
//! already depends on the CUDA toolchain, and the arms must all allocate from
//! the same context or their timings are not comparable.
//!
//! # Primary context, not a fresh one
//!
//! `CudaContext` retains the **primary** context rather than calling
//! `cuCtxCreate`. The vendored reference TSDF (arm A0) is compiled against the
//! CUDA *runtime* API, which binds to the primary context. Creating a separate
//! driver context would put runtime allocations and driver allocations in
//! different contexts, and device pointers would not be valid across them.
//! That fails as an illegal-address fault at kernel launch, far from the cause.

use std::ffi::{c_char, c_int, c_uint, c_void};

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
    // Re-exported for triton-aot's module loading.
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> CUresult;
    fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: c_int) -> CUresult;
    fn cuDevicePrimaryCtxRelease_v2(dev: c_int) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuLaunchKernel(
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

pub fn check(op: &'static str, code: CUresult) -> Result<(), CuError> {
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

/// Retains the device's primary context for the process lifetime.
pub struct CudaContext {
    ctx: CUcontext,
    dev: c_int,
}

impl CudaContext {
    pub fn new(device_ordinal: i32) -> Result<Self, CuError> {
        // SAFETY: standard driver init sequence; all out-params are owned locals.
        unsafe {
            check("cuInit", cuInit(0))?;
            let mut dev: c_int = 0;
            check("cuDeviceGet", cuDeviceGet(&mut dev, device_ordinal))?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            // Primary context: shared with the CUDA runtime API, which the
            // vendored reference TSDF uses. See the module docs.
            check(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut ctx, dev),
            )?;
            check("cuCtxSetCurrent", cuCtxSetCurrent(ctx))?;
            Ok(Self { ctx, dev })
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
            // SAFETY: retained above, released once.
            unsafe { cuDevicePrimaryCtxRelease_v2(self.dev) };
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

