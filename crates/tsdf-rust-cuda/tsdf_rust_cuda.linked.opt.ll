; ModuleID = '/home/petr/Documents/triton_stereo_depth_inference/crates/tsdf-rust-cuda/tsdf_rust_cuda.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@llvm.used = appending global [2 x ptr] [ptr @alloc_kernel, ptr @update_kernel], section "llvm.metadata"

; Function Attrs: convergent nofree nounwind
define ptx_kernel void @alloc_kernel(ptr readonly captures(none) %v0, i64 %v1, ptr captures(none) %v2, i64 %v3, ptr captures(none) %v4, i64 %v5, ptr writeonly captures(none) %v6, i64 %v7, ptr captures(none) %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i12 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #7
  %v6.i13 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #7
  %v5.i14 = icmp eq i32 %v4.i12, 1
  %v7.i15 = icmp eq i32 %v6.i13, 1
  %v8.not.not.i = and i1 %v5.i14, %v7.i15
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #7
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #7
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i16 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i16
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v46 = trunc i64 %v22.i to i32
  %v47.not = icmp sgt i32 %v10, %v46
  br i1 %v47.not, label %bb3, label %bb21

bb3:                                              ; preds = %entry
  %sext = mul i64 %v22.i, 12884901888
  %0 = ashr exact i64 %sext, 30
  %v52 = getelementptr inbounds i8, ptr %v0, i64 %0
  %v53 = load float, ptr %v52, align 4
  %v54 = getelementptr inbounds nuw i8, ptr %v52, i64 4
  %v55 = load float, ptr %v54, align 4
  %v56 = getelementptr inbounds nuw i8, ptr %v52, i64 8
  %v57 = load float, ptr %v56, align 4
  %v58 = fsub contract float %v53, %v15
  %v59 = fsub contract float %v55, %v16
  %v60 = fsub contract float %v57, %v17
  %v61 = fmul contract float %v58, %v58
  %v62 = fmul contract float %v59, %v59
  %v63 = fadd contract float %v61, %v62
  %v64 = fmul contract float %v60, %v60
  %v65 = fadd contract float %v63, %v64
  %v66 = fcmp ule float %v18, 0.000000e+00
  %v68 = fcmp ule float %v65, %v18
  %or.cond = select i1 %v66, i1 true, i1 %v68
  br i1 %or.cond, label %bb6, label %bb21

bb6:                                              ; preds = %bb3
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #8
  %.not.i = icmp eq i32 %1, 0
  %2 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #8
  %.not1.i = icmp eq i32 %2, 0
  br i1 %.not.i, label %8, label %3

3:                                                ; preds = %bb6
  br i1 %.not1.i, label %6, label %4

4:                                                ; preds = %3
  %5 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v65) #8
  br label %__nv_sqrtf.exit

6:                                                ; preds = %3
  %7 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v65) #8
  br label %__nv_sqrtf.exit

8:                                                ; preds = %bb6
  br i1 %.not1.i, label %11, label %9

9:                                                ; preds = %8
  %10 = tail call float @llvm.nvvm.sqrt.rn.f(float %v65) #8
  br label %__nv_sqrtf.exit

11:                                               ; preds = %8
  %12 = tail call float @llvm.nvvm.sqrt.approx.f(float %v65) #8
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %4, %6, %9, %11
  %.0.i = phi float [ %5, %4 ], [ %7, %6 ], [ %10, %9 ], [ %12, %11 ]
  %v96 = fcmp ule float %.0.i, 0x3EB0C6F7A0000000
  br i1 %v96, label %bb21, label %bb7

bb7:                                              ; preds = %__nv_sqrtf.exit
  %v71 = fdiv contract float %v58, %.0.i
  %v72 = fdiv contract float %v59, %.0.i
  %v73 = fdiv contract float %v60, %.0.i
  %v74 = fdiv contract float 1.000000e+00, %v13
  %v75 = fmul contract float %v14, %v74
  %v76 = tail call float @llvm.ceil.f32(float %v75) #7
  %v98 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v76) #7
  %v12.i22 = tail call range(i32 1, -2147483648) i32 @llvm.smax.i32(i32 %v98, i32 1)
  %v77 = sub nsw i32 0, %v12.i22
  %v130 = fneg float %v14
  %v41.not.i32.not = icmp eq i32 %v12, -1
  %13 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #8
  %.not1.i20 = icmp eq i32 %13, 0
  br label %bb11

bb11:                                             ; preds = %bb7, %bb18
  %v7837 = phi i32 [ %v77, %bb7 ], [ %v95, %bb18 ]
  %v81 = sitofp i32 %v7837 to float
  %v82 = fmul contract float %v13, %v81
  %v83 = fmul contract float %v71, %v82
  %v84 = fadd contract float %v53, %v83
  %v85 = fmul contract float %v74, %v84
  %v86 = tail call float @llvm.floor.f32(float %v85) #7
  %v100 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v86) #7
  %v101 = fmul contract float %v72, %v82
  %v102 = fadd contract float %v55, %v101
  %v103 = fmul contract float %v74, %v102
  %v104 = tail call float @llvm.floor.f32(float %v103) #7
  %v105 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v104) #7
  %v106 = fmul contract float %v73, %v82
  %v107 = fadd contract float %v57, %v106
  %v108 = fmul contract float %v74, %v107
  %v109 = tail call float @llvm.floor.f32(float %v108) #7
  %v110 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v109) #7
  %v111 = sitofp i32 %v100 to float
  %v112 = fadd contract float %v111, 5.000000e-01
  %v113 = fmul contract float %v13, %v112
  %v114 = sitofp i32 %v105 to float
  %v115 = fadd contract float %v114, 5.000000e-01
  %v116 = fmul contract float %v13, %v115
  %v117 = sitofp i32 %v110 to float
  %v118 = fadd contract float %v117, 5.000000e-01
  %v119 = fmul contract float %v13, %v118
  %v120 = fsub contract float %v113, %v15
  %v121 = fsub contract float %v116, %v16
  %v122 = fsub contract float %v119, %v17
  %v123 = fmul contract float %v120, %v120
  %v124 = fmul contract float %v121, %v121
  %v125 = fadd contract float %v123, %v124
  %v126 = fmul contract float %v122, %v122
  %v127 = fadd contract float %v126, %v125
  br i1 %.not.i, label %19, label %14

14:                                               ; preds = %bb11
  br i1 %.not1.i20, label %17, label %15

15:                                               ; preds = %14
  %16 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v127) #8
  br label %__nv_sqrtf.exit21

17:                                               ; preds = %14
  %18 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v127) #8
  br label %__nv_sqrtf.exit21

19:                                               ; preds = %bb11
  br i1 %.not1.i20, label %22, label %20

20:                                               ; preds = %19
  %21 = tail call float @llvm.nvvm.sqrt.rn.f(float %v127) #8
  br label %__nv_sqrtf.exit21

22:                                               ; preds = %19
  %23 = tail call float @llvm.nvvm.sqrt.approx.f(float %v127) #8
  br label %__nv_sqrtf.exit21

__nv_sqrtf.exit21:                                ; preds = %15, %17, %20, %22
  %.0.i19 = phi float [ %16, %15 ], [ %18, %17 ], [ %21, %20 ], [ %23, %22 ]
  %v129 = fsub contract float %.0.i, %.0.i19
  %v131 = fcmp ult float %v129, %v130
  br i1 %v131, label %bb18, label %bb12

bb12:                                             ; preds = %__nv_sqrtf.exit21
  %v10.i = sdiv i32 %v100, 8
  %24 = and i32 %v100, 7
  %v12.i = icmp eq i32 %24, 0
  %v0.lobit.i = ashr i32 %v100, 31
  %spec.select.i = select i1 %v12.i, i32 0, i32 %v0.lobit.i
  %v18.i1 = add nsw i32 %spec.select.i, %v10.i
  %v10.i2 = sdiv i32 %v105, 8
  %25 = and i32 %v105, 7
  %v12.i3 = icmp eq i32 %25, 0
  %v0.lobit.i4 = ashr i32 %v105, 31
  %spec.select.i5 = select i1 %v12.i3, i32 0, i32 %v0.lobit.i4
  %v18.i6 = add nsw i32 %spec.select.i5, %v10.i2
  %v10.i7 = sdiv i32 %v110, 8
  %26 = and i32 %v110, 7
  %v12.i8 = icmp eq i32 %26, 0
  %v0.lobit.i9 = ashr i32 %v110, 31
  %spec.select.i10 = select i1 %v12.i8, i32 0, i32 %v0.lobit.i9
  %v18.i11 = add nsw i32 %spec.select.i10, %v10.i7
  %narrow.i = add nsw i32 %v18.i1, 1048576
  %v19.i = zext i32 %narrow.i to i64
  %v21.i = shl i64 %v19.i, 42
  %narrow1.i = add nsw i32 %v18.i6, 1048576
  %v23.i = sext i32 %narrow1.i to i64
  %v25.i = shl nsw i64 %v23.i, 21
  %narrow2.i = add nsw i32 %v18.i11, 1048576
  %v28.i = sext i32 %narrow2.i to i64
  %v26.i = or i64 %v25.i, %v28.i
  %v29.i = or i64 %v26.i, %v21.i
  %v32.i = mul i32 %v18.i1, 73856093
  %v34.i = mul i32 %v18.i6, 19349663
  %v35.i = xor i32 %v32.i, %v34.i
  %v37.i = mul i32 %v18.i11, 83492791
  %v38.i = xor i32 %v35.i, %v37.i
  %v39.i = and i32 %v38.i, %v12
  br i1 %v41.not.i32.not, label %bb28.i, label %bb2.i

bb2.i:                                            ; preds = %bb12, %bb27.i
  %v40.i33 = phi i32 [ %v85.i, %bb27.i ], [ 0, %bb12 ]
  %v43.i = add i32 %v40.i33, %v39.i
  %v44.i = and i32 %v43.i, %v12
  %v45.i = zext i32 %v44.i to i64
  %v47.idx.i = shl nuw nsw i64 %v45.i, 4
  %v47.i = getelementptr inbounds nuw i8, ptr %v2, i64 %v47.idx.i
  %v51.i = load atomic i64, ptr %v47.i syncscope("device") acquire, align 8
  %v52.not.i = icmp eq i64 %v51.i, %v29.i
  br i1 %v52.not.i, label %bb4.i, label %bb9.i

bb4.i:                                            ; preds = %bb2.i
  %v48.i.le30 = getelementptr inbounds nuw i8, ptr %v47.i, i64 8
  %v55.i = load atomic i32, ptr %v48.i.le30 syncscope("device") acquire, align 4
  %v57.i35 = icmp sgt i32 %v55.i, -1
  br i1 %v57.i35, label %bb18, label %bb6.i

bb6.i:                                            ; preds = %bb4.i, %bb6.i
  %v59.i = load atomic i32, ptr %v48.i.le30 syncscope("device") acquire, align 4
  %v57.i = icmp sgt i32 %v59.i, -1
  br i1 %v57.i, label %bb18, label %bb6.i

bb9.i:                                            ; preds = %bb2.i
  %v60.i = icmp eq i64 %v51.i, -1
  br i1 %v60.i, label %bb10.i, label %bb27.i

bb10.i:                                           ; preds = %bb9.i
  %v61.i = cmpxchg ptr %v47.i, i64 -1, i64 %v29.i syncscope("device") acq_rel acquire, align 8
  %v91.i = extractvalue { i64, i1 } %v61.i, 1
  br i1 %v91.i, label %bb13.i, label %bb37.i

bb13.i:                                           ; preds = %bb10.i
  fence syncscope("device") release
  %v67.i = atomicrmw add ptr %v4, i32 1 syncscope("device") monotonic, align 4
  fence syncscope("device") acquire
  %v68.not.i = icmp slt i32 %v67.i, %v11
  br i1 %v68.not.i, label %bb19.i, label %bb15.i

bb15.i:                                           ; preds = %bb13.i
  fence syncscope("device") release
  %v70.i = atomicrmw sub ptr %v4, i32 1 syncscope("device") monotonic, align 4
  fence syncscope("device") acquire
  store atomic i64 -1, ptr %v47.i syncscope("device") release, align 8
  %v72.i = atomicrmw add ptr %v8, i32 1 syncscope("device") monotonic, align 4
  br label %bb18

bb19.i:                                           ; preds = %bb13.i
  %v48.i.le28 = getelementptr inbounds nuw i8, ptr %v47.i, i64 8
  %v73.i = mul i32 %v67.i, 3
  %v74.i = sext i32 %v73.i to i64
  %v75.i = getelementptr inbounds i32, ptr %v6, i64 %v74.i
  store i32 %v18.i1, ptr %v75.i, align 4
  %v76.i = getelementptr inbounds nuw i8, ptr %v75.i, i64 4
  store i32 %v18.i6, ptr %v76.i, align 4
  %v77.i = getelementptr inbounds nuw i8, ptr %v75.i, i64 8
  store i32 %v18.i11, ptr %v77.i, align 4
  store atomic i32 %v67.i, ptr %v48.i.le28 syncscope("device") release, align 4
  br label %bb18

bb21.i:                                           ; preds = %bb37.i
  %v48.i.le = getelementptr inbounds nuw i8, ptr %v47.i, i64 8
  %v80.i = load atomic i32, ptr %v48.i.le syncscope("device") acquire, align 4
  %v82.i34 = icmp sgt i32 %v80.i, -1
  br i1 %v82.i34, label %bb18, label %bb23.i

bb23.i:                                           ; preds = %bb21.i, %bb23.i
  %v84.i = load atomic i32, ptr %v48.i.le syncscope("device") acquire, align 4
  %v82.i = icmp sgt i32 %v84.i, -1
  br i1 %v82.i, label %bb18, label %bb23.i

bb27.i:                                           ; preds = %bb37.i, %bb9.i
  %v85.i = add nuw i32 %v40.i33, 1
  %exitcond.not = icmp eq i32 %v40.i33, %v12
  br i1 %exitcond.not, label %bb28.i, label %bb2.i

bb28.i:                                           ; preds = %bb27.i, %bb12
  %v87.i = atomicrmw add ptr %v8, i32 1 syncscope("device") monotonic, align 4
  br label %bb18

bb37.i:                                           ; preds = %bb10.i
  %v62.i = extractvalue { i64, i1 } %v61.i, 0
  %v64.not.i = icmp eq i64 %v62.i, %v29.i
  br i1 %v64.not.i, label %bb21.i, label %bb27.i

bb18:                                             ; preds = %bb23.i, %bb6.i, %bb21.i, %bb4.i, %bb28.i, %bb19.i, %bb15.i, %__nv_sqrtf.exit21
  %v95 = add i32 %v7837, 1
  %v79.not = icmp sgt i32 %v95, %v12.i22
  br i1 %v79.not, label %bb21, label %bb11

bb21:                                             ; preds = %bb18, %bb3, %__nv_sqrtf.exit, %entry
  ret void
}

; Function Attrs: convergent nofree nounwind memory(argmem: readwrite)
define ptx_kernel void @update_kernel(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr captures(none) %v4, i64 %v5, ptr captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, float %v10, float %v11, float %v12, float %v13, float %v14, float %v15, float %v16) #1 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i16 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #7
  %v6.i17 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #7
  %v5.i18 = icmp eq i32 %v4.i16, 1
  %v7.i19 = icmp eq i32 %v6.i17, 1
  %v8.not.not.i = and i1 %v5.i18, %v7.i19
  %v13.i20 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #7
  %v14.i = icmp eq i32 %v13.i20, 1
  %v15.i21 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #7
  %v16.i = icmp eq i32 %v15.i21, 1
  %v17.i22 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i22
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v41 = trunc i64 %v22.i to i32
  %v42.not = icmp sgt i32 %v8, %v41
  br i1 %v42.not, label %bb3, label %bb31

bb3:                                              ; preds = %entry
  %sext = mul i64 %v22.i, 12884901888
  %0 = ashr exact i64 %sext, 30
  %v47 = getelementptr inbounds i8, ptr %v0, i64 %0
  %v48 = load float, ptr %v47, align 4
  %v49 = getelementptr inbounds nuw i8, ptr %v47, i64 4
  %v50 = load float, ptr %v49, align 4
  %v51 = getelementptr inbounds nuw i8, ptr %v47, i64 8
  %v52 = load float, ptr %v51, align 4
  %v53 = fsub contract float %v48, %v13
  %v54 = fsub contract float %v50, %v14
  %v55 = fsub contract float %v52, %v15
  %v56 = fmul contract float %v53, %v53
  %v57 = fmul contract float %v54, %v54
  %v58 = fadd contract float %v56, %v57
  %v59 = fmul contract float %v55, %v55
  %v60 = fadd contract float %v58, %v59
  %v61 = fcmp ule float %v16, 0.000000e+00
  %v63 = fcmp ule float %v60, %v16
  %or.cond = select i1 %v61, i1 true, i1 %v63
  br i1 %or.cond, label %bb6, label %bb31

bb6:                                              ; preds = %bb3
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #8
  %.not.i = icmp eq i32 %1, 0
  %2 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #8
  %.not1.i = icmp eq i32 %2, 0
  br i1 %.not.i, label %8, label %3

3:                                                ; preds = %bb6
  br i1 %.not1.i, label %6, label %4

4:                                                ; preds = %3
  %5 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v60) #8
  br label %__nv_sqrtf.exit

6:                                                ; preds = %3
  %7 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v60) #8
  br label %__nv_sqrtf.exit

8:                                                ; preds = %bb6
  br i1 %.not1.i, label %11, label %9

9:                                                ; preds = %8
  %10 = tail call float @llvm.nvvm.sqrt.rn.f(float %v60) #8
  br label %__nv_sqrtf.exit

11:                                               ; preds = %8
  %12 = tail call float @llvm.nvvm.sqrt.approx.f(float %v60) #8
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %4, %6, %9, %11
  %.0.i = phi float [ %5, %4 ], [ %7, %6 ], [ %10, %9 ], [ %12, %11 ]
  %v121 = fcmp ule float %.0.i, 0x3EB0C6F7A0000000
  br i1 %v121, label %bb31, label %bb7

bb7:                                              ; preds = %__nv_sqrtf.exit
  %v66 = fdiv contract float %v53, %.0.i
  %v67 = fdiv contract float %v54, %.0.i
  %v68 = fdiv contract float %v55, %.0.i
  %v69 = fdiv contract float 1.000000e+00, %v10
  %v70 = fdiv contract float 1.000000e+00, %v11
  %v71 = fmul contract float %v11, %v69
  %v72 = tail call float @llvm.ceil.f32(float %v71) #7
  %v123 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v72) #7
  %v12.i28 = tail call range(i32 1, -2147483648) i32 @llvm.smax.i32(i32 %v123, i32 1)
  %v73 = sub nsw i32 0, %v12.i28
  %v155 = fneg float %v11
  %v33.not.i31.not = icmp eq i32 %v9, -1
  %v108 = fcmp ule float %v12, 0.000000e+00
  %13 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #8
  %.not1.i26 = icmp eq i32 %13, 0
  br label %bb11

bb11:                                             ; preds = %bb7, %bb10.backedge
  %v7435 = phi i32 [ %v73, %bb7 ], [ %v74.be, %bb10.backedge ]
  %v77 = sitofp i32 %v7435 to float
  %v78 = fmul contract float %v10, %v77
  %v79 = fmul contract float %v66, %v78
  %v80 = fadd contract float %v48, %v79
  %v81 = fmul contract float %v69, %v80
  %v82 = tail call float @llvm.floor.f32(float %v81) #7
  %v125 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v82) #7
  %v126 = fmul contract float %v67, %v78
  %v127 = fadd contract float %v50, %v126
  %v128 = fmul contract float %v69, %v127
  %v129 = tail call float @llvm.floor.f32(float %v128) #7
  %v130 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v129) #7
  %v131 = fmul contract float %v68, %v78
  %v132 = fadd contract float %v52, %v131
  %v133 = fmul contract float %v69, %v132
  %v134 = tail call float @llvm.floor.f32(float %v133) #7
  %v135 = tail call i32 @llvm.fptosi.sat.i32.f32(float %v134) #7
  %v136 = sitofp i32 %v125 to float
  %v137 = fadd contract float %v136, 5.000000e-01
  %v138 = fmul contract float %v10, %v137
  %v139 = sitofp i32 %v130 to float
  %v140 = fadd contract float %v139, 5.000000e-01
  %v141 = fmul contract float %v10, %v140
  %v142 = sitofp i32 %v135 to float
  %v143 = fadd contract float %v142, 5.000000e-01
  %v144 = fmul contract float %v10, %v143
  %v145 = fsub contract float %v138, %v13
  %v146 = fsub contract float %v141, %v14
  %v147 = fsub contract float %v144, %v15
  %v148 = fmul contract float %v145, %v145
  %v149 = fmul contract float %v146, %v146
  %v150 = fadd contract float %v148, %v149
  %v151 = fmul contract float %v147, %v147
  %v152 = fadd contract float %v151, %v150
  br i1 %.not.i, label %19, label %14

14:                                               ; preds = %bb11
  br i1 %.not1.i26, label %17, label %15

15:                                               ; preds = %14
  %16 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v152) #8
  br label %__nv_sqrtf.exit27

17:                                               ; preds = %14
  %18 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v152) #8
  br label %__nv_sqrtf.exit27

19:                                               ; preds = %bb11
  br i1 %.not1.i26, label %22, label %20

20:                                               ; preds = %19
  %21 = tail call float @llvm.nvvm.sqrt.rn.f(float %v152) #8
  br label %__nv_sqrtf.exit27

22:                                               ; preds = %19
  %23 = tail call float @llvm.nvvm.sqrt.approx.f(float %v152) #8
  br label %__nv_sqrtf.exit27

__nv_sqrtf.exit27:                                ; preds = %15, %17, %20, %22
  %.0.i25 = phi float [ %16, %15 ], [ %18, %17 ], [ %21, %20 ], [ %23, %22 ]
  %v154 = fsub contract float %.0.i, %.0.i25
  %v156 = fcmp uge float %v154, %v155
  br i1 %v156, label %bb13, label %bb10.backedge

bb10.backedge:                                    ; preds = %tsdf_rust_cuda__kernels__find_block.exit, %bb19, %bb23, %__nv_sqrtf.exit27
  %v74.be = add i32 %v7435, 1
  %v75.not = icmp sgt i32 %v74.be, %v12.i28
  br i1 %v75.not, label %bb31, label %bb11

bb13:                                             ; preds = %__nv_sqrtf.exit27
  %v10.i = sdiv i32 %v125, 8
  %24 = and i32 %v125, 7
  %v12.i = icmp eq i32 %24, 0
  %v0.lobit.i = ashr i32 %v125, 31
  %spec.select.i = select i1 %v12.i, i32 0, i32 %v0.lobit.i
  %v18.i1 = add nsw i32 %spec.select.i, %v10.i
  %v10.i2 = sdiv i32 %v130, 8
  %25 = and i32 %v130, 7
  %v12.i3 = icmp eq i32 %25, 0
  %v0.lobit.i4 = ashr i32 %v130, 31
  %spec.select.i5 = select i1 %v12.i3, i32 0, i32 %v0.lobit.i4
  %v18.i6 = add nsw i32 %spec.select.i5, %v10.i2
  %v10.i7 = sdiv i32 %v135, 8
  %26 = and i32 %v135, 7
  %v12.i8 = icmp eq i32 %26, 0
  %v0.lobit.i9 = ashr i32 %v135, 31
  %spec.select.i10 = select i1 %v12.i8, i32 0, i32 %v0.lobit.i9
  %v18.i11 = add nsw i32 %spec.select.i10, %v10.i7
  %narrow.i = add nsw i32 %v18.i1, 1048576
  %v11.i = zext i32 %narrow.i to i64
  %v13.i = shl i64 %v11.i, 42
  %narrow1.i = add nsw i32 %v18.i6, 1048576
  %v15.i = sext i32 %narrow1.i to i64
  %v17.i12 = shl nsw i64 %v15.i, 21
  %narrow2.i = add nsw i32 %v18.i11, 1048576
  %v20.i14 = sext i32 %narrow2.i to i64
  %v18.i13 = or i64 %v17.i12, %v20.i14
  %v21.i = or i64 %v18.i13, %v13.i
  %v24.i = mul i32 %v18.i1, 73856093
  %v26.i = mul i32 %v18.i6, 19349663
  %v27.i = xor i32 %v24.i, %v26.i
  %v29.i = mul i32 %v18.i11, 83492791
  %v30.i = xor i32 %v27.i, %v29.i
  %v31.i = and i32 %v30.i, %v9
  br i1 %v33.not.i31.not, label %tsdf_rust_cuda__kernels__find_block.exit, label %bb2.i

bb1.i:                                            ; preds = %bb5.i
  %v49.i = add nuw i32 %v32.i32, 1
  %exitcond.not = icmp eq i32 %v32.i32, %v9
  br i1 %exitcond.not, label %tsdf_rust_cuda__kernels__find_block.exit, label %bb2.i

bb2.i:                                            ; preds = %bb13, %bb1.i
  %v32.i32 = phi i32 [ %v49.i, %bb1.i ], [ 0, %bb13 ]
  %v35.i = add i32 %v32.i32, %v31.i
  %v36.i = and i32 %v35.i, %v9
  %v37.i = zext i32 %v36.i to i64
  %v39.idx.i = shl nuw nsw i64 %v37.i, 4
  %v39.i = getelementptr inbounds nuw i8, ptr %v2, i64 %v39.idx.i
  %v43.i = load atomic i64, ptr %v39.i syncscope("device") acquire, align 8
  %v44.i = icmp eq i64 %v43.i, -1
  br i1 %v44.i, label %tsdf_rust_cuda__kernels__find_block.exit, label %bb5.i

bb5.i:                                            ; preds = %bb2.i
  %v45.not.i = icmp eq i64 %v43.i, %v21.i
  br i1 %v45.not.i, label %bb6.i, label %bb1.i

bb6.i:                                            ; preds = %bb5.i
  %v40.i.le = getelementptr inbounds nuw i8, ptr %v39.i, i64 8
  %v48.i = load atomic i32, ptr %v40.i.le syncscope("device") acquire, align 4
  %v51.i33 = icmp sgt i32 %v48.i, -1
  br i1 %v51.i33, label %tsdf_rust_cuda__kernels__find_block.exit, label %bb9.i

bb9.i:                                            ; preds = %bb6.i, %bb9.i
  %v53.i = load atomic i32, ptr %v40.i.le syncscope("device") acquire, align 4
  %v51.i = icmp sgt i32 %v53.i, -1
  br i1 %v51.i, label %tsdf_rust_cuda__kernels__find_block.exit, label %bb9.i

tsdf_rust_cuda__kernels__find_block.exit:         ; preds = %bb2.i, %bb1.i, %bb9.i, %bb13, %bb6.i
  %v55.i = phi i32 [ -1, %bb13 ], [ %v48.i, %bb6.i ], [ %v53.i, %bb9.i ], [ -1, %bb1.i ], [ -1, %bb2.i ]
  %v89 = icmp slt i32 %v55.i, 0
  br i1 %v89, label %bb10.backedge, label %bb18

bb18:                                             ; preds = %tsdf_rust_cuda__kernels__find_block.exit
  %v97 = shl i32 %v55.i, 9
  %27 = shl i32 %v135, 3
  %v94 = add i32 %27, %v130
  %28 = shl i32 %v18.i11, 6
  %29 = shl i32 %v18.i6, 3
  %30 = add i32 %28, %29
  %v99 = sub i32 %v94, %30
  %v100 = shl i32 %v99, 3
  %31 = shl i32 %v18.i1, 3
  %v101 = sub i32 %v125, %31
  %v92 = add i32 %v101, %v100
  %v102 = add i32 %v92, %v97
  %v103 = sext i32 %v102 to i64
  %v105 = getelementptr inbounds float, ptr %v6, i64 %v103
  br i1 %v108, label %bb23, label %bb19

bb19:                                             ; preds = %bb18
  %v110 = load atomic float, ptr %v105 syncscope("device") monotonic, align 4
  %v111 = fcmp ult float %v110, %v12
  br i1 %v111, label %bb23, label %bb10.backedge

bb23:                                             ; preds = %bb19, %bb18
  %v113 = fmul contract float %v70, %v154
  %v8.inv.i = fcmp olt float %v113, -1.000000e+00
  %v0.v1.i = select i1 %v8.inv.i, float -1.000000e+00, float %v113
  %v12.inv.i = fcmp ogt float %v0.v1.i, 1.000000e+00
  %v14.i29 = select i1 %v12.inv.i, float 1.000000e+00, float %v0.v1.i
  %v115 = atomicrmw fadd ptr %v105, float 1.000000e+00 syncscope("device") monotonic, align 4
  %v117 = getelementptr inbounds float, ptr %v4, i64 %v103
  %v119 = atomicrmw fadd ptr %v117, float %v14.i29 syncscope("device") monotonic, align 4
  br label %bb10.backedge

bb31:                                             ; preds = %bb10.backedge, %bb3, %__nv_sqrtf.exit, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.ceil.f32(float) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.floor.f32(float) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fptosi.sat.i32.f32(float) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #3

; Function Attrs: nofree nosync nounwind memory(none)
declare noundef i32 @__nvvm_reflect(ptr noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #6

attributes #0 = { convergent nofree nounwind }
attributes #1 = { convergent nofree nounwind memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent }
attributes #8 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
