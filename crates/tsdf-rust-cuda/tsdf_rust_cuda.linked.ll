; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [2 x ptr] [ptr @alloc_kernel, ptr @update_kernel], section "llvm.metadata"
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1

; Function Attrs: convergent
define ptx_kernel void @alloc_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
entry:
  %v19 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v1, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v3, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v5, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v7, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v9, 1
  br label %bb0

bb0:                                              ; preds = %entry
  %v29 = phi { ptr, i64 } [ %v20, %entry ]
  %v30 = phi { ptr, i64 } [ %v22, %entry ]
  %v31 = phi { ptr, i64 } [ %v24, %entry ]
  %v32 = phi { ptr, i64 } [ %v26, %entry ]
  %v33 = phi { ptr, i64 } [ %v28, %entry ]
  %v34 = phi i32 [ %v10, %entry ]
  %v35 = phi i32 [ %v11, %entry ]
  %v36 = phi i32 [ %v12, %entry ]
  %v37 = phi float [ %v13, %entry ]
  %v38 = phi float [ %v14, %entry ]
  %v39 = phi float [ %v15, %entry ]
  %v40 = phi float [ %v16, %entry ]
  %v41 = phi float [ %v17, %entry ]
  %v42 = phi float [ %v18, %entry ]
  %v43 = alloca {}, align 1
  %v44 = bitcast ptr %v43 to ptr
  %v45 = call i64 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECskrHgD3sK77p_14tsdf_rust_cuda(ptr %v44) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v46 = trunc i64 %v45 to i32
  %v47 = icmp sge i32 %v46, %v34
  %v48 = xor i1 %v47, true
  br i1 %v48, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  br label %bb21

bb3:                                              ; preds = %bb1
  %v49 = extractvalue { ptr, i64 } %v29, 0
  %v50 = mul i32 %v46, 3
  %v51 = sext i32 %v50 to i64
  %v52 = getelementptr inbounds float, ptr %v49, i64 %v51
  %v53 = load float, ptr %v52, align 4
  %v54 = getelementptr inbounds float, ptr %v52, i64 1
  %v55 = load float, ptr %v54, align 4
  %v56 = getelementptr inbounds float, ptr %v52, i64 2
  %v57 = load float, ptr %v56, align 4
  %v58 = fsub contract float %v53, %v39
  %v59 = fsub contract float %v55, %v40
  %v60 = fsub contract float %v57, %v41
  %v61 = fmul contract float %v58, %v58
  %v62 = fmul contract float %v59, %v59
  %v63 = fadd contract float %v61, %v62
  %v64 = fmul contract float %v60, %v60
  %v65 = fadd contract float %v63, %v64
  %v66 = fcmp ogt float %v42, 0.000000e+00
  %v67 = xor i1 %v66, true
  br i1 %v67, label %bb6, label %bb4

bb4:                                              ; preds = %bb3
  %v68 = fcmp ogt float %v65, %v42
  %v69 = xor i1 %v68, true
  br i1 %v69, label %bb6, label %bb5

bb5:                                              ; preds = %bb4
  br label %bb20

bb6:                                              ; preds = %bb4, %bb3
  %v70 = call float @__nv_sqrtf(float %v65) #0
  br label %bb22

bb7:                                              ; preds = %bb22
  %v71 = fdiv contract float %v58, %v70
  %v72 = fdiv contract float %v59, %v70
  %v73 = fdiv contract float %v60, %v70
  %v74 = fdiv contract float 1.000000e+00, %v37
  %v75 = fmul contract float %v38, %v74
  %v76 = call float @llvm.ceil.f32(float %v75) #0
  br label %bb23

bb8:                                              ; preds = %bb22
  br label %bb20

bb9:                                              ; preds = %bb23
  %v77 = sub i32 0, %v99
  br label %bb10

bb10:                                             ; preds = %bb18, %bb9
  %v78 = phi i32 [ %v77, %bb9 ], [ %v95, %bb18 ]
  %v79 = icmp sle i32 %v78, %v99
  %v80 = xor i1 %v79, true
  br i1 %v80, label %bb19, label %bb11

bb11:                                             ; preds = %bb10
  %v81 = sitofp i32 %v78 to float
  %v82 = fmul contract float %v81, %v37
  %v83 = fmul contract float %v71, %v82
  %v84 = fadd contract float %v53, %v83
  %v85 = fmul contract float %v84, %v74
  %v86 = call float @llvm.floor.f32(float %v85) #0
  br label %bb24

bb12:                                             ; preds = %bb27
  %v87 = extractvalue { ptr, i64 } %v30, 0
  %v88 = extractvalue { ptr, i64 } %v31, 0
  %v89 = extractvalue { ptr, i64 } %v33, 0
  %v90 = extractvalue { ptr, i64 } %v32, 0
  %v91 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v100, i32 8) #0
  br label %bb13

bb13:                                             ; preds = %bb12
  %v92 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v105, i32 8) #0
  br label %bb14

bb14:                                             ; preds = %bb13
  %v93 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v110, i32 8) #0
  br label %bb15

bb15:                                             ; preds = %bb14
  %v94 = call i32 @tsdf_rust_cuda__kernels__find_or_insert(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
  br label %bb16

bb16:                                             ; preds = %bb15
  br label %bb18

bb17:                                             ; preds = %bb27
  br label %bb18

bb18:                                             ; preds = %bb17, %bb16
  %v95 = add i32 %v78, 1
  br label %bb10

bb19:                                             ; preds = %bb10
  br label %bb21

bb20:                                             ; preds = %bb8, %bb5
  br label %bb21

bb21:                                             ; preds = %bb20, %bb19, %bb2
  ret void

bb22:                                             ; preds = %bb6
  %v96 = fcmp ogt float %v70, 0x3EB0C6F7A0000000
  %v97 = xor i1 %v96, true
  br i1 %v97, label %bb8, label %bb7

bb23:                                             ; preds = %bb7
  %v98 = call i32 @llvm.fptosi.sat.i32.f32(float %v76) #0
  %v99 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCskrHgD3sK77p_14tsdf_rust_cuda(i32 %v98, i32 1) #0
  br label %bb9

bb24:                                             ; preds = %bb11
  %v100 = call i32 @llvm.fptosi.sat.i32.f32(float %v86) #0
  %v101 = fmul contract float %v72, %v82
  %v102 = fadd contract float %v55, %v101
  %v103 = fmul contract float %v102, %v74
  %v104 = call float @llvm.floor.f32(float %v103) #0
  br label %bb25

bb25:                                             ; preds = %bb24
  %v105 = call i32 @llvm.fptosi.sat.i32.f32(float %v104) #0
  %v106 = fmul contract float %v73, %v82
  %v107 = fadd contract float %v57, %v106
  %v108 = fmul contract float %v107, %v74
  %v109 = call float @llvm.floor.f32(float %v108) #0
  br label %bb26

bb26:                                             ; preds = %bb25
  %v110 = call i32 @llvm.fptosi.sat.i32.f32(float %v109) #0
  %v111 = sitofp i32 %v100 to float
  %v112 = fadd contract float %v111, 5.000000e-01
  %v113 = fmul contract float %v112, %v37
  %v114 = sitofp i32 %v105 to float
  %v115 = fadd contract float %v114, 5.000000e-01
  %v116 = fmul contract float %v115, %v37
  %v117 = sitofp i32 %v110 to float
  %v118 = fadd contract float %v117, 5.000000e-01
  %v119 = fmul contract float %v118, %v37
  %v120 = fsub contract float %v113, %v39
  %v121 = fsub contract float %v116, %v40
  %v122 = fsub contract float %v119, %v41
  %v123 = fmul contract float %v120, %v120
  %v124 = fmul contract float %v121, %v121
  %v125 = fadd contract float %v123, %v124
  %v126 = fmul contract float %v122, %v122
  %v127 = fadd contract float %v125, %v126
  %v128 = call float @__nv_sqrtf(float %v127) #0
  br label %bb27

bb27:                                             ; preds = %bb26
  %v129 = fsub contract float %v70, %v128
  %v130 = fneg float %v38
  %v131 = fcmp oge float %v129, %v130
  %v132 = xor i1 %v131, true
  br i1 %v132, label %bb17, label %bb12
}

; Function Attrs: convergent
define ptx_kernel void @update_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, float %v10, float %v11, float %v12, float %v13, float %v14, float %v15, float %v16) #0 {
entry:
  %v17 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v1, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v3, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v5, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v7, 1
  br label %bb0

bb0:                                              ; preds = %entry
  %v25 = phi { ptr, i64 } [ %v18, %entry ]
  %v26 = phi { ptr, i64 } [ %v20, %entry ]
  %v27 = phi { ptr, i64 } [ %v22, %entry ]
  %v28 = phi { ptr, i64 } [ %v24, %entry ]
  %v29 = phi i32 [ %v8, %entry ]
  %v30 = phi i32 [ %v9, %entry ]
  %v31 = phi float [ %v10, %entry ]
  %v32 = phi float [ %v11, %entry ]
  %v33 = phi float [ %v12, %entry ]
  %v34 = phi float [ %v13, %entry ]
  %v35 = phi float [ %v14, %entry ]
  %v36 = phi float [ %v15, %entry ]
  %v37 = phi float [ %v16, %entry ]
  %v38 = alloca {}, align 1
  %v39 = bitcast ptr %v38 to ptr
  %v40 = call i64 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECskrHgD3sK77p_14tsdf_rust_cuda(ptr %v39) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v41 = trunc i64 %v40 to i32
  %v42 = icmp sge i32 %v41, %v29
  %v43 = xor i1 %v42, true
  br i1 %v43, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  br label %bb31

bb3:                                              ; preds = %bb1
  %v44 = extractvalue { ptr, i64 } %v25, 0
  %v45 = mul i32 %v41, 3
  %v46 = sext i32 %v45 to i64
  %v47 = getelementptr inbounds float, ptr %v44, i64 %v46
  %v48 = load float, ptr %v47, align 4
  %v49 = getelementptr inbounds float, ptr %v47, i64 1
  %v50 = load float, ptr %v49, align 4
  %v51 = getelementptr inbounds float, ptr %v47, i64 2
  %v52 = load float, ptr %v51, align 4
  %v53 = fsub contract float %v48, %v34
  %v54 = fsub contract float %v50, %v35
  %v55 = fsub contract float %v52, %v36
  %v56 = fmul contract float %v53, %v53
  %v57 = fmul contract float %v54, %v54
  %v58 = fadd contract float %v56, %v57
  %v59 = fmul contract float %v55, %v55
  %v60 = fadd contract float %v58, %v59
  %v61 = fcmp ogt float %v37, 0.000000e+00
  %v62 = xor i1 %v61, true
  br i1 %v62, label %bb6, label %bb4

bb4:                                              ; preds = %bb3
  %v63 = fcmp ogt float %v60, %v37
  %v64 = xor i1 %v63, true
  br i1 %v64, label %bb6, label %bb5

bb5:                                              ; preds = %bb4
  br label %bb30

bb6:                                              ; preds = %bb4, %bb3
  %v65 = call float @__nv_sqrtf(float %v60) #0
  br label %bb32

bb7:                                              ; preds = %bb32
  %v66 = fdiv contract float %v53, %v65
  %v67 = fdiv contract float %v54, %v65
  %v68 = fdiv contract float %v55, %v65
  %v69 = fdiv contract float 1.000000e+00, %v31
  %v70 = fdiv contract float 1.000000e+00, %v32
  %v71 = fmul contract float %v32, %v69
  %v72 = call float @llvm.ceil.f32(float %v71) #0
  br label %bb33

bb8:                                              ; preds = %bb32
  br label %bb30

bb9:                                              ; preds = %bb33
  %v73 = sub i32 0, %v124
  br label %bb10

bb10:                                             ; preds = %bb28, %bb12, %bb9
  %v74 = phi i32 [ %v73, %bb9 ], [ %v83, %bb12 ], [ %v120, %bb28 ]
  %v75 = icmp sle i32 %v74, %v124
  %v76 = xor i1 %v75, true
  br i1 %v76, label %bb29, label %bb11

bb11:                                             ; preds = %bb10
  %v77 = sitofp i32 %v74 to float
  %v78 = fmul contract float %v77, %v31
  %v79 = fmul contract float %v66, %v78
  %v80 = fadd contract float %v48, %v79
  %v81 = fmul contract float %v80, %v69
  %v82 = call float @llvm.floor.f32(float %v81) #0
  br label %bb34

bb12:                                             ; preds = %bb37
  %v83 = add i32 %v74, 1
  br label %bb10

bb13:                                             ; preds = %bb37
  %v84 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v125, i32 8) #0
  br label %bb14

bb14:                                             ; preds = %bb13
  %v85 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v130, i32 8) #0
  br label %bb15

bb15:                                             ; preds = %bb14
  %v86 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v135, i32 8) #0
  br label %bb16

bb16:                                             ; preds = %bb15
  %v87 = extractvalue { ptr, i64 } %v26, 0
  %v88 = call i32 @tsdf_rust_cuda__kernels__find_block(ptr %v87, i32 %v30, i32 %v84, i32 %v85, i32 %v86) #0
  br label %bb17

bb17:                                             ; preds = %bb16
  %v89 = icmp sge i32 %v88, 0
  %v90 = xor i1 %v89, true
  br i1 %v90, label %bb28, label %bb18

bb18:                                             ; preds = %bb17
  %v91 = mul i32 %v84, 8
  %v92 = sub i32 %v125, %v91
  %v93 = mul i32 %v85, 8
  %v94 = sub i32 %v130, %v93
  %v95 = mul i32 %v86, 8
  %v96 = sub i32 %v135, %v95
  %v97 = mul i32 %v88, 512
  %v98 = mul i32 %v96, 8
  %v99 = add i32 %v98, %v94
  %v100 = mul i32 %v99, 8
  %v101 = add i32 %v97, %v100
  %v102 = add i32 %v101, %v92
  %v103 = sext i32 %v102 to i64
  %v104 = extractvalue { ptr, i64 } %v28, 0
  %v105 = getelementptr inbounds float, ptr %v104, i64 %v103
  %v106 = bitcast ptr %v105 to ptr
  %v107 = bitcast ptr %v105 to ptr
  %v108 = fcmp ogt float %v33, 0.000000e+00
  %v109 = xor i1 %v108, true
  br i1 %v109, label %bb23, label %bb19

bb19:                                             ; preds = %bb18
  %v110 = load atomic float, ptr %v107 syncscope("device") monotonic, align 4
  br label %bb20

bb20:                                             ; preds = %bb19
  %v111 = fcmp oge float %v110, %v33
  %v112 = xor i1 %v111, true
  br i1 %v112, label %bb22, label %bb21

bb21:                                             ; preds = %bb20
  br label %bb27

bb22:                                             ; preds = %bb20
  br label %bb23

bb23:                                             ; preds = %bb22, %bb18
  %v113 = fmul contract float %v154, %v70
  %v114 = call float @core__f32___impl_f32___clamp(float %v113, float -1.000000e+00, float 1.000000e+00) #0
  br label %bb24

bb24:                                             ; preds = %bb23
  %v115 = atomicrmw fadd ptr %v107, float 1.000000e+00 syncscope("device") monotonic, align 4
  br label %bb25

bb25:                                             ; preds = %bb24
  %v116 = extractvalue { ptr, i64 } %v27, 0
  %v117 = getelementptr inbounds float, ptr %v116, i64 %v103
  %v118 = bitcast ptr %v117 to ptr
  %v119 = atomicrmw fadd ptr %v118, float %v114 syncscope("device") monotonic, align 4
  br label %bb26

bb26:                                             ; preds = %bb25
  br label %bb27

bb27:                                             ; preds = %bb26, %bb21
  br label %bb28

bb28:                                             ; preds = %bb27, %bb17
  %v120 = add i32 %v74, 1
  br label %bb10

bb29:                                             ; preds = %bb10
  br label %bb31

bb30:                                             ; preds = %bb8, %bb5
  br label %bb31

bb31:                                             ; preds = %bb30, %bb29, %bb2
  ret void

bb32:                                             ; preds = %bb6
  %v121 = fcmp ogt float %v65, 0x3EB0C6F7A0000000
  %v122 = xor i1 %v121, true
  br i1 %v122, label %bb8, label %bb7

bb33:                                             ; preds = %bb7
  %v123 = call i32 @llvm.fptosi.sat.i32.f32(float %v72) #0
  %v124 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCskrHgD3sK77p_14tsdf_rust_cuda(i32 %v123, i32 1) #0
  br label %bb9

bb34:                                             ; preds = %bb11
  %v125 = call i32 @llvm.fptosi.sat.i32.f32(float %v82) #0
  %v126 = fmul contract float %v67, %v78
  %v127 = fadd contract float %v50, %v126
  %v128 = fmul contract float %v127, %v69
  %v129 = call float @llvm.floor.f32(float %v128) #0
  br label %bb35

bb35:                                             ; preds = %bb34
  %v130 = call i32 @llvm.fptosi.sat.i32.f32(float %v129) #0
  %v131 = fmul contract float %v68, %v78
  %v132 = fadd contract float %v52, %v131
  %v133 = fmul contract float %v132, %v69
  %v134 = call float @llvm.floor.f32(float %v133) #0
  br label %bb36

bb36:                                             ; preds = %bb35
  %v135 = call i32 @llvm.fptosi.sat.i32.f32(float %v134) #0
  %v136 = sitofp i32 %v125 to float
  %v137 = fadd contract float %v136, 5.000000e-01
  %v138 = fmul contract float %v137, %v31
  %v139 = sitofp i32 %v130 to float
  %v140 = fadd contract float %v139, 5.000000e-01
  %v141 = fmul contract float %v140, %v31
  %v142 = sitofp i32 %v135 to float
  %v143 = fadd contract float %v142, 5.000000e-01
  %v144 = fmul contract float %v143, %v31
  %v145 = fsub contract float %v138, %v34
  %v146 = fsub contract float %v141, %v35
  %v147 = fsub contract float %v144, %v36
  %v148 = fmul contract float %v145, %v145
  %v149 = fmul contract float %v146, %v146
  %v150 = fadd contract float %v148, %v149
  %v151 = fmul contract float %v147, %v147
  %v152 = fadd contract float %v150, %v151
  %v153 = call float @__nv_sqrtf(float %v152) #0
  br label %bb37

bb37:                                             ; preds = %bb36
  %v154 = fsub contract float %v65, %v153
  %v155 = fneg float %v32
  %v156 = fcmp olt float %v154, %v155
  %v157 = xor i1 %v156, true
  br i1 %v157, label %bb13, label %bb12
}

; Function Attrs: alwaysinline convergent
define i64 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECskrHgD3sK77p_14tsdf_rust_cuda(ptr %v0) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2

bb2:                                              ; preds = %bb1
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3

bb3:                                              ; preds = %bb2
  %v5 = zext i32 %v2 to i64
  %v6 = zext i32 %v3 to i64
  %v7 = zext i32 %v4 to i64
  %v8 = icmp eq i64 %v6, 0
  br i1 %v8, label %bb10, label %bb8

bb4:                                              ; preds = %bb12
  %v9 = xor i1 %v20, true
  br i1 %v9, label %bb6, label %bb5

bb5:                                              ; preds = %bb4
  %v10 = icmp ne i64 %v19, -1
  br label %bb7

bb6:                                              ; preds = %bb4
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  %v11 = phi i1 [ %v10, %bb5 ], [ false, %bb6 ]
  %v12 = xor i1 %v11, true
  br i1 %v12, label %bb14, label %bb13

bb8:                                              ; preds = %bb3
  %v13 = sub i64 -1, %v7
  %v14 = udiv i64 %v13, %v6
  %v15 = icmp ugt i64 %v5, %v14
  %v16 = xor i1 %v15, true
  br i1 %v16, label %bb11, label %bb9

bb9:                                              ; preds = %bb8
  br label %bb10

bb10:                                             ; preds = %bb9, %bb3
  br label %bb12

bb11:                                             ; preds = %bb8
  %v17 = mul i64 %v5, %v6
  %v18 = add i64 %v17, %v7
  br label %bb12

bb12:                                             ; preds = %bb11, %bb10
  %v19 = phi i64 [ -1, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECskrHgD3sK77p_14tsdf_rust_cuda(ptr %v1) #0
  br label %bb4

bb13:                                             ; preds = %bb7
  %v21 = icmp eq i64 %v19, -1
  br i1 %v21, label %bb14, label %bb15

bb14:                                             ; preds = %bb13, %bb7
  br label %bb15

bb15:                                             ; preds = %bb14, %bb13
  %v22 = phi i64 [ %v19, %bb13 ], [ -1, %bb14 ]
  ret i64 %v22
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.ceil.f32(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.floor.f32(float) #2

; Function Attrs: alwaysinline convergent
define i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v0, i32 %v1) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = icmp eq i32 %v3, 0
  %v5 = xor i1 %v4, true
  br i1 %v5, label %bb1, label %bb9

bb1:                                              ; preds = %bb0
  %v6 = icmp eq i32 %v3, -1
  %v7 = icmp eq i32 %v2, -2147483648
  %v8 = and i1 %v6, %v7
  %v9 = xor i1 %v8, true
  br i1 %v9, label %bb2, label %bb10

bb2:                                              ; preds = %bb1
  %v10 = sdiv i32 %v2, %v3
  %v11 = srem i32 %v2, %v3
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %bb6, label %bb3

bb3:                                              ; preds = %bb2
  %v13 = icmp slt i32 %v2, 0
  %v14 = icmp slt i32 %v3, 0
  %v15 = icmp ne i1 %v13, %v14
  %v16 = xor i1 %v15, true
  br i1 %v16, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  %v17 = sub i32 %v10, 1
  br label %bb8

bb5:                                              ; preds = %bb3
  br label %bb7

bb6:                                              ; preds = %bb2
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  br label %bb8

bb8:                                              ; preds = %bb7, %bb4
  %v18 = phi i32 [ %v17, %bb4 ], [ %v10, %bb7 ]
  ret i32 %v18

bb9:                                              ; preds = %bb0
  call void @llvm.trap() #0
  unreachable

bb10:                                             ; preds = %bb1
  call void @llvm.trap() #0
  unreachable
}

; Function Attrs: alwaysinline convergent
define i32 @tsdf_rust_cuda__kernels__find_block(ptr %v0, i32 %v1, i32 %v2, i32 %v3, i32 %v4) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v5 = phi ptr [ %v0, %entry ]
  %v6 = phi i32 [ %v1, %entry ]
  %v7 = phi i32 [ %v2, %entry ]
  %v8 = phi i32 [ %v3, %entry ]
  %v9 = phi i32 [ %v4, %entry ]
  %v10 = sext i32 %v7 to i64
  %v11 = add i64 %v10, 1048576
  %v12 = and i64 42, 63
  %v13 = shl i64 %v11, %v12
  %v14 = sext i32 %v8 to i64
  %v15 = add i64 %v14, 1048576
  %v16 = and i64 21, 63
  %v17 = shl i64 %v15, %v16
  %v18 = or i64 %v13, %v17
  %v19 = sext i32 %v9 to i64
  %v20 = add i64 %v19, 1048576
  %v21 = or i64 %v18, %v20
  %v22 = add i32 %v6, 1
  %v23 = bitcast i32 %v7 to i32
  %v24 = mul i32 %v23, 73856093
  %v25 = bitcast i32 %v8 to i32
  %v26 = mul i32 %v25, 19349663
  %v27 = xor i32 %v24, %v26
  %v28 = bitcast i32 %v9 to i32
  %v29 = mul i32 %v28, 83492791
  %v30 = xor i32 %v27, %v29
  %v31 = and i32 %v30, %v6
  br label %bb1

bb1:                                              ; preds = %bb7, %bb0
  %v32 = phi i32 [ 0, %bb0 ], [ %v49, %bb7 ]
  %v33 = icmp ult i32 %v32, %v22
  %v34 = xor i1 %v33, true
  br i1 %v34, label %bb12, label %bb2

bb2:                                              ; preds = %bb1
  %v35 = add i32 %v31, %v32
  %v36 = and i32 %v35, %v6
  %v37 = zext i32 %v36 to i64
  %v38 = mul i64 %v37, 2
  %v39 = getelementptr inbounds i64, ptr %v5, i64 %v38
  %v40 = getelementptr inbounds i64, ptr %v39, i64 1
  %v41 = bitcast ptr %v40 to ptr
  %v42 = bitcast ptr %v39 to ptr
  %v43 = load atomic i64, ptr %v42 syncscope("device") acquire, align 8
  br label %bb3

bb3:                                              ; preds = %bb2
  %v44 = icmp eq i64 %v43, -1
  br i1 %v44, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  br label %bb13

bb5:                                              ; preds = %bb3
  %v45 = icmp eq i64 %v43, %v21
  %v46 = xor i1 %v45, true
  br i1 %v46, label %bb7, label %bb6

bb6:                                              ; preds = %bb5
  %v47 = bitcast ptr %v40 to ptr
  %v48 = load atomic i32, ptr %v47 syncscope("device") acquire, align 4
  br label %bb8

bb7:                                              ; preds = %bb5
  %v49 = add i32 %v32, 1
  br label %bb1

bb8:                                              ; preds = %bb10, %bb6
  %v50 = phi i32 [ %v48, %bb6 ], [ %v53, %bb10 ]
  %v51 = icmp slt i32 %v50, 0
  %v52 = xor i1 %v51, true
  br i1 %v52, label %bb11, label %bb9

bb9:                                              ; preds = %bb8
  %v53 = load atomic i32, ptr %v47 syncscope("device") acquire, align 4
  br label %bb10

bb10:                                             ; preds = %bb9
  br label %bb8

bb11:                                             ; preds = %bb8
  br label %bb13

bb12:                                             ; preds = %bb1
  br label %bb14

bb13:                                             ; preds = %bb11, %bb4
  %v54 = phi i32 [ -1, %bb4 ], [ %v50, %bb11 ]
  br label %bb14

bb14:                                             ; preds = %bb13, %bb12
  %v55 = phi i32 [ -1, %bb12 ], [ %v54, %bb13 ]
  ret i32 %v55
}

; Function Attrs: convergent
define float @core__f32___impl_f32___clamp(float %v0, float %v1, float %v2) #0 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v3 = phi float [ %v0, %entry ]
  %v4 = phi float [ %v1, %entry ]
  %v5 = phi float [ %v2, %entry ]
  %v6 = fcmp ole float %v4, %v5
  %v7 = xor i1 %v6, true
  br i1 %v7, label %bb2, label %bb1

bb1:                                              ; preds = %bb0
  %v8 = fcmp olt float %v3, %v4
  %v9 = xor i1 %v8, true
  br i1 %v9, label %bb4, label %bb3

bb2:                                              ; preds = %bb0
  call void asm sideeffect "trap;", ""()
  unreachable

bb3:                                              ; preds = %bb1
  br label %bb5

bb4:                                              ; preds = %bb1
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3
  %v11 = phi float [ %v4, %bb3 ], [ %v3, %bb4 ]
  %v12 = fcmp ogt float %v11, %v5
  %v13 = xor i1 %v12, true
  br i1 %v13, label %bb7, label %bb6

bb6:                                              ; preds = %bb5
  br label %bb8

bb7:                                              ; preds = %bb5
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  %v14 = phi float [ %v5, %bb6 ], [ %v11, %bb7 ]
  ret float %v14
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fptosi.sat.i32.f32(float) #2

; Function Attrs: convergent
define i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCskrHgD3sK77p_14tsdf_rust_cuda(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = alloca i32, align 4
  %v5 = alloca i32, align 4
  store i32 %v2, ptr %v4, align 4
  store i32 %v3, ptr %v5, align 4
  %v6 = bitcast ptr %v5 to ptr
  %v7 = bitcast ptr %v4 to ptr
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(ptr %v6, ptr %v7) #0
  br label %bb1

bb1:                                              ; preds = %bb0
  %v9 = xor i1 %v8, true
  br i1 %v9, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  %v10 = load i32, ptr %v4, align 4
  br label %bb4

bb3:                                              ; preds = %bb1
  %v11 = load i32, ptr %v5, align 4
  br label %bb4

bb4:                                              ; preds = %bb3, %bb2
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

; Function Attrs: alwaysinline convergent
define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(ptr %v0, ptr %v1) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i32, ptr %v2, align 4
  %v5 = load i32, ptr %v3, align 4
  %v6 = icmp slt i32 %v4, %v5
  ret i1 %v6
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

; Function Attrs: alwaysinline convergent
define i1 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECskrHgD3sK77p_14tsdf_rust_cuda(ptr %v0) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, true
  br i1 %v3, label %bb2, label %bb1

bb1:                                              ; preds = %bb0
  br label %bb8

bb2:                                              ; preds = %bb0
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3

bb3:                                              ; preds = %bb2
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6

bb5:                                              ; preds = %bb3
  br label %bb7

bb6:                                              ; preds = %bb4
  %v7 = icmp eq i32 %v6, 1
  br label %bb7

bb7:                                              ; preds = %bb6, %bb5
  %v8 = phi i1 [ false, %bb5 ], [ %v7, %bb6 ]
  br label %bb8

bb8:                                              ; preds = %bb7, %bb1
  %v9 = phi i1 [ true, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, true
  br i1 %v10, label %bb9, label %bb10

bb9:                                              ; preds = %bb8
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, true
  br i1 %v12, label %bb11, label %bb10

bb10:                                             ; preds = %bb9, %bb8
  br label %bb17

bb11:                                             ; preds = %bb9
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12

bb12:                                             ; preds = %bb11
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14

bb13:                                             ; preds = %bb12
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15

bb14:                                             ; preds = %bb12
  br label %bb16

bb15:                                             ; preds = %bb13
  %v16 = icmp eq i32 %v15, 1
  br label %bb16

bb16:                                             ; preds = %bb15, %bb14
  %v17 = phi i1 [ false, %bb14 ], [ %v16, %bb15 ]
  br label %bb17

bb17:                                             ; preds = %bb16, %bb10
  %v18 = phi i1 [ true, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, true
  br i1 %v19, label %bb19, label %bb18

bb18:                                             ; preds = %bb17
  br label %bb20

bb19:                                             ; preds = %bb17
  br label %bb20

bb20:                                             ; preds = %bb19, %bb18
  %v20 = phi i1 [ %v18, %bb18 ], [ false, %bb19 ]
  ret i1 %v20
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #4

; Function Attrs: alwaysinline convergent
define i32 @tsdf_rust_cuda__kernels__find_or_insert(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) #1 {
entry:
  br label %bb0

bb0:                                              ; preds = %entry
  %v9 = phi ptr [ %v0, %entry ]
  %v10 = phi i32 [ %v1, %entry ]
  %v11 = phi ptr [ %v2, %entry ]
  %v12 = phi i32 [ %v3, %entry ]
  %v13 = phi ptr [ %v4, %entry ]
  %v14 = phi ptr [ %v5, %entry ]
  %v15 = phi i32 [ %v6, %entry ]
  %v16 = phi i32 [ %v7, %entry ]
  %v17 = phi i32 [ %v8, %entry ]
  %v18 = sext i32 %v15 to i64
  %v19 = add i64 %v18, 1048576
  %v20 = and i64 42, 63
  %v21 = shl i64 %v19, %v20
  %v22 = sext i32 %v16 to i64
  %v23 = add i64 %v22, 1048576
  %v24 = and i64 21, 63
  %v25 = shl i64 %v23, %v24
  %v26 = or i64 %v21, %v25
  %v27 = sext i32 %v17 to i64
  %v28 = add i64 %v27, 1048576
  %v29 = or i64 %v26, %v28
  %v30 = add i32 %v10, 1
  %v31 = bitcast i32 %v15 to i32
  %v32 = mul i32 %v31, 73856093
  %v33 = bitcast i32 %v16 to i32
  %v34 = mul i32 %v33, 19349663
  %v35 = xor i32 %v32, %v34
  %v36 = bitcast i32 %v17 to i32
  %v37 = mul i32 %v36, 83492791
  %v38 = xor i32 %v35, %v37
  %v39 = and i32 %v38, %v10
  br label %bb1

bb1:                                              ; preds = %bb27, %bb0
  %v40 = phi i32 [ 0, %bb0 ], [ %v85, %bb27 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, true
  br i1 %v42, label %bb28, label %bb2

bb2:                                              ; preds = %bb1
  %v43 = add i32 %v39, %v40
  %v44 = and i32 %v43, %v10
  %v45 = zext i32 %v44 to i64
  %v46 = mul i64 %v45, 2
  %v47 = getelementptr inbounds i64, ptr %v9, i64 %v46
  %v48 = getelementptr inbounds i64, ptr %v47, i64 1
  %v49 = bitcast ptr %v48 to ptr
  %v50 = bitcast ptr %v47 to ptr
  %v51 = load atomic i64, ptr %v50 syncscope("device") acquire, align 8
  br label %bb3

bb3:                                              ; preds = %bb2
  %v52 = icmp eq i64 %v51, %v29
  %v53 = xor i1 %v52, true
  br i1 %v53, label %bb9, label %bb4

bb4:                                              ; preds = %bb3
  %v54 = bitcast ptr %v48 to ptr
  %v55 = load atomic i32, ptr %v54 syncscope("device") acquire, align 4
  br label %bb5

bb5:                                              ; preds = %bb7, %bb4
  %v56 = phi i32 [ %v55, %bb4 ], [ %v59, %bb7 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, true
  br i1 %v58, label %bb8, label %bb6

bb6:                                              ; preds = %bb5
  %v59 = load atomic i32, ptr %v54 syncscope("device") acquire, align 4
  br label %bb7

bb7:                                              ; preds = %bb6
  br label %bb5

bb8:                                              ; preds = %bb5
  br label %bb31

bb9:                                              ; preds = %bb3
  %v60 = icmp eq i64 %v51, -1
  br i1 %v60, label %bb10, label %bb27

bb10:                                             ; preds = %bb9
  %v61 = cmpxchg ptr %v50, i64 -1, i64 %v29 syncscope("device") acq_rel acquire, align 8
  %v62 = extractvalue { i64, i1 } %v61, 0
  br label %bb33

bb11:                                             ; preds = %bb37
  unreachable

bb12:                                             ; preds = %bb37
  %v63 = extractvalue { i64, i64 } %v103, 1
  %v64 = icmp eq i64 %v63, %v29
  %v65 = xor i1 %v64, true
  br i1 %v65, label %bb26, label %bb21

bb13:                                             ; preds = %bb36
  %v66 = bitcast ptr %v11 to ptr
  fence syncscope("device") release
  %v67 = atomicrmw add ptr %v66, i32 1 syncscope("device") monotonic, align 4
  fence syncscope("device") acquire
  br label %bb14

bb14:                                             ; preds = %bb13
  %v68 = icmp sge i32 %v67, %v12
  %v69 = xor i1 %v68, true
  br i1 %v69, label %bb19, label %bb15

bb15:                                             ; preds = %bb14
  fence syncscope("device") release
  %v70 = atomicrmw sub ptr %v66, i32 1 syncscope("device") monotonic, align 4
  fence syncscope("device") acquire
  br label %bb16

bb16:                                             ; preds = %bb15
  store atomic i64 -1, ptr %v50 syncscope("device") release, align 8
  br label %bb17

bb17:                                             ; preds = %bb16
  %v71 = bitcast ptr %v13 to ptr
  %v72 = atomicrmw add ptr %v71, i32 1 syncscope("device") monotonic, align 4
  br label %bb18

bb18:                                             ; preds = %bb17
  br label %bb30

bb19:                                             ; preds = %bb14
  %v73 = mul i32 %v67, 3
  %v74 = sext i32 %v73 to i64
  %v75 = getelementptr inbounds i32, ptr %v14, i64 %v74
  store i32 %v15, ptr %v75, align 4
  %v76 = getelementptr inbounds i32, ptr %v75, i64 1
  store i32 %v16, ptr %v76, align 4
  %v77 = getelementptr inbounds i32, ptr %v75, i64 2
  store i32 %v17, ptr %v77, align 4
  %v78 = bitcast ptr %v48 to ptr
  store atomic i32 %v67, ptr %v78 syncscope("device") release, align 4
  br label %bb20

bb20:                                             ; preds = %bb19
  br label %bb30

bb21:                                             ; preds = %bb12
  %v79 = bitcast ptr %v48 to ptr
  %v80 = load atomic i32, ptr %v79 syncscope("device") acquire, align 4
  br label %bb22

bb22:                                             ; preds = %bb24, %bb21
  %v81 = phi i32 [ %v80, %bb21 ], [ %v84, %bb24 ]
  %v82 = icmp slt i32 %v81, 0
  %v83 = xor i1 %v82, true
  br i1 %v83, label %bb25, label %bb23

bb23:                                             ; preds = %bb22
  %v84 = load atomic i32, ptr %v79 syncscope("device") acquire, align 4
  br label %bb24

bb24:                                             ; preds = %bb23
  br label %bb22

bb25:                                             ; preds = %bb22
  br label %bb30

bb26:                                             ; preds = %bb12
  br label %bb27

bb27:                                             ; preds = %bb26, %bb9
  %v85 = add i32 %v40, 1
  br label %bb1

bb28:                                             ; preds = %bb1
  %v86 = bitcast ptr %v13 to ptr
  %v87 = atomicrmw add ptr %v86, i32 1 syncscope("device") monotonic, align 4
  br label %bb29

bb29:                                             ; preds = %bb28
  br label %bb32

bb30:                                             ; preds = %bb25, %bb20, %bb18
  %v88 = phi i32 [ -1, %bb18 ], [ %v67, %bb20 ], [ %v81, %bb25 ]
  br label %bb31

bb31:                                             ; preds = %bb30, %bb8
  %v89 = phi i32 [ %v56, %bb8 ], [ %v88, %bb30 ]
  br label %bb32

bb32:                                             ; preds = %bb31, %bb29
  %v90 = phi i32 [ -1, %bb29 ], [ %v89, %bb31 ]
  ret i32 %v90

bb33:                                             ; preds = %bb10
  %v91 = icmp eq i64 %v62, -1
  br i1 %v91, label %bb34, label %bb35

bb34:                                             ; preds = %bb33
  %v92 = insertvalue { i64, i64 } undef, i64 0, 0
  %v93 = insertvalue { i64, i64 } %v92, i64 %v62, 1
  %v94 = extractvalue { i64, i64 } %v93, 0
  %v95 = extractvalue { i64, i64 } %v93, 1
  br label %bb36

bb35:                                             ; preds = %bb33
  %v96 = insertvalue { i64, i64 } undef, i64 1, 0
  %v97 = insertvalue { i64, i64 } %v96, i64 %v62, 1
  %v98 = extractvalue { i64, i64 } %v97, 0
  %v99 = extractvalue { i64, i64 } %v97, 1
  br label %bb36

bb36:                                             ; preds = %bb35, %bb34
  %v100 = phi i64 [ %v94, %bb34 ], [ %v98, %bb35 ]
  %v101 = phi i64 [ %v95, %bb34 ], [ %v99, %bb35 ]
  %v102 = insertvalue { i64, i64 } undef, i64 %v100, 0
  %v103 = insertvalue { i64, i64 } %v102, i64 %v101, 1
  %v104 = extractvalue { i64, i64 } %v103, 0
  %v105 = bitcast i64 %v104 to i64
  %v106 = icmp eq i64 %v105, 0
  br i1 %v106, label %bb13, label %bb37

bb37:                                             ; preds = %bb36
  %v107 = icmp eq i64 %v105, 1
  br i1 %v107, label %bb12, label %bb11
}

; Function Attrs: alwaysinline nounwind
define internal float @__nv_sqrtf(float %x) #5 {
  %1 = call i32 @__nvvm_reflect(ptr @.str) #8
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %3, label %10

3:                                                ; preds = %0
  %4 = call i32 @__nvvm_reflect(ptr @.str.2) #8
  %5 = icmp ne i32 %4, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %3
  %7 = call float @llvm.nvvm.sqrt.rn.ftz.f(float %x) #8
  br label %__nvvm_sqrt_f.exit

8:                                                ; preds = %3
  %9 = call float @llvm.nvvm.sqrt.approx.ftz.f(float %x) #8
  br label %__nvvm_sqrt_f.exit

10:                                               ; preds = %0
  %11 = call i32 @__nvvm_reflect(ptr @.str.2) #8
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %15

13:                                               ; preds = %10
  %14 = call float @llvm.nvvm.sqrt.rn.f(float %x) #8
  br label %__nvvm_sqrt_f.exit

15:                                               ; preds = %10
  %16 = call float @llvm.nvvm.sqrt.approx.f(float %x) #8
  br label %__nvvm_sqrt_f.exit

__nvvm_sqrt_f.exit:                               ; preds = %6, %8, %13, %15
  %.0 = phi float [ %7, %6 ], [ %9, %8 ], [ %14, %13 ], [ %16, %15 ]
  ret float %.0
}

declare i32 @__nvvm_reflect(ptr) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #7

attributes #0 = { convergent }
attributes #1 = { alwaysinline convergent }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { alwaysinline nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #8 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
