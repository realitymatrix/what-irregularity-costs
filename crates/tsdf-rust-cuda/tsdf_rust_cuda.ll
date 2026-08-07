; ModuleID = 'builtin.module'
source_filename = "tsdf_rust_cuda"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

declare float @__nv_sqrtf(float)
declare float @__nv_ceilf(float)
declare i32 @llvm.fptosi.sat.i32.f32(float)
declare float @__nv_floorf(float)

define void @update_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, i32 %v14, i32 %v15, float %v16, float %v17, float %v18, float %v19, float %v20, float %v21, float %v22) #0 {
entry:
  %v23 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v1, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v3, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v5, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v7, 1
  %v31 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v32 = insertvalue { ptr, i64 } %v31, i64 %v9, 1
  %v33 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v34 = insertvalue { ptr, i64 } %v33, i64 %v11, 1
  %v35 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v36 = insertvalue { ptr, i64 } %v35, i64 %v13, 1
  br label %bb0
bb0:
  %v37 = phi { ptr, i64 } [ %v24, %entry ]
  %v38 = phi { ptr, i64 } [ %v26, %entry ]
  %v39 = phi { ptr, i64 } [ %v28, %entry ]
  %v40 = phi { ptr, i64 } [ %v30, %entry ]
  %v41 = phi { ptr, i64 } [ %v32, %entry ]
  %v42 = phi { ptr, i64 } [ %v34, %entry ]
  %v43 = phi { ptr, i64 } [ %v36, %entry ]
  %v44 = phi i32 [ %v14, %entry ]
  %v45 = phi i32 [ %v15, %entry ]
  %v46 = phi float [ %v16, %entry ]
  %v47 = phi float [ %v17, %entry ]
  %v48 = phi float [ %v18, %entry ]
  %v49 = phi float [ %v19, %entry ]
  %v50 = phi float [ %v20, %entry ]
  %v51 = phi float [ %v21, %entry ]
  %v52 = phi float [ %v22, %entry ]
  %v53 = alloca {  }, align 1
  %v54 = bitcast ptr %v53 to ptr
  %v55 = call i64 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshAQQRahQDk9_14tsdf_rust_cuda(ptr %v54) #0
  br label %bb1
bb1:
  %v56 = trunc i64 %v55 to i32
  %v57 = icmp sge i32 %v56, %v44
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb3, label %bb2
bb2:
  br label %bb31
bb3:
  %v59 = extractvalue { ptr, i64 } %v37, 0
  %v60 = mul i32 %v56, 3
  %v61 = sext i32 %v60 to i64
  %v62 = getelementptr inbounds float, ptr %v59, i64 %v61
  %v63 = load float, ptr %v62, align 4
  %v64 = getelementptr inbounds float, ptr %v62, i64 1
  %v65 = load float, ptr %v64, align 4
  %v66 = getelementptr inbounds float, ptr %v62, i64 2
  %v67 = load float, ptr %v66, align 4
  %v68 = fsub contract float %v63, %v49
  %v69 = fsub contract float %v65, %v50
  %v70 = fsub contract float %v67, %v51
  %v71 = fmul contract float %v68, %v68
  %v72 = fmul contract float %v69, %v69
  %v73 = fadd contract float %v71, %v72
  %v74 = fmul contract float %v70, %v70
  %v75 = fadd contract float %v73, %v74
  %v76 = fcmp ogt float %v52, 0.0
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb6, label %bb4
bb4:
  %v78 = fcmp ogt float %v75, %v52
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb6, label %bb5
bb5:
  br label %bb30
bb6:
  %v80 = call float @__nv_sqrtf(float %v75) #0
  br label %bb32
bb7:
  %v81 = fdiv contract float %v68, %v80
  %v82 = fdiv contract float %v69, %v80
  %v83 = fdiv contract float %v70, %v80
  %v84 = fdiv contract float 1.0, %v46
  %v85 = fdiv contract float 1.0, %v47
  %v86 = fmul contract float %v47, %v84
  %v87 = call float @__nv_ceilf(float %v86) #0
  br label %bb33
bb8:
  br label %bb30
bb9:
  %v88 = sub i32 0, %v149
  br label %bb10
bb10:
  %v89 = phi i32 [ %v88, %bb9 ], [ %v98, %bb12 ], [ %v145, %bb28 ]
  %v90 = icmp sle i32 %v89, %v149
  %v91 = xor i1 %v90, 1
  br i1 %v91, label %bb29, label %bb11
bb11:
  %v92 = sitofp i32 %v89 to float
  %v93 = fmul contract float %v92, %v46
  %v94 = fmul contract float %v81, %v93
  %v95 = fadd contract float %v63, %v94
  %v96 = fmul contract float %v95, %v84
  %v97 = call float @__nv_floorf(float %v96) #0
  br label %bb34
bb12:
  %v98 = add i32 %v89, 1
  br label %bb10
bb13:
  %v99 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v150, i32 8) #0
  br label %bb14
bb14:
  %v100 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v155, i32 8) #0
  br label %bb15
bb15:
  %v101 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v160, i32 8) #0
  br label %bb16
bb16:
  %v102 = extractvalue { ptr, i64 } %v38, 0
  %v103 = call i32 @tsdf_rust_cuda__kernels__find_block(ptr %v102, i32 %v45, i32 %v99, i32 %v100, i32 %v101) #0
  br label %bb17
bb17:
  %v104 = icmp sge i32 %v103, 0
  %v105 = xor i1 %v104, 1
  br i1 %v105, label %bb28, label %bb18
bb18:
  %v106 = mul i32 %v99, 8
  %v107 = sub i32 %v150, %v106
  %v108 = mul i32 %v100, 8
  %v109 = sub i32 %v155, %v108
  %v110 = mul i32 %v101, 8
  %v111 = sub i32 %v160, %v110
  %v112 = mul i32 %v103, 512
  %v113 = mul i32 %v111, 8
  %v114 = add i32 %v113, %v109
  %v115 = mul i32 %v114, 8
  %v116 = add i32 %v112, %v115
  %v117 = add i32 %v116, %v107
  %v118 = sext i32 %v117 to i64
  %v119 = extractvalue { ptr, i64 } %v40, 0
  %v120 = getelementptr inbounds float, ptr %v119, i64 %v118
  %v121 = bitcast ptr %v120 to ptr
  %v122 = load volatile float, ptr %v120, align 4
  br label %bb38
bb19:
  %v123 = fcmp oge float %v122, %v48
  %v124 = xor i1 %v123, 1
  br i1 %v124, label %bb20, label %bb27
bb20:
  %v125 = bitcast ptr %v120 to ptr
  %v126 = fmul contract float %v179, %v85
  %v127 = call float @core__f32___impl_f32___clamp(float %v126, float -1.0, float 1.0) #0
  br label %bb21
bb21:
  %v128 = atomicrmw fadd ptr %v125, float 1.0 syncscope("device") monotonic
  br label %bb22
bb22:
  %v129 = extractvalue { ptr, i64 } %v39, 0
  %v130 = getelementptr inbounds float, ptr %v129, i64 %v118
  %v131 = bitcast ptr %v130 to ptr
  %v132 = atomicrmw fadd ptr %v131, float %v127 syncscope("device") monotonic
  br label %bb23
bb23:
  %v133 = extractvalue { ptr, i64 } %v41, 0
  %v134 = getelementptr inbounds float, ptr %v133, i64 %v118
  %v135 = bitcast ptr %v134 to ptr
  %v136 = atomicrmw fadd ptr %v135, float 128.0 syncscope("device") monotonic
  br label %bb24
bb24:
  %v137 = extractvalue { ptr, i64 } %v42, 0
  %v138 = getelementptr inbounds float, ptr %v137, i64 %v118
  %v139 = bitcast ptr %v138 to ptr
  %v140 = atomicrmw fadd ptr %v139, float 128.0 syncscope("device") monotonic
  br label %bb25
bb25:
  %v141 = extractvalue { ptr, i64 } %v43, 0
  %v142 = getelementptr inbounds float, ptr %v141, i64 %v118
  %v143 = bitcast ptr %v142 to ptr
  %v144 = atomicrmw fadd ptr %v143, float 128.0 syncscope("device") monotonic
  br label %bb26
bb26:
  br label %bb27
bb27:
  br label %bb28
bb28:
  %v145 = add i32 %v89, 1
  br label %bb10
bb29:
  br label %bb31
bb30:
  br label %bb31
bb31:
  ret void
bb32:
  %v146 = fcmp ogt float %v80, 0.0000009999999974752427
  %v147 = xor i1 %v146, 1
  br i1 %v147, label %bb8, label %bb7
bb33:
  %v148 = call i32 @llvm.fptosi.sat.i32.f32(float %v87) #0
  %v149 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCshAQQRahQDk9_14tsdf_rust_cuda(i32 %v148, i32 1) #0
  br label %bb9
bb34:
  %v150 = call i32 @llvm.fptosi.sat.i32.f32(float %v97) #0
  %v151 = fmul contract float %v82, %v93
  %v152 = fadd contract float %v65, %v151
  %v153 = fmul contract float %v152, %v84
  %v154 = call float @__nv_floorf(float %v153) #0
  br label %bb35
bb35:
  %v155 = call i32 @llvm.fptosi.sat.i32.f32(float %v154) #0
  %v156 = fmul contract float %v83, %v93
  %v157 = fadd contract float %v67, %v156
  %v158 = fmul contract float %v157, %v84
  %v159 = call float @__nv_floorf(float %v158) #0
  br label %bb36
bb36:
  %v160 = call i32 @llvm.fptosi.sat.i32.f32(float %v159) #0
  %v161 = sitofp i32 %v150 to float
  %v162 = fadd contract float %v161, 0.5
  %v163 = fmul contract float %v162, %v46
  %v164 = sitofp i32 %v155 to float
  %v165 = fadd contract float %v164, 0.5
  %v166 = fmul contract float %v165, %v46
  %v167 = sitofp i32 %v160 to float
  %v168 = fadd contract float %v167, 0.5
  %v169 = fmul contract float %v168, %v46
  %v170 = fsub contract float %v163, %v49
  %v171 = fsub contract float %v166, %v50
  %v172 = fsub contract float %v169, %v51
  %v173 = fmul contract float %v170, %v170
  %v174 = fmul contract float %v171, %v171
  %v175 = fadd contract float %v173, %v174
  %v176 = fmul contract float %v172, %v172
  %v177 = fadd contract float %v175, %v176
  %v178 = call float @__nv_sqrtf(float %v177) #0
  br label %bb37
bb37:
  %v179 = fsub contract float %v80, %v178
  %v180 = fneg float %v47
  %v181 = fcmp olt float %v179, %v180
  %v182 = xor i1 %v181, 1
  br i1 %v182, label %bb13, label %bb12
bb38:
  %v183 = fcmp ogt float %v48, 0.0
  %v184 = xor i1 %v183, 1
  br i1 %v184, label %bb20, label %bb19
}

define void @alloc_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
bb0:
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
  %v43 = alloca {  }, align 1
  %v44 = bitcast ptr %v43 to ptr
  %v45 = call i64 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshAQQRahQDk9_14tsdf_rust_cuda(ptr %v44) #0
  br label %bb1
bb1:
  %v46 = trunc i64 %v45 to i32
  %v47 = icmp sge i32 %v46, %v34
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb3, label %bb2
bb2:
  br label %bb21
bb3:
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
  %v66 = fcmp ogt float %v42, 0.0
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb6, label %bb4
bb4:
  %v68 = fcmp ogt float %v65, %v42
  %v69 = xor i1 %v68, 1
  br i1 %v69, label %bb6, label %bb5
bb5:
  br label %bb20
bb6:
  %v70 = call float @__nv_sqrtf(float %v65) #0
  br label %bb22
bb7:
  %v71 = fdiv contract float %v58, %v70
  %v72 = fdiv contract float %v59, %v70
  %v73 = fdiv contract float %v60, %v70
  %v74 = fdiv contract float 1.0, %v37
  %v75 = fmul contract float %v38, %v74
  %v76 = call float @__nv_ceilf(float %v75) #0
  br label %bb23
bb8:
  br label %bb20
bb9:
  %v77 = sub i32 0, %v99
  br label %bb10
bb10:
  %v78 = phi i32 [ %v77, %bb9 ], [ %v95, %bb18 ]
  %v79 = icmp sle i32 %v78, %v99
  %v80 = xor i1 %v79, 1
  br i1 %v80, label %bb19, label %bb11
bb11:
  %v81 = sitofp i32 %v78 to float
  %v82 = fmul contract float %v81, %v37
  %v83 = fmul contract float %v71, %v82
  %v84 = fadd contract float %v53, %v83
  %v85 = fmul contract float %v84, %v74
  %v86 = call float @__nv_floorf(float %v85) #0
  br label %bb24
bb12:
  %v87 = extractvalue { ptr, i64 } %v30, 0
  %v88 = extractvalue { ptr, i64 } %v31, 0
  %v89 = extractvalue { ptr, i64 } %v33, 0
  %v90 = extractvalue { ptr, i64 } %v32, 0
  %v91 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v100, i32 8) #0
  br label %bb13
bb13:
  %v92 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v105, i32 8) #0
  br label %bb14
bb14:
  %v93 = call i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v110, i32 8) #0
  br label %bb15
bb15:
  %v94 = call i32 @tsdf_rust_cuda__kernels__find_or_insert(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
  br label %bb16
bb16:
  br label %bb18
bb17:
  br label %bb18
bb18:
  %v95 = add i32 %v78, 1
  br label %bb10
bb19:
  br label %bb21
bb20:
  br label %bb21
bb21:
  ret void
bb22:
  %v96 = fcmp ogt float %v70, 0.0000009999999974752427
  %v97 = xor i1 %v96, 1
  br i1 %v97, label %bb8, label %bb7
bb23:
  %v98 = call i32 @llvm.fptosi.sat.i32.f32(float %v76) #0
  %v99 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCshAQQRahQDk9_14tsdf_rust_cuda(i32 %v98, i32 1) #0
  br label %bb9
bb24:
  %v100 = call i32 @llvm.fptosi.sat.i32.f32(float %v86) #0
  %v101 = fmul contract float %v72, %v82
  %v102 = fadd contract float %v55, %v101
  %v103 = fmul contract float %v102, %v74
  %v104 = call float @__nv_floorf(float %v103) #0
  br label %bb25
bb25:
  %v105 = call i32 @llvm.fptosi.sat.i32.f32(float %v104) #0
  %v106 = fmul contract float %v73, %v82
  %v107 = fadd contract float %v57, %v106
  %v108 = fmul contract float %v107, %v74
  %v109 = call float @__nv_floorf(float %v108) #0
  br label %bb26
bb26:
  %v110 = call i32 @llvm.fptosi.sat.i32.f32(float %v109) #0
  %v111 = sitofp i32 %v100 to float
  %v112 = fadd contract float %v111, 0.5
  %v113 = fmul contract float %v112, %v37
  %v114 = sitofp i32 %v105 to float
  %v115 = fadd contract float %v114, 0.5
  %v116 = fmul contract float %v115, %v37
  %v117 = sitofp i32 %v110 to float
  %v118 = fadd contract float %v117, 0.5
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
bb27:
  %v129 = fsub contract float %v70, %v128
  %v130 = fneg float %v38
  %v131 = fcmp oge float %v129, %v130
  %v132 = xor i1 %v131, 1
  br i1 %v132, label %bb17, label %bb12
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define i64 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshAQQRahQDk9_14tsdf_rust_cuda(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v5 = zext i32 %v2 to i64
  %v6 = zext i32 %v3 to i64
  %v7 = zext i32 %v4 to i64
  %v8 = icmp eq i64 %v6, 0
  br i1 %v8, label %bb10, label %bb8
bb4:
  %v9 = xor i1 %v20, 1
  br i1 %v9, label %bb6, label %bb5
bb5:
  %v10 = icmp ne i64 %v19, 18446744073709551615
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v11 = phi i1 [ %v10, %bb5 ], [ 0, %bb6 ]
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb14, label %bb13
bb8:
  %v13 = sub i64 18446744073709551615, %v7
  %v14 = udiv i64 %v13, %v6
  %v15 = icmp ugt i64 %v5, %v14
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb11, label %bb9
bb9:
  br label %bb10
bb10:
  br label %bb12
bb11:
  %v17 = mul i64 %v5, %v6
  %v18 = add i64 %v17, %v7
  br label %bb12
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshAQQRahQDk9_14tsdf_rust_cuda(ptr %v1) #0
  br label %bb4
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v21, label %bb14, label %bb15
bb14:
  br label %bb15
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22
}

declare void @llvm.trap()

define i32 @tsdf_rust_cuda__kernels__floor_div(i32 %v0, i32 %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = icmp eq i32 %v3, 0
  %v5 = xor i1 %v4, 1
  br i1 %v5, label %bb1, label %bb9
bb1:
  %v6 = icmp eq i32 %v3, 4294967295
  %v7 = icmp eq i32 %v2, 2147483648
  %v8 = and i1 %v6, %v7
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb2, label %bb10
bb2:
  %v10 = sdiv i32 %v2, %v3
  %v11 = srem i32 %v2, %v3
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %bb6, label %bb3
bb3:
  %v13 = icmp slt i32 %v2, 0
  %v14 = icmp slt i32 %v3, 0
  %v15 = icmp ne i1 %v13, %v14
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb5, label %bb4
bb4:
  %v17 = sub i32 %v10, 1
  br label %bb8
bb5:
  br label %bb7
bb6:
  br label %bb7
bb7:
  br label %bb8
bb8:
  %v18 = phi i32 [ %v17, %bb4 ], [ %v10, %bb7 ]
  ret i32 %v18
bb9:
  call void @llvm.trap() #0
  unreachable
bb10:
  call void @llvm.trap() #0
  unreachable
}

define i32 @tsdf_rust_cuda__kernels__find_block(ptr %v0, i32 %v1, i32 %v2, i32 %v3, i32 %v4) alwaysinline #0 {
entry:
  br label %bb0
bb0:
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
bb1:
  %v32 = phi i32 [ 0, %bb0 ], [ %v49, %bb6 ]
  %v33 = icmp ult i32 %v32, %v22
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb11, label %bb2
bb2:
  %v35 = add i32 %v31, %v32
  %v36 = and i32 %v35, %v6
  %v37 = zext i32 %v36 to i64
  %v38 = mul i64 %v37, 2
  %v39 = getelementptr inbounds i64, ptr %v5, i64 %v38
  %v40 = getelementptr inbounds i64, ptr %v39, i64 1
  %v41 = bitcast ptr %v40 to ptr
  %v42 = bitcast ptr %v39 to ptr
  %v43 = load i64, ptr %v42, align 8
  %v44 = icmp eq i64 %v43, 18446744073709551615
  br i1 %v44, label %bb3, label %bb4
bb3:
  br label %bb12
bb4:
  %v45 = icmp eq i64 %v43, %v21
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb6, label %bb5
bb5:
  %v47 = bitcast ptr %v40 to ptr
  %v48 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v47)
  br label %bb7
bb6:
  %v49 = add i32 %v32, 1
  br label %bb1
bb7:
  %v50 = phi i32 [ %v48, %bb5 ], [ %v53, %bb9 ]
  %v51 = icmp slt i32 %v50, 0
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb10, label %bb8
bb8:
  %v53 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v47)
  br label %bb9
bb9:
  br label %bb7
bb10:
  br label %bb12
bb11:
  br label %bb13
bb12:
  %v54 = phi i32 [ 4294967295, %bb3 ], [ %v50, %bb10 ]
  br label %bb13
bb13:
  %v55 = phi i32 [ 4294967295, %bb11 ], [ %v54, %bb12 ]
  ret i32 %v55
}

define float @core__f32___impl_f32___clamp(float %v0, float %v1, float %v2) #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi float [ %v0, %entry ]
  %v4 = phi float [ %v1, %entry ]
  %v5 = phi float [ %v2, %entry ]
  %v6 = fcmp ole float %v4, %v5
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb2, label %bb1
bb1:
  %v8 = fcmp olt float %v3, %v4
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb4, label %bb3
bb2:
  call void asm sideeffect "trap;", ""()
  unreachable
bb3:
  br label %bb5
bb4:
  br label %bb5
bb5:
  %v11 = phi float [ %v4, %bb3 ], [ %v3, %bb4 ]
  %v12 = fcmp ogt float %v11, %v5
  %v13 = xor i1 %v12, 1
  br i1 %v13, label %bb7, label %bb6
bb6:
  br label %bb8
bb7:
  br label %bb8
bb8:
  %v14 = phi float [ %v5, %bb6 ], [ %v11, %bb7 ]
  ret float %v14
}

define i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCshAQQRahQDk9_14tsdf_rust_cuda(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
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
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v10 = load i32, ptr %v4, align 4
  br label %bb4
bb3:
  %v11 = load i32, ptr %v5, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

declare void @llvm.nvvm.membar.gl()

define i32 @tsdf_rust_cuda__kernels__find_or_insert(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
entry:
  br label %bb0
bb0:
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
bb1:
  %v40 = phi i32 [ 0, %bb0 ], [ %v89, %bb29 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb30, label %bb2
bb2:
  %v43 = add i32 %v39, %v40
  %v44 = and i32 %v43, %v10
  %v45 = zext i32 %v44 to i64
  %v46 = mul i64 %v45, 2
  %v47 = getelementptr inbounds i64, ptr %v9, i64 %v46
  %v48 = getelementptr inbounds i64, ptr %v47, i64 1
  %v49 = bitcast ptr %v48 to ptr
  %v50 = bitcast ptr %v47 to ptr
  %v51 = call i64 asm sideeffect "ld.relaxed.gpu.b64 $0, [$1];", "=l,l,~{memory}"(ptr %v50)
  br label %bb3
bb3:
  %v52 = icmp eq i64 %v51, %v29
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb9, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  %v56 = phi i32 [ %v55, %bb4 ], [ %v59, %bb7 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb8, label %bb6
bb6:
  %v59 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb7
bb7:
  br label %bb5
bb8:
  br label %bb33
bb9:
  %v60 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v60, label %bb10, label %bb29
bb10:
  %v61 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v62 = extractvalue { i64, i1 } %v61, 0
  br label %bb35
bb11:
  unreachable
bb12:
  %v63 = extractvalue { i64, i64 } %v107, 1
  %v64 = icmp eq i64 %v63, %v29
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb28, label %bb23
bb13:
  %v66 = bitcast ptr %v11 to ptr
  %v67 = atomicrmw add ptr %v66, i32 1 syncscope("device") monotonic
  br label %bb14
bb14:
  %v68 = icmp sge i32 %v67, %v12
  %v69 = xor i1 %v68, 1
  br i1 %v69, label %bb20, label %bb15
bb15:
  %v70 = atomicrmw sub ptr %v66, i32 1 syncscope("device") monotonic
  br label %bb16
bb16:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb17
bb17:
  %v72 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb18
bb18:
  %v73 = bitcast ptr %v13 to ptr
  %v74 = atomicrmw add ptr %v73, i64 1 syncscope("device") monotonic
  br label %bb19
bb19:
  br label %bb32
bb20:
  %v75 = mul i32 %v67, 3
  %v76 = sext i32 %v75 to i64
  %v77 = getelementptr inbounds i32, ptr %v14, i64 %v76
  store i32 %v15, ptr %v77, align 4
  %v78 = getelementptr inbounds i32, ptr %v77, i64 1
  store i32 %v16, ptr %v78, align 4
  %v79 = getelementptr inbounds i32, ptr %v77, i64 2
  store i32 %v17, ptr %v79, align 4
  call void @llvm.nvvm.membar.gl() #0
  br label %bb21
bb21:
  %v81 = bitcast ptr %v48 to ptr
  %v82 = atomicrmw xchg ptr %v81, i32 %v67 syncscope("device") monotonic
  br label %bb22
bb22:
  br label %bb32
bb23:
  %v83 = bitcast ptr %v48 to ptr
  %v84 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v83)
  br label %bb24
bb24:
  %v85 = phi i32 [ %v84, %bb23 ], [ %v88, %bb26 ]
  %v86 = icmp slt i32 %v85, 0
  %v87 = xor i1 %v86, 1
  br i1 %v87, label %bb27, label %bb25
bb25:
  %v88 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v83)
  br label %bb26
bb26:
  br label %bb24
bb27:
  br label %bb32
bb28:
  br label %bb29
bb29:
  %v89 = add i32 %v40, 1
  br label %bb1
bb30:
  %v90 = bitcast ptr %v13 to ptr
  %v91 = atomicrmw add ptr %v90, i64 1 syncscope("device") monotonic
  br label %bb31
bb31:
  br label %bb34
bb32:
  %v92 = phi i32 [ 4294967295, %bb19 ], [ %v67, %bb22 ], [ %v85, %bb27 ]
  br label %bb33
bb33:
  %v93 = phi i32 [ %v56, %bb8 ], [ %v92, %bb32 ]
  br label %bb34
bb34:
  %v94 = phi i32 [ 4294967295, %bb31 ], [ %v93, %bb33 ]
  ret i32 %v94
bb35:
  %v95 = icmp eq i64 %v62, 18446744073709551615
  br i1 %v95, label %bb36, label %bb37
bb36:
  %v96 = insertvalue { i64, i64 } undef, i64 0, 0
  %v97 = insertvalue { i64, i64 } %v96, i64 %v62, 1
  %v98 = extractvalue { i64, i64 } %v97, 0
  %v99 = extractvalue { i64, i64 } %v97, 1
  br label %bb38
bb37:
  %v100 = insertvalue { i64, i64 } undef, i64 1, 0
  %v101 = insertvalue { i64, i64 } %v100, i64 %v62, 1
  %v102 = extractvalue { i64, i64 } %v101, 0
  %v103 = extractvalue { i64, i64 } %v101, 1
  br label %bb38
bb38:
  %v104 = phi i64 [ %v98, %bb36 ], [ %v102, %bb37 ]
  %v105 = phi i64 [ %v99, %bb36 ], [ %v103, %bb37 ]
  %v106 = insertvalue { i64, i64 } undef, i64 %v104, 0
  %v107 = insertvalue { i64, i64 } %v106, i64 %v105, 1
  %v108 = extractvalue { i64, i64 } %v107, 0
  %v109 = bitcast i64 %v108 to i64
  %v110 = icmp eq i64 %v109, 0
  br i1 %v110, label %bb13, label %bb39
bb39:
  %v111 = icmp eq i64 %v109, 1
  br i1 %v111, label %bb12, label %bb11
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCscaYDm7YuRWO_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECshAQQRahQDk9_14tsdf_rust_cuda(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, 1
  br i1 %v3, label %bb2, label %bb1
bb1:
  br label %bb8
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3
bb3:
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6
bb5:
  br label %bb7
bb6:
  %v7 = icmp eq i32 %v6, 1
  br label %bb7
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1
  br i1 %v10, label %bb9, label %bb10
bb9:
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12
bb12:
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15
bb14:
  br label %bb16
bb15:
  %v16 = icmp eq i32 %v15, 1
  br label %bb16
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1
  br i1 %v19, label %bb19, label %bb18
bb18:
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i32, ptr %v2, align 4
  %v5 = load i32, ptr %v3, align 4
  %v6 = icmp slt i32 %v4, %v5
  ret i1 %v6
}


@llvm.used = appending global [2 x ptr] [ptr @alloc_kernel, ptr @update_kernel], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{ptr @update_kernel, !"kernel", i32 1}
!1 = !{ptr @alloc_kernel, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1}

!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 2}
