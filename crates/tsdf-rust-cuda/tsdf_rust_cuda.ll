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

define void @alloc_kernel_nofence(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_KBX_Kb0_KBX_KB19_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define void @alloc_kernel_nocas(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb0_Kb1_KB11_KB11_KB11_KBX_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define void @alloc_kernel_nopublish(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_Kb0_KBX_KB11_KB11_KB11_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define void @alloc_kernel_slotidx(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @tsdf_rust_cuda__kernels__find_or_insert_slotidx(ptr %v87, i32 %v35, ptr %v88, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_KBX_KBX_KBX_Kb0_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define void @alloc_kernel_cas128(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @tsdf_rust_cuda__kernels__find_or_insert_128(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define void @alloc_kernel_nocount(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_Kb0_KBX_KBX_KB15_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define void @cas128_selftest(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v11 = icmp eq i32 %v10, 0
  br i1 %v11, label %bb3, label %bb2
bb2:
  br label %bb5
bb3:
  %v12 = extractvalue { ptr, i64 } %v8, 0
  %v13 = extractvalue { ptr, i64 } %v8, 0
  %v14 = ptrtoint ptr %v13 to i64
  %v15 = call { i64, i64 } asm sideeffect "{ .reg .b128 t, e, d;\0A\09.reg .u64 g;\0A\09cvta.to.global.u64 g, $6;\0A\09mov.b128 e, {$2, $3};\0A\09mov.b128 d, {$4, $5};\0A\09atom.global.acq_rel.gpu.cas.b128 t, [g], e, d;\0A\09mov.b128 {$0, $1}, t; }", "=l,=l,l,l,l,l,l,~{memory}"(i64 18446744073709551615, i64 4294967295, i64 42, i64 7, i64 %v14) #0
  %v16 = extractvalue { i64, i64 } %v15, 0
  %v17 = extractvalue { i64, i64 } %v15, 1
  %v18 = insertvalue { i64, i64 } undef, i64 %v16, 0
  %v19 = insertvalue { i64, i64 } %v18, i64 %v17, 1
  br label %bb4
bb4:
  %v20 = extractvalue { i64, i64 } %v19, 0
  %v21 = extractvalue { i64, i64 } %v19, 1
  %v22 = extractvalue { ptr, i64 } %v9, 0
  %v23 = bitcast i64 %v20 to i64
  store i64 %v23, ptr %v22, align 8
  %v24 = getelementptr inbounds i64, ptr %v22, i64 1
  %v25 = bitcast i64 %v21 to i64
  store i64 %v25, ptr %v24, align 8
  %v26 = getelementptr inbounds i64, ptr %v22, i64 2
  %v27 = load i64, ptr %v12, align 8
  store i64 %v27, ptr %v26, align 8
  %v28 = getelementptr inbounds i64, ptr %v22, i64 3
  %v29 = getelementptr inbounds i64, ptr %v13, i64 1
  %v30 = bitcast ptr %v29 to ptr
  %v31 = load i64, ptr %v30, align 8
  store i64 %v31, ptr %v28, align 8
  br label %bb5
bb5:
  ret void
}

define void @alloc_kernel_countcas(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_KBX_KBX_KBX_KBX_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define void @alloc_kernel_nospin(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, float %v13, float %v14, float %v15, float %v16, float %v17, float %v18) #0 {
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
  %v94 = call i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_Kb0_KBX_KBX_KBX_KB11_EB4_(ptr %v87, i32 %v36, ptr %v88, i32 %v35, ptr %v89, ptr %v90, i32 %v91, i32 %v92, i32 %v93) #0
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

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_KBX_Kb0_KBX_KB19_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v88, %bb40 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb41, label %bb2
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
  br i1 %v53, label %bb11, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  %v56 = phi i32 [ %v55, %bb5 ], [ %v59, %bb8 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb9, label %bb7
bb7:
  %v59 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb8
bb8:
  br label %bb6
bb9:
  br label %bb10
bb10:
  br label %bb46
bb11:
  %v60 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v60, label %bb12, label %bb40
bb12:
  br label %bb13
bb13:
  br label %bb14
bb14:
  %v61 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v62 = extractvalue { i64, i1 } %v61, 0
  br label %bb48
bb15:
  unreachable
bb16:
  %v63 = extractvalue { i64, i64 } %v107, 1
  %v64 = icmp eq i64 %v63, %v29
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb39, label %bb32
bb17:
  br label %bb18
bb18:
  %v66 = bitcast ptr %v11 to ptr
  %v67 = atomicrmw add ptr %v66, i32 1 syncscope("device") monotonic
  br label %bb19
bb19:
  %v68 = icmp sge i32 %v67, %v12
  %v69 = xor i1 %v68, 1
  br i1 %v69, label %bb25, label %bb20
bb20:
  %v70 = bitcast ptr %v11 to ptr
  %v71 = atomicrmw sub ptr %v70, i32 1 syncscope("device") monotonic
  br label %bb21
bb21:
  br label %bb22
bb22:
  %v72 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb23
bb23:
  %v73 = bitcast ptr %v13 to ptr
  %v74 = atomicrmw add ptr %v73, i64 1 syncscope("device") monotonic
  br label %bb24
bb24:
  br label %bb43
bb25:
  br label %bb26
bb26:
  %v75 = mul i32 %v67, 3
  %v76 = sext i32 %v75 to i64
  %v77 = getelementptr inbounds i32, ptr %v14, i64 %v76
  store i32 %v15, ptr %v77, align 4
  %v78 = getelementptr inbounds i32, ptr %v77, i64 1
  store i32 %v16, ptr %v78, align 4
  %v79 = getelementptr inbounds i32, ptr %v77, i64 2
  store i32 %v17, ptr %v79, align 4
  br label %bb27
bb27:
  br label %bb28
bb28:
  br label %bb29
bb29:
  %v80 = bitcast ptr %v48 to ptr
  %v81 = atomicrmw xchg ptr %v80, i32 %v67 syncscope("device") monotonic
  br label %bb30
bb30:
  br label %bb31
bb31:
  br label %bb43
bb32:
  %v82 = bitcast ptr %v48 to ptr
  %v83 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v82)
  br label %bb33
bb33:
  br label %bb34
bb34:
  %v84 = phi i32 [ %v83, %bb33 ], [ %v87, %bb36 ]
  %v85 = icmp slt i32 %v84, 0
  %v86 = xor i1 %v85, 1
  br i1 %v86, label %bb37, label %bb35
bb35:
  %v87 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v82)
  br label %bb36
bb36:
  br label %bb34
bb37:
  br label %bb38
bb38:
  br label %bb44
bb39:
  br label %bb40
bb40:
  %v88 = add i32 %v40, 1
  br label %bb1
bb41:
  %v89 = bitcast ptr %v13 to ptr
  %v90 = atomicrmw add ptr %v89, i64 1 syncscope("device") monotonic
  br label %bb42
bb42:
  br label %bb47
bb43:
  %v91 = phi i32 [ 4294967295, %bb24 ], [ %v67, %bb31 ]
  br label %bb44
bb44:
  %v92 = phi i32 [ %v84, %bb38 ], [ %v91, %bb43 ]
  br label %bb45
bb45:
  br label %bb46
bb46:
  %v93 = phi i32 [ %v56, %bb10 ], [ %v92, %bb45 ]
  br label %bb47
bb47:
  %v94 = phi i32 [ 4294967295, %bb42 ], [ %v93, %bb46 ]
  ret i32 %v94
bb48:
  %v95 = icmp eq i64 %v62, 18446744073709551615
  br i1 %v95, label %bb49, label %bb50
bb49:
  %v96 = insertvalue { i64, i64 } undef, i64 0, 0
  %v97 = insertvalue { i64, i64 } %v96, i64 %v62, 1
  %v98 = extractvalue { i64, i64 } %v97, 0
  %v99 = extractvalue { i64, i64 } %v97, 1
  br label %bb51
bb50:
  %v100 = insertvalue { i64, i64 } undef, i64 1, 0
  %v101 = insertvalue { i64, i64 } %v100, i64 %v62, 1
  %v102 = extractvalue { i64, i64 } %v101, 0
  %v103 = extractvalue { i64, i64 } %v101, 1
  br label %bb51
bb51:
  %v104 = phi i64 [ %v98, %bb49 ], [ %v102, %bb50 ]
  %v105 = phi i64 [ %v99, %bb49 ], [ %v103, %bb50 ]
  %v106 = insertvalue { i64, i64 } undef, i64 %v104, 0
  %v107 = insertvalue { i64, i64 } %v106, i64 %v105, 1
  %v108 = extractvalue { i64, i64 } %v107, 0
  %v109 = bitcast i64 %v108 to i64
  %v110 = icmp eq i64 %v109, 0
  br i1 %v110, label %bb17, label %bb52
bb52:
  %v111 = icmp eq i64 %v109, 1
  br i1 %v111, label %bb16, label %bb15
}

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb0_Kb1_KB11_KB11_KB11_KBX_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v61, %bb14 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb15, label %bb2
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
  br i1 %v53, label %bb11, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  %v56 = phi i32 [ %v55, %bb5 ], [ %v59, %bb8 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb9, label %bb7
bb7:
  %v59 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb8
bb8:
  br label %bb6
bb9:
  br label %bb10
bb10:
  br label %bb18
bb11:
  %v60 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v60, label %bb12, label %bb14
bb12:
  br label %bb13
bb13:
  br label %bb17
bb14:
  %v61 = add i32 %v40, 1
  br label %bb1
bb15:
  %v62 = bitcast ptr %v13 to ptr
  %v63 = atomicrmw add ptr %v62, i64 1 syncscope("device") monotonic
  br label %bb16
bb16:
  br label %bb19
bb17:
  br label %bb18
bb18:
  %v64 = phi i32 [ %v56, %bb10 ], [ 4294967295, %bb17 ]
  br label %bb19
bb19:
  %v65 = phi i32 [ 4294967295, %bb16 ], [ %v64, %bb18 ]
  ret i32 %v65
}

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_Kb0_KBX_KB11_KB11_KB11_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v73, %bb28 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb29, label %bb2
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
  br i1 %v53, label %bb7, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  br label %bb34
bb7:
  %v56 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v56, label %bb8, label %bb28
bb8:
  br label %bb9
bb9:
  br label %bb10
bb10:
  %v57 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v58 = extractvalue { i64, i1 } %v57, 0
  br label %bb36
bb11:
  unreachable
bb12:
  %v59 = extractvalue { i64, i64 } %v92, 1
  %v60 = icmp eq i64 %v59, %v29
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb27, label %bb24
bb13:
  br label %bb14
bb14:
  %v62 = bitcast ptr %v11 to ptr
  %v63 = atomicrmw add ptr %v62, i32 1 syncscope("device") monotonic
  br label %bb15
bb15:
  %v64 = icmp sge i32 %v63, %v12
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb21, label %bb16
bb16:
  %v66 = bitcast ptr %v11 to ptr
  %v67 = atomicrmw sub ptr %v66, i32 1 syncscope("device") monotonic
  br label %bb17
bb17:
  br label %bb18
bb18:
  %v68 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb19
bb19:
  %v69 = bitcast ptr %v13 to ptr
  %v70 = atomicrmw add ptr %v69, i64 1 syncscope("device") monotonic
  br label %bb20
bb20:
  br label %bb31
bb21:
  br label %bb22
bb22:
  br label %bb23
bb23:
  br label %bb31
bb24:
  %v71 = bitcast ptr %v48 to ptr
  %v72 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v71)
  br label %bb25
bb25:
  br label %bb26
bb26:
  br label %bb32
bb27:
  br label %bb28
bb28:
  %v73 = add i32 %v40, 1
  br label %bb1
bb29:
  %v74 = bitcast ptr %v13 to ptr
  %v75 = atomicrmw add ptr %v74, i64 1 syncscope("device") monotonic
  br label %bb30
bb30:
  br label %bb35
bb31:
  %v76 = phi i32 [ 4294967295, %bb20 ], [ %v63, %bb23 ]
  br label %bb32
bb32:
  %v77 = phi i32 [ %v72, %bb26 ], [ %v76, %bb31 ]
  br label %bb33
bb33:
  br label %bb34
bb34:
  %v78 = phi i32 [ %v55, %bb6 ], [ %v77, %bb33 ]
  br label %bb35
bb35:
  %v79 = phi i32 [ 4294967295, %bb30 ], [ %v78, %bb34 ]
  ret i32 %v79
bb36:
  %v80 = icmp eq i64 %v58, 18446744073709551615
  br i1 %v80, label %bb37, label %bb38
bb37:
  %v81 = insertvalue { i64, i64 } undef, i64 0, 0
  %v82 = insertvalue { i64, i64 } %v81, i64 %v58, 1
  %v83 = extractvalue { i64, i64 } %v82, 0
  %v84 = extractvalue { i64, i64 } %v82, 1
  br label %bb39
bb38:
  %v85 = insertvalue { i64, i64 } undef, i64 1, 0
  %v86 = insertvalue { i64, i64 } %v85, i64 %v58, 1
  %v87 = extractvalue { i64, i64 } %v86, 0
  %v88 = extractvalue { i64, i64 } %v86, 1
  br label %bb39
bb39:
  %v89 = phi i64 [ %v83, %bb37 ], [ %v87, %bb38 ]
  %v90 = phi i64 [ %v84, %bb37 ], [ %v88, %bb38 ]
  %v91 = insertvalue { i64, i64 } undef, i64 %v89, 0
  %v92 = insertvalue { i64, i64 } %v91, i64 %v90, 1
  %v93 = extractvalue { i64, i64 } %v92, 0
  %v94 = bitcast i64 %v93 to i64
  %v95 = icmp eq i64 %v94, 0
  br i1 %v95, label %bb13, label %bb40
bb40:
  %v96 = icmp eq i64 %v94, 1
  br i1 %v96, label %bb12, label %bb11
}

define i32 @tsdf_rust_cuda__kernels__find_or_insert_slotidx(ptr %v0, i32 %v1, ptr %v2, ptr %v3, ptr %v4, i32 %v5, i32 %v6, i32 %v7) #0 {
entry:
  br label %bb0
bb0:
  %v8 = phi ptr [ %v0, %entry ]
  %v9 = phi i32 [ %v1, %entry ]
  %v10 = phi ptr [ %v2, %entry ]
  %v11 = phi ptr [ %v3, %entry ]
  %v12 = phi ptr [ %v4, %entry ]
  %v13 = phi i32 [ %v5, %entry ]
  %v14 = phi i32 [ %v6, %entry ]
  %v15 = phi i32 [ %v7, %entry ]
  %v16 = bitcast i32 %v9 to i32
  %v17 = sub i32 %v16, 1
  %v18 = sext i32 %v13 to i64
  %v19 = add i64 %v18, 1048576
  %v20 = and i64 42, 63
  %v21 = shl i64 %v19, %v20
  %v22 = sext i32 %v14 to i64
  %v23 = add i64 %v22, 1048576
  %v24 = and i64 21, 63
  %v25 = shl i64 %v23, %v24
  %v26 = or i64 %v21, %v25
  %v27 = sext i32 %v15 to i64
  %v28 = add i64 %v27, 1048576
  %v29 = or i64 %v26, %v28
  %v30 = bitcast i32 %v13 to i32
  %v31 = mul i32 %v30, 73856093
  %v32 = bitcast i32 %v14 to i32
  %v33 = mul i32 %v32, 19349663
  %v34 = xor i32 %v31, %v33
  %v35 = bitcast i32 %v15 to i32
  %v36 = mul i32 %v35, 83492791
  %v37 = xor i32 %v34, %v36
  %v38 = and i32 %v37, %v17
  br label %bb1
bb1:
  %v39 = phi i32 [ 0, %bb0 ], [ %v67, %bb13 ]
  %v40 = icmp ule i32 %v39, %v17
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb14, label %bb2
bb2:
  %v42 = add i32 %v38, %v39
  %v43 = and i32 %v42, %v17
  %v44 = zext i32 %v43 to i64
  %v45 = mul i64 %v44, 2
  %v46 = getelementptr inbounds i64, ptr %v8, i64 %v45
  %v47 = bitcast ptr %v46 to ptr
  %v48 = call i64 asm sideeffect "ld.relaxed.gpu.b64 $0, [$1];", "=l,l,~{memory}"(ptr %v47)
  br label %bb3
bb3:
  %v49 = icmp eq i64 %v48, %v29
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb5, label %bb4
bb4:
  %v51 = trunc i64 %v44 to i32
  br label %bb17
bb5:
  %v52 = icmp eq i64 %v48, 18446744073709551615
  br i1 %v52, label %bb6, label %bb13
bb6:
  %v53 = cmpxchg ptr %v47, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v54 = extractvalue { i64, i1 } %v53, 0
  br label %bb19
bb7:
  unreachable
bb8:
  %v55 = extractvalue { i64, i64 } %v85, 1
  %v56 = icmp eq i64 %v55, %v29
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb12, label %bb11
bb9:
  %v58 = mul i64 %v44, 3
  %v59 = getelementptr inbounds i32, ptr %v12, i64 %v58
  store i32 %v13, ptr %v59, align 4
  %v60 = getelementptr inbounds i32, ptr %v59, i64 1
  store i32 %v14, ptr %v60, align 4
  %v61 = getelementptr inbounds i32, ptr %v59, i64 2
  store i32 %v15, ptr %v61, align 4
  %v62 = bitcast ptr %v10 to ptr
  %v63 = atomicrmw add ptr %v62, i32 1 syncscope("device") monotonic
  br label %bb10
bb10:
  %v64 = trunc i64 %v44 to i32
  br label %bb16
bb11:
  %v65 = extractvalue { i64, i64 } %v85, 1
  %v66 = trunc i64 %v44 to i32
  br label %bb16
bb12:
  br label %bb13
bb13:
  %v67 = add i32 %v39, 1
  br label %bb1
bb14:
  %v68 = bitcast ptr %v11 to ptr
  %v69 = atomicrmw add ptr %v68, i64 1 syncscope("device") monotonic
  br label %bb15
bb15:
  br label %bb18
bb16:
  %v70 = phi i32 [ %v64, %bb10 ], [ %v66, %bb11 ]
  br label %bb17
bb17:
  %v71 = phi i32 [ %v51, %bb4 ], [ %v70, %bb16 ]
  br label %bb18
bb18:
  %v72 = phi i32 [ 4294967295, %bb15 ], [ %v71, %bb17 ]
  ret i32 %v72
bb19:
  %v73 = icmp eq i64 %v54, 18446744073709551615
  br i1 %v73, label %bb20, label %bb21
bb20:
  %v74 = insertvalue { i64, i64 } undef, i64 0, 0
  %v75 = insertvalue { i64, i64 } %v74, i64 %v54, 1
  %v76 = extractvalue { i64, i64 } %v75, 0
  %v77 = extractvalue { i64, i64 } %v75, 1
  br label %bb22
bb21:
  %v78 = insertvalue { i64, i64 } undef, i64 1, 0
  %v79 = insertvalue { i64, i64 } %v78, i64 %v54, 1
  %v80 = extractvalue { i64, i64 } %v79, 0
  %v81 = extractvalue { i64, i64 } %v79, 1
  br label %bb22
bb22:
  %v82 = phi i64 [ %v76, %bb20 ], [ %v80, %bb21 ]
  %v83 = phi i64 [ %v77, %bb20 ], [ %v81, %bb21 ]
  %v84 = insertvalue { i64, i64 } undef, i64 %v82, 0
  %v85 = insertvalue { i64, i64 } %v84, i64 %v83, 1
  %v86 = extractvalue { i64, i64 } %v85, 0
  %v87 = bitcast i64 %v86 to i64
  %v88 = icmp eq i64 %v87, 0
  br i1 %v88, label %bb9, label %bb23
bb23:
  %v89 = icmp eq i64 %v87, 1
  br i1 %v89, label %bb8, label %bb7
}

declare void @llvm.nvvm.membar.gl()

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_KBX_KBX_KBX_Kb0_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v90, %bb42 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb43, label %bb2
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
  br i1 %v53, label %bb11, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  %v56 = phi i32 [ %v55, %bb5 ], [ %v59, %bb8 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb9, label %bb7
bb7:
  %v59 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb8
bb8:
  br label %bb6
bb9:
  br label %bb10
bb10:
  br label %bb48
bb11:
  %v60 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v60, label %bb12, label %bb42
bb12:
  br label %bb13
bb13:
  br label %bb14
bb14:
  %v61 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v62 = extractvalue { i64, i1 } %v61, 0
  br label %bb50
bb15:
  unreachable
bb16:
  %v63 = extractvalue { i64, i64 } %v109, 1
  %v64 = icmp eq i64 %v63, %v29
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb41, label %bb34
bb17:
  br label %bb18
bb18:
  %v66 = bitcast ptr %v11 to ptr
  %v67 = atomicrmw add ptr %v66, i32 1 syncscope("device") monotonic
  br label %bb19
bb19:
  %v68 = icmp sge i32 %v67, %v12
  %v69 = xor i1 %v68, 1
  br i1 %v69, label %bb26, label %bb20
bb20:
  %v70 = bitcast ptr %v11 to ptr
  %v71 = atomicrmw sub ptr %v70, i32 1 syncscope("device") monotonic
  br label %bb21
bb21:
  br label %bb22
bb22:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb23
bb23:
  %v73 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb24
bb24:
  %v74 = bitcast ptr %v13 to ptr
  %v75 = atomicrmw add ptr %v74, i64 1 syncscope("device") monotonic
  br label %bb25
bb25:
  br label %bb45
bb26:
  br label %bb27
bb27:
  %v76 = mul i32 %v67, 3
  %v77 = sext i32 %v76 to i64
  %v78 = getelementptr inbounds i32, ptr %v14, i64 %v77
  store i32 %v15, ptr %v78, align 4
  %v79 = getelementptr inbounds i32, ptr %v78, i64 1
  store i32 %v16, ptr %v79, align 4
  %v80 = getelementptr inbounds i32, ptr %v78, i64 2
  store i32 %v17, ptr %v80, align 4
  br label %bb28
bb28:
  br label %bb29
bb29:
  br label %bb30
bb30:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb31
bb31:
  %v82 = bitcast ptr %v48 to ptr
  %v83 = atomicrmw xchg ptr %v82, i32 %v67 syncscope("device") monotonic
  br label %bb32
bb32:
  br label %bb33
bb33:
  br label %bb45
bb34:
  %v84 = bitcast ptr %v48 to ptr
  %v85 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v84)
  br label %bb35
bb35:
  br label %bb36
bb36:
  %v86 = phi i32 [ %v85, %bb35 ], [ %v89, %bb38 ]
  %v87 = icmp slt i32 %v86, 0
  %v88 = xor i1 %v87, 1
  br i1 %v88, label %bb39, label %bb37
bb37:
  %v89 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v84)
  br label %bb38
bb38:
  br label %bb36
bb39:
  br label %bb40
bb40:
  br label %bb46
bb41:
  br label %bb42
bb42:
  %v90 = add i32 %v40, 1
  br label %bb1
bb43:
  %v91 = bitcast ptr %v13 to ptr
  %v92 = atomicrmw add ptr %v91, i64 1 syncscope("device") monotonic
  br label %bb44
bb44:
  br label %bb49
bb45:
  %v93 = phi i32 [ 4294967295, %bb25 ], [ %v67, %bb33 ]
  br label %bb46
bb46:
  %v94 = phi i32 [ %v86, %bb40 ], [ %v93, %bb45 ]
  br label %bb47
bb47:
  br label %bb48
bb48:
  %v95 = phi i32 [ %v56, %bb10 ], [ %v94, %bb47 ]
  br label %bb49
bb49:
  %v96 = phi i32 [ 4294967295, %bb44 ], [ %v95, %bb48 ]
  ret i32 %v96
bb50:
  %v97 = icmp eq i64 %v62, 18446744073709551615
  br i1 %v97, label %bb51, label %bb52
bb51:
  %v98 = insertvalue { i64, i64 } undef, i64 0, 0
  %v99 = insertvalue { i64, i64 } %v98, i64 %v62, 1
  %v100 = extractvalue { i64, i64 } %v99, 0
  %v101 = extractvalue { i64, i64 } %v99, 1
  br label %bb53
bb52:
  %v102 = insertvalue { i64, i64 } undef, i64 1, 0
  %v103 = insertvalue { i64, i64 } %v102, i64 %v62, 1
  %v104 = extractvalue { i64, i64 } %v103, 0
  %v105 = extractvalue { i64, i64 } %v103, 1
  br label %bb53
bb53:
  %v106 = phi i64 [ %v100, %bb51 ], [ %v104, %bb52 ]
  %v107 = phi i64 [ %v101, %bb51 ], [ %v105, %bb52 ]
  %v108 = insertvalue { i64, i64 } undef, i64 %v106, 0
  %v109 = insertvalue { i64, i64 } %v108, i64 %v107, 1
  %v110 = extractvalue { i64, i64 } %v109, 0
  %v111 = bitcast i64 %v110 to i64
  %v112 = icmp eq i64 %v111, 0
  br i1 %v112, label %bb17, label %bb54
bb54:
  %v113 = icmp eq i64 %v111, 1
  br i1 %v113, label %bb16, label %bb15
}

define i32 @tsdf_rust_cuda__kernels__find_or_insert_128(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) #0 {
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
  %v40 = phi i32 [ 4294967295, %bb0 ], [ %v92, %bb21 ]
  %v41 = phi i32 [ 0, %bb0 ], [ %v93, %bb21 ]
  %v42 = icmp ult i32 %v41, %v30
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb22, label %bb2
bb2:
  %v44 = add i32 %v39, %v41
  %v45 = and i32 %v44, %v10
  %v46 = zext i32 %v45 to i64
  %v47 = mul i64 %v46, 2
  %v48 = getelementptr inbounds i64, ptr %v9, i64 %v47
  %v49 = getelementptr inbounds i64, ptr %v48, i64 1
  %v50 = bitcast ptr %v49 to ptr
  %v51 = bitcast ptr %v48 to ptr
  %v52 = call i64 asm sideeffect "ld.acquire.gpu.b64 $0, [$1];", "=l,l,~{memory}"(ptr %v51)
  br label %bb3
bb3:
  %v53 = icmp eq i64 %v52, %v29
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb6, label %bb4
bb4:
  %v55 = bitcast ptr %v49 to ptr
  %v56 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v55)
  br label %bb5
bb5:
  br label %bb26
bb6:
  %v57 = icmp eq i64 %v52, 18446744073709551615
  br i1 %v57, label %bb7, label %bb21
bb7:
  %v58 = icmp slt i32 %v40, 0
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb14, label %bb8
bb8:
  %v60 = bitcast ptr %v11 to ptr
  %v61 = atomicrmw add ptr %v60, i32 1 syncscope("device") monotonic
  br label %bb9
bb9:
  %v62 = icmp sge i32 %v61, %v12
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb13, label %bb10
bb10:
  %v64 = atomicrmw sub ptr %v60, i32 1 syncscope("device") monotonic
  br label %bb11
bb11:
  %v65 = bitcast ptr %v13 to ptr
  %v66 = atomicrmw add ptr %v65, i64 1 syncscope("device") monotonic
  br label %bb12
bb12:
  br label %bb25
bb13:
  %v67 = mul i32 %v61, 3
  %v68 = sext i32 %v67 to i64
  %v69 = getelementptr inbounds i32, ptr %v14, i64 %v68
  store i32 %v15, ptr %v69, align 4
  %v70 = getelementptr inbounds i32, ptr %v69, i64 1
  store i32 %v16, ptr %v70, align 4
  %v71 = getelementptr inbounds i32, ptr %v69, i64 2
  store i32 %v17, ptr %v71, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  %v72 = phi i32 [ %v61, %bb13 ], [ %v40, %bb14 ]
  %v73 = bitcast i64 %v29 to i64
  %v74 = bitcast i32 %v72 to i32
  %v75 = zext i32 %v74 to i64
  %v76 = ptrtoint ptr %v48 to i64
  %v77 = call { i64, i64 } asm sideeffect "{ .reg .b128 t, e, d;\0A\09.reg .u64 g;\0A\09cvta.to.global.u64 g, $6;\0A\09mov.b128 e, {$2, $3};\0A\09mov.b128 d, {$4, $5};\0A\09atom.global.acq_rel.gpu.cas.b128 t, [g], e, d;\0A\09mov.b128 {$0, $1}, t; }", "=l,=l,l,l,l,l,l,~{memory}"(i64 18446744073709551615, i64 4294967295, i64 %v73, i64 %v75, i64 %v76) #0
  %v78 = extractvalue { i64, i64 } %v77, 0
  %v79 = extractvalue { i64, i64 } %v77, 1
  %v80 = insertvalue { i64, i64 } undef, i64 %v78, 0
  %v81 = insertvalue { i64, i64 } %v80, i64 %v79, 1
  br label %bb16
bb16:
  %v82 = extractvalue { i64, i64 } %v81, 0
  %v83 = extractvalue { i64, i64 } %v81, 1
  %v84 = icmp eq i64 %v82, 18446744073709551615
  br i1 %v84, label %bb17, label %bb18
bb17:
  br label %bb24
bb18:
  %v85 = bitcast i64 %v82 to i64
  %v86 = icmp eq i64 %v85, %v29
  %v87 = xor i1 %v86, 1
  br i1 %v87, label %bb20, label %bb19
bb19:
  %v88 = bitcast ptr %v11 to ptr
  %v89 = add i32 %v72, 1
  %v90 = cmpxchg ptr %v88, i32 %v89, i32 %v72 syncscope("device") monotonic monotonic
  %v91 = extractvalue { i32, i1 } %v90, 0
  br label %bb28
bb20:
  br label %bb21
bb21:
  %v92 = phi i32 [ %v40, %bb6 ], [ %v72, %bb20 ]
  %v93 = add i32 %v41, 1
  br label %bb1
bb22:
  %v94 = bitcast ptr %v13 to ptr
  %v95 = atomicrmw add ptr %v94, i64 1 syncscope("device") monotonic
  br label %bb23
bb23:
  br label %bb27
bb24:
  %v96 = phi i32 [ %v72, %bb17 ], [ %v119, %bb34 ]
  br label %bb25
bb25:
  %v97 = phi i32 [ 4294967295, %bb12 ], [ %v96, %bb24 ]
  br label %bb26
bb26:
  %v98 = phi i32 [ %v56, %bb5 ], [ %v97, %bb25 ]
  br label %bb27
bb27:
  %v99 = phi i32 [ 4294967295, %bb23 ], [ %v98, %bb26 ]
  ret i32 %v99
bb28:
  %v100 = icmp eq i32 %v91, %v89
  %v101 = xor i1 %v100, 1
  br i1 %v101, label %bb30, label %bb29
bb29:
  %v102 = insertvalue { i32, i32 } undef, i32 0, 0
  %v103 = insertvalue { i32, i32 } %v102, i32 %v91, 1
  %v104 = extractvalue { i32, i32 } %v103, 0
  %v105 = extractvalue { i32, i32 } %v103, 1
  br label %bb31
bb30:
  %v106 = insertvalue { i32, i32 } undef, i32 1, 0
  %v107 = insertvalue { i32, i32 } %v106, i32 %v91, 1
  %v108 = extractvalue { i32, i32 } %v107, 0
  %v109 = extractvalue { i32, i32 } %v107, 1
  br label %bb31
bb31:
  %v110 = phi i32 [ %v104, %bb29 ], [ %v108, %bb30 ]
  %v111 = phi i32 [ %v105, %bb29 ], [ %v109, %bb30 ]
  %v112 = insertvalue { i32, i32 } undef, i32 %v110, 0
  %v113 = insertvalue { i32, i32 } %v112, i32 %v111, 1
  %v114 = extractvalue { i32, i32 } %v113, 0
  %v115 = zext i32 %v114 to i64
  %v116 = icmp eq i64 %v115, 0
  br i1 %v116, label %bb34, label %bb32
bb32:
  %v117 = icmp eq i64 %v115, 1
  br i1 %v117, label %bb34, label %bb33
bb33:
  unreachable
bb34:
  %v118 = and i64 %v83, 4294967295
  %v119 = trunc i64 %v118 to i32
  br label %bb24
}

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_Kb0_KBX_KBX_KB15_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v91, %bb42 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb43, label %bb2
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
  br i1 %v53, label %bb11, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  %v56 = phi i32 [ %v55, %bb5 ], [ %v59, %bb8 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb9, label %bb7
bb7:
  %v59 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb8
bb8:
  br label %bb6
bb9:
  br label %bb10
bb10:
  br label %bb48
bb11:
  %v60 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v60, label %bb12, label %bb42
bb12:
  br label %bb13
bb13:
  br label %bb14
bb14:
  %v61 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v62 = extractvalue { i64, i1 } %v61, 0
  br label %bb50
bb15:
  unreachable
bb16:
  %v63 = extractvalue { i64, i64 } %v110, 1
  %v64 = icmp eq i64 %v63, %v29
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb41, label %bb34
bb17:
  br label %bb18
bb18:
  %v66 = and i32 %v15, 1023
  %v67 = icmp slt i32 %v66, 0
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb57, label %bb55
bb19:
  %v69 = icmp sge i32 %v118, %v12
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb26, label %bb20
bb20:
  %v71 = bitcast ptr %v11 to ptr
  %v72 = atomicrmw sub ptr %v71, i32 1 syncscope("device") monotonic
  br label %bb21
bb21:
  br label %bb22
bb22:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb23
bb23:
  %v74 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb24
bb24:
  %v75 = bitcast ptr %v13 to ptr
  %v76 = atomicrmw add ptr %v75, i64 1 syncscope("device") monotonic
  br label %bb25
bb25:
  br label %bb45
bb26:
  br label %bb27
bb27:
  %v77 = mul i32 %v118, 3
  %v78 = sext i32 %v77 to i64
  %v79 = getelementptr inbounds i32, ptr %v14, i64 %v78
  store i32 %v15, ptr %v79, align 4
  %v80 = getelementptr inbounds i32, ptr %v79, i64 1
  store i32 %v16, ptr %v80, align 4
  %v81 = getelementptr inbounds i32, ptr %v79, i64 2
  store i32 %v17, ptr %v81, align 4
  br label %bb28
bb28:
  br label %bb29
bb29:
  br label %bb30
bb30:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb31
bb31:
  %v83 = bitcast ptr %v48 to ptr
  %v84 = atomicrmw xchg ptr %v83, i32 %v118 syncscope("device") monotonic
  br label %bb32
bb32:
  br label %bb33
bb33:
  br label %bb45
bb34:
  %v85 = bitcast ptr %v48 to ptr
  %v86 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v85)
  br label %bb35
bb35:
  br label %bb36
bb36:
  %v87 = phi i32 [ %v86, %bb35 ], [ %v90, %bb38 ]
  %v88 = icmp slt i32 %v87, 0
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb39, label %bb37
bb37:
  %v90 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v85)
  br label %bb38
bb38:
  br label %bb36
bb39:
  br label %bb40
bb40:
  br label %bb46
bb41:
  br label %bb42
bb42:
  %v91 = add i32 %v40, 1
  br label %bb1
bb43:
  %v92 = bitcast ptr %v13 to ptr
  %v93 = atomicrmw add ptr %v92, i64 1 syncscope("device") monotonic
  br label %bb44
bb44:
  br label %bb49
bb45:
  %v94 = phi i32 [ 4294967295, %bb25 ], [ %v118, %bb33 ]
  br label %bb46
bb46:
  %v95 = phi i32 [ %v87, %bb40 ], [ %v94, %bb45 ]
  br label %bb47
bb47:
  br label %bb48
bb48:
  %v96 = phi i32 [ %v56, %bb10 ], [ %v95, %bb47 ]
  br label %bb49
bb49:
  %v97 = phi i32 [ 4294967295, %bb44 ], [ %v96, %bb48 ]
  ret i32 %v97
bb50:
  %v98 = icmp eq i64 %v62, 18446744073709551615
  br i1 %v98, label %bb51, label %bb52
bb51:
  %v99 = insertvalue { i64, i64 } undef, i64 0, 0
  %v100 = insertvalue { i64, i64 } %v99, i64 %v62, 1
  %v101 = extractvalue { i64, i64 } %v100, 0
  %v102 = extractvalue { i64, i64 } %v100, 1
  br label %bb53
bb52:
  %v103 = insertvalue { i64, i64 } undef, i64 1, 0
  %v104 = insertvalue { i64, i64 } %v103, i64 %v62, 1
  %v105 = extractvalue { i64, i64 } %v104, 0
  %v106 = extractvalue { i64, i64 } %v104, 1
  br label %bb53
bb53:
  %v107 = phi i64 [ %v101, %bb51 ], [ %v105, %bb52 ]
  %v108 = phi i64 [ %v102, %bb51 ], [ %v106, %bb52 ]
  %v109 = insertvalue { i64, i64 } undef, i64 %v107, 0
  %v110 = insertvalue { i64, i64 } %v109, i64 %v108, 1
  %v111 = extractvalue { i64, i64 } %v110, 0
  %v112 = bitcast i64 %v111 to i64
  %v113 = icmp eq i64 %v112, 0
  br i1 %v113, label %bb17, label %bb54
bb54:
  %v114 = icmp eq i64 %v112, 1
  br i1 %v114, label %bb16, label %bb15
bb55:
  %v115 = icmp eq i32 %v66, 2147483648
  %v116 = xor i1 %v115, 1
  br i1 %v116, label %bb56, label %bb59
bb56:
  %v117 = sub i32 0, %v66
  br label %bb58
bb57:
  br label %bb58
bb58:
  %v118 = phi i32 [ %v117, %bb56 ], [ %v66, %bb57 ]
  br label %bb19
bb59:
  call void @llvm.trap() #0
  unreachable
}

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_KBX_KBX_KBX_KBX_KBX_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v92, %bb44 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb45, label %bb2
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
  br i1 %v53, label %bb11, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  %v56 = phi i32 [ %v55, %bb5 ], [ %v59, %bb8 ]
  %v57 = icmp slt i32 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb9, label %bb7
bb7:
  %v59 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb8
bb8:
  br label %bb6
bb9:
  br label %bb10
bb10:
  br label %bb50
bb11:
  %v60 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v60, label %bb12, label %bb44
bb12:
  br label %bb13
bb13:
  br label %bb14
bb14:
  %v61 = bitcast ptr %v13 to ptr
  %v62 = atomicrmw add ptr %v61, i64 1 syncscope("device") monotonic
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v63 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v64 = extractvalue { i64, i1 } %v63, 0
  br label %bb52
bb17:
  unreachable
bb18:
  %v65 = extractvalue { i64, i64 } %v111, 1
  %v66 = icmp eq i64 %v65, %v29
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb43, label %bb36
bb19:
  br label %bb20
bb20:
  %v68 = bitcast ptr %v11 to ptr
  %v69 = atomicrmw add ptr %v68, i32 1 syncscope("device") monotonic
  br label %bb21
bb21:
  %v70 = icmp sge i32 %v69, %v12
  %v71 = xor i1 %v70, 1
  br i1 %v71, label %bb28, label %bb22
bb22:
  %v72 = bitcast ptr %v11 to ptr
  %v73 = atomicrmw sub ptr %v72, i32 1 syncscope("device") monotonic
  br label %bb23
bb23:
  br label %bb24
bb24:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb25
bb25:
  %v75 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb26
bb26:
  %v76 = bitcast ptr %v13 to ptr
  %v77 = atomicrmw add ptr %v76, i64 1 syncscope("device") monotonic
  br label %bb27
bb27:
  br label %bb47
bb28:
  br label %bb29
bb29:
  %v78 = mul i32 %v69, 3
  %v79 = sext i32 %v78 to i64
  %v80 = getelementptr inbounds i32, ptr %v14, i64 %v79
  store i32 %v15, ptr %v80, align 4
  %v81 = getelementptr inbounds i32, ptr %v80, i64 1
  store i32 %v16, ptr %v81, align 4
  %v82 = getelementptr inbounds i32, ptr %v80, i64 2
  store i32 %v17, ptr %v82, align 4
  br label %bb30
bb30:
  br label %bb31
bb31:
  br label %bb32
bb32:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb33
bb33:
  %v84 = bitcast ptr %v48 to ptr
  %v85 = atomicrmw xchg ptr %v84, i32 %v69 syncscope("device") monotonic
  br label %bb34
bb34:
  br label %bb35
bb35:
  br label %bb47
bb36:
  %v86 = bitcast ptr %v48 to ptr
  %v87 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v86)
  br label %bb37
bb37:
  br label %bb38
bb38:
  %v88 = phi i32 [ %v87, %bb37 ], [ %v91, %bb40 ]
  %v89 = icmp slt i32 %v88, 0
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb41, label %bb39
bb39:
  %v91 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v86)
  br label %bb40
bb40:
  br label %bb38
bb41:
  br label %bb42
bb42:
  br label %bb48
bb43:
  br label %bb44
bb44:
  %v92 = add i32 %v40, 1
  br label %bb1
bb45:
  %v93 = bitcast ptr %v13 to ptr
  %v94 = atomicrmw add ptr %v93, i64 1 syncscope("device") monotonic
  br label %bb46
bb46:
  br label %bb51
bb47:
  %v95 = phi i32 [ 4294967295, %bb27 ], [ %v69, %bb35 ]
  br label %bb48
bb48:
  %v96 = phi i32 [ %v88, %bb42 ], [ %v95, %bb47 ]
  br label %bb49
bb49:
  br label %bb50
bb50:
  %v97 = phi i32 [ %v56, %bb10 ], [ %v96, %bb49 ]
  br label %bb51
bb51:
  %v98 = phi i32 [ 4294967295, %bb46 ], [ %v97, %bb50 ]
  ret i32 %v98
bb52:
  %v99 = icmp eq i64 %v64, 18446744073709551615
  br i1 %v99, label %bb53, label %bb54
bb53:
  %v100 = insertvalue { i64, i64 } undef, i64 0, 0
  %v101 = insertvalue { i64, i64 } %v100, i64 %v64, 1
  %v102 = extractvalue { i64, i64 } %v101, 0
  %v103 = extractvalue { i64, i64 } %v101, 1
  br label %bb55
bb54:
  %v104 = insertvalue { i64, i64 } undef, i64 1, 0
  %v105 = insertvalue { i64, i64 } %v104, i64 %v64, 1
  %v106 = extractvalue { i64, i64 } %v105, 0
  %v107 = extractvalue { i64, i64 } %v105, 1
  br label %bb55
bb55:
  %v108 = phi i64 [ %v102, %bb53 ], [ %v106, %bb54 ]
  %v109 = phi i64 [ %v103, %bb53 ], [ %v107, %bb54 ]
  %v110 = insertvalue { i64, i64 } undef, i64 %v108, 0
  %v111 = insertvalue { i64, i64 } %v110, i64 %v109, 1
  %v112 = extractvalue { i64, i64 } %v111, 0
  %v113 = bitcast i64 %v112 to i64
  %v114 = icmp eq i64 %v113, 0
  br i1 %v114, label %bb19, label %bb56
bb56:
  %v115 = icmp eq i64 %v113, 1
  br i1 %v115, label %bb18, label %bb17
}

define i32 @_RINvNtCshAQQRahQDk9_14tsdf_rust_cuda7kernels14find_or_insertKb1_Kb0_KBX_KBX_KBX_KB11_EB4_(ptr %v0, i32 %v1, ptr %v2, i32 %v3, ptr %v4, ptr %v5, i32 %v6, i32 %v7, i32 %v8) alwaysinline #0 {
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
  %v40 = phi i32 [ 0, %bb0 ], [ %v82, %bb34 ]
  %v41 = icmp ult i32 %v40, %v30
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb35, label %bb2
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
  br i1 %v53, label %bb7, label %bb4
bb4:
  %v54 = bitcast ptr %v48 to ptr
  %v55 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v54)
  br label %bb5
bb5:
  br label %bb6
bb6:
  br label %bb40
bb7:
  %v56 = icmp eq i64 %v51, 18446744073709551615
  br i1 %v56, label %bb8, label %bb34
bb8:
  br label %bb9
bb9:
  br label %bb10
bb10:
  %v57 = cmpxchg ptr %v50, i64 18446744073709551615, i64 %v29 syncscope("device") monotonic monotonic
  %v58 = extractvalue { i64, i1 } %v57, 0
  br label %bb42
bb11:
  unreachable
bb12:
  %v59 = extractvalue { i64, i64 } %v101, 1
  %v60 = icmp eq i64 %v59, %v29
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb33, label %bb30
bb13:
  br label %bb14
bb14:
  %v62 = bitcast ptr %v11 to ptr
  %v63 = atomicrmw add ptr %v62, i32 1 syncscope("device") monotonic
  br label %bb15
bb15:
  %v64 = icmp sge i32 %v63, %v12
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb22, label %bb16
bb16:
  %v66 = bitcast ptr %v11 to ptr
  %v67 = atomicrmw sub ptr %v66, i32 1 syncscope("device") monotonic
  br label %bb17
bb17:
  br label %bb18
bb18:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb19
bb19:
  %v69 = atomicrmw xchg ptr %v50, i64 18446744073709551615 syncscope("device") monotonic
  br label %bb20
bb20:
  %v70 = bitcast ptr %v13 to ptr
  %v71 = atomicrmw add ptr %v70, i64 1 syncscope("device") monotonic
  br label %bb21
bb21:
  br label %bb37
bb22:
  br label %bb23
bb23:
  %v72 = mul i32 %v63, 3
  %v73 = sext i32 %v72 to i64
  %v74 = getelementptr inbounds i32, ptr %v14, i64 %v73
  store i32 %v15, ptr %v74, align 4
  %v75 = getelementptr inbounds i32, ptr %v74, i64 1
  store i32 %v16, ptr %v75, align 4
  %v76 = getelementptr inbounds i32, ptr %v74, i64 2
  store i32 %v17, ptr %v76, align 4
  br label %bb24
bb24:
  br label %bb25
bb25:
  br label %bb26
bb26:
  call void @llvm.nvvm.membar.gl() #0
  br label %bb27
bb27:
  %v78 = bitcast ptr %v48 to ptr
  %v79 = atomicrmw xchg ptr %v78, i32 %v63 syncscope("device") monotonic
  br label %bb28
bb28:
  br label %bb29
bb29:
  br label %bb37
bb30:
  %v80 = bitcast ptr %v48 to ptr
  %v81 = call i32 asm sideeffect "ld.relaxed.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %v80)
  br label %bb31
bb31:
  br label %bb32
bb32:
  br label %bb38
bb33:
  br label %bb34
bb34:
  %v82 = add i32 %v40, 1
  br label %bb1
bb35:
  %v83 = bitcast ptr %v13 to ptr
  %v84 = atomicrmw add ptr %v83, i64 1 syncscope("device") monotonic
  br label %bb36
bb36:
  br label %bb41
bb37:
  %v85 = phi i32 [ 4294967295, %bb21 ], [ %v63, %bb29 ]
  br label %bb38
bb38:
  %v86 = phi i32 [ %v81, %bb32 ], [ %v85, %bb37 ]
  br label %bb39
bb39:
  br label %bb40
bb40:
  %v87 = phi i32 [ %v55, %bb6 ], [ %v86, %bb39 ]
  br label %bb41
bb41:
  %v88 = phi i32 [ 4294967295, %bb36 ], [ %v87, %bb40 ]
  ret i32 %v88
bb42:
  %v89 = icmp eq i64 %v58, 18446744073709551615
  br i1 %v89, label %bb43, label %bb44
bb43:
  %v90 = insertvalue { i64, i64 } undef, i64 0, 0
  %v91 = insertvalue { i64, i64 } %v90, i64 %v58, 1
  %v92 = extractvalue { i64, i64 } %v91, 0
  %v93 = extractvalue { i64, i64 } %v91, 1
  br label %bb45
bb44:
  %v94 = insertvalue { i64, i64 } undef, i64 1, 0
  %v95 = insertvalue { i64, i64 } %v94, i64 %v58, 1
  %v96 = extractvalue { i64, i64 } %v95, 0
  %v97 = extractvalue { i64, i64 } %v95, 1
  br label %bb45
bb45:
  %v98 = phi i64 [ %v92, %bb43 ], [ %v96, %bb44 ]
  %v99 = phi i64 [ %v93, %bb43 ], [ %v97, %bb44 ]
  %v100 = insertvalue { i64, i64 } undef, i64 %v98, 0
  %v101 = insertvalue { i64, i64 } %v100, i64 %v99, 1
  %v102 = extractvalue { i64, i64 } %v101, 0
  %v103 = bitcast i64 %v102 to i64
  %v104 = icmp eq i64 %v103, 0
  br i1 %v104, label %bb13, label %bb46
bb46:
  %v105 = icmp eq i64 %v103, 1
  br i1 %v105, label %bb12, label %bb11
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


@llvm.used = appending global [11 x ptr] [ptr @alloc_kernel, ptr @alloc_kernel_cas128, ptr @alloc_kernel_countcas, ptr @alloc_kernel_nocas, ptr @alloc_kernel_nocount, ptr @alloc_kernel_nofence, ptr @alloc_kernel_nopublish, ptr @alloc_kernel_nospin, ptr @alloc_kernel_slotidx, ptr @cas128_selftest, ptr @update_kernel], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{ptr @update_kernel, !"kernel", i32 1}
!1 = !{ptr @alloc_kernel_nofence, !"kernel", i32 1}
!2 = !{ptr @alloc_kernel_nocas, !"kernel", i32 1}
!3 = !{ptr @alloc_kernel_nopublish, !"kernel", i32 1}
!4 = !{ptr @alloc_kernel_slotidx, !"kernel", i32 1}
!5 = !{ptr @alloc_kernel, !"kernel", i32 1}
!6 = !{ptr @alloc_kernel_cas128, !"kernel", i32 1}
!7 = !{ptr @alloc_kernel_nocount, !"kernel", i32 1}
!8 = !{ptr @cas128_selftest, !"kernel", i32 1}
!9 = !{ptr @alloc_kernel_countcas, !"kernel", i32 1}
!10 = !{ptr @alloc_kernel_nospin, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}

!nvvmir.version = !{!11}
!11 = !{i32 2, i32 0, i32 3, i32 2}
