; ModuleID = 'jit__threefry_seed'
source_filename = "jit__threefry_seed"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 16 dereferenceable(4) %0, ptr noalias align 128 dereferenceable(8) %1) #0 {
  %3 = call i32 @fused_concatenate__epilogue__concatenate_8_1(ptr %0, i32 0, i32 0)
  store i32 %3, ptr %1, align 4
  %4 = load i32, ptr %0, align 4, !invariant.load !1
  %5 = call i32 @fused_concatenate__epilogue__concatenate_8_1(ptr %0, i32 1, i32 %4)
  %6 = getelementptr inbounds i32, ptr %1, i32 1
  store i32 %5, ptr %6, align 4
  ret void
}

define internal i32 @fused_concatenate__epilogue__concatenate_8_1(ptr noalias %0, i32 %1, i32 %2) {
  ret i32 %2
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
