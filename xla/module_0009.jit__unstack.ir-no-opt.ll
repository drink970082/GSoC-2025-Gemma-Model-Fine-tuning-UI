; ModuleID = 'jit__unstack'
source_filename = "jit__unstack"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_slice_1(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 128 dereferenceable(4) %1) #0 {
  %3 = getelementptr inbounds i32, ptr %0, i32 1
  %4 = load i32, ptr %3, align 4, !invariant.load !1
  store i32 %4, ptr %1, align 4
  ret void
}

define ptx_kernel void @wrapped_slice(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 128 dereferenceable(4) %1) #0 {
  %3 = load i32, ptr %0, align 4, !invariant.load !1
  store i32 %3, ptr %1, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
