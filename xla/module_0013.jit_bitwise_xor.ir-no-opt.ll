; ModuleID = 'jit_bitwise_xor'
source_filename = "jit_bitwise_xor"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @wrapped_xor(ptr noalias align 16 dereferenceable(4) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 128 dereferenceable(4) %2) #0 {
  %4 = load i32, ptr %0, align 4, !invariant.load !1
  %5 = load i32, ptr %1, align 4, !invariant.load !1
  %6 = xor i32 %4, %5
  store i32 %6, ptr %2, align 4
  ret void
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
