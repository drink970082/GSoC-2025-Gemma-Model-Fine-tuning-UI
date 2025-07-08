; ModuleID = 'jit_bitwise_xor'
source_filename = "jit_bitwise_xor"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @wrapped_xor(ptr noalias readonly align 16 captures(none) dereferenceable(4) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 128 captures(none) dereferenceable(4) initializes((0, 4)) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load i32, ptr addrspace(1) %4, align 16, !invariant.load !2
  %8 = load i32, ptr addrspace(1) %5, align 16, !invariant.load !2
  %9 = xor i32 %8, %7
  store i32 %9, ptr addrspace(1) %6, align 128
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
