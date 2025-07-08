; ModuleID = 'jit__threefry_seed'
source_filename = "jit__threefry_seed"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(4) %0, ptr noalias writeonly align 128 captures(none) dereferenceable(8) initializes((0, 8)) %1) local_unnamed_addr #0 {
  %3 = addrspacecast ptr %1 to ptr addrspace(1)
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = load i32, ptr addrspace(1) %4, align 16, !invariant.load !2
  %6 = insertelement <2 x i32> <i32 0, i32 poison>, i32 %5, i32 1
  store <2 x i32> %6, ptr addrspace(1) %3, align 128
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
