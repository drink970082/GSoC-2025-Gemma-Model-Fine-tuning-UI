; ModuleID = 'jit__threefry_fold_in'
source_filename = "jit__threefry_fold_in"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @input_concatenate_fusion(ptr noalias readonly align 16 captures(none) dereferenceable(8) %0, ptr noalias readonly align 16 captures(none) dereferenceable(4) %1, ptr noalias writeonly align 128 captures(none) dereferenceable(8) initializes((0, 8)) %2) local_unnamed_addr #0 {
  %4 = addrspacecast ptr %0 to ptr addrspace(1)
  %5 = addrspacecast ptr %1 to ptr addrspace(1)
  %6 = addrspacecast ptr %2 to ptr addrspace(1)
  %7 = load <2 x i32>, ptr addrspace(1) %4, align 16, !invariant.load !2
  %.val2 = extractelement <2 x i32> %7, i32 0
  %.val13 = extractelement <2 x i32> %7, i32 1
  %8 = load i32, ptr addrspace(1) %5, align 16, !invariant.load !2
  %9 = add i32 %.val13, %8
  %10 = add i32 %9, %.val2
  %11 = tail call i32 @llvm.fshl.i32(i32 %9, i32 %9, i32 13)
  %12 = xor i32 %10, %11
  %13 = add i32 %12, %10
  %14 = tail call i32 @llvm.fshl.i32(i32 %12, i32 %12, i32 15)
  %15 = xor i32 %13, %14
  %16 = add i32 %15, %13
  %17 = tail call i32 @llvm.fshl.i32(i32 %15, i32 %15, i32 26)
  %18 = xor i32 %16, %17
  %19 = add i32 %18, %16
  %20 = add i32 %19, %.val13
  %21 = tail call i32 @llvm.fshl.i32(i32 %18, i32 %18, i32 6)
  %22 = xor i32 %19, %21
  %23 = xor i32 %.val2, %.val13
  %24 = xor i32 %23, 466688986
  %25 = add i32 %24, 1
  %26 = add i32 %25, %22
  %27 = add i32 %20, %26
  %28 = tail call i32 @llvm.fshl.i32(i32 %26, i32 %26, i32 17)
  %29 = xor i32 %27, %28
  %30 = add i32 %29, %27
  %31 = tail call i32 @llvm.fshl.i32(i32 %29, i32 %29, i32 29)
  %32 = xor i32 %30, %31
  %33 = add i32 %32, %30
  %34 = tail call i32 @llvm.fshl.i32(i32 %32, i32 %32, i32 16)
  %35 = xor i32 %33, %34
  %36 = add i32 %35, %33
  %37 = add i32 %36, %24
  %38 = tail call i32 @llvm.fshl.i32(i32 %35, i32 %35, i32 24)
  %39 = xor i32 %36, %38
  %40 = add i32 %.val2, 2
  %41 = add i32 %40, %39
  %42 = add i32 %37, %41
  %43 = tail call i32 @llvm.fshl.i32(i32 %41, i32 %41, i32 13)
  %44 = xor i32 %42, %43
  %45 = add i32 %44, %42
  %46 = tail call i32 @llvm.fshl.i32(i32 %44, i32 %44, i32 15)
  %47 = xor i32 %45, %46
  %48 = add i32 %47, %45
  %49 = tail call i32 @llvm.fshl.i32(i32 %47, i32 %47, i32 26)
  %50 = xor i32 %48, %49
  %51 = add i32 %50, %48
  %52 = add i32 %51, %.val2
  %53 = tail call i32 @llvm.fshl.i32(i32 %50, i32 %50, i32 6)
  %54 = xor i32 %51, %53
  %55 = add i32 %.val13, 3
  %56 = add i32 %55, %54
  %57 = add i32 %52, %56
  %58 = tail call i32 @llvm.fshl.i32(i32 %56, i32 %56, i32 17)
  %59 = xor i32 %57, %58
  %60 = add i32 %59, %57
  %61 = tail call i32 @llvm.fshl.i32(i32 %59, i32 %59, i32 29)
  %62 = xor i32 %60, %61
  %63 = add i32 %62, %60
  %64 = tail call i32 @llvm.fshl.i32(i32 %62, i32 %62, i32 16)
  %65 = xor i32 %63, %64
  %66 = add i32 %65, %63
  %67 = add i32 %66, %.val13
  %68 = tail call i32 @llvm.fshl.i32(i32 %65, i32 %65, i32 24)
  %69 = xor i32 %66, %68
  %70 = add i32 %24, 4
  %71 = add i32 %70, %69
  %72 = add i32 %67, %71
  %73 = tail call i32 @llvm.fshl.i32(i32 %71, i32 %71, i32 13)
  %74 = xor i32 %72, %73
  %75 = add i32 %74, %72
  %76 = tail call i32 @llvm.fshl.i32(i32 %74, i32 %74, i32 15)
  %77 = xor i32 %75, %76
  %78 = add i32 %77, %75
  %79 = tail call i32 @llvm.fshl.i32(i32 %77, i32 %77, i32 26)
  %80 = xor i32 %78, %79
  %81 = add i32 %80, %78
  %82 = add i32 %81, %24
  %83 = tail call i32 @llvm.fshl.i32(i32 %80, i32 %80, i32 6)
  %84 = xor i32 %81, %83
  %85 = add i32 %.val2, 5
  %86 = add i32 %85, %84
  %87 = insertelement <2 x i32> poison, i32 %82, i32 0
  %88 = insertelement <2 x i32> %87, i32 %86, i32 1
  store <2 x i32> %88, ptr addrspace(1) %6, align 128
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "nvvm.reqntid"="1,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{}
