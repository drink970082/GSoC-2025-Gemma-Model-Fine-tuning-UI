; ModuleID = 'jit__threefry_fold_in'
source_filename = "jit__threefry_fold_in"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @input_concatenate_fusion(ptr noalias align 16 dereferenceable(8) %0, ptr noalias align 16 dereferenceable(4) %1, ptr noalias align 128 dereferenceable(8) %2) #0 {
  %4 = call i32 @fused_concatenate_slice_7_83(ptr %0, ptr %1, i32 0)
  %5 = load i32, ptr %1, align 4, !invariant.load !1
  %6 = call i32 @fused_concatenate_slice_9_99(ptr %0, ptr %1, i32 0)
  %7 = add i32 %5, %6
  %8 = add i32 %4, %7
  %9 = shl i32 %7, 13
  %10 = lshr i32 %7, 19
  %11 = or i32 %9, %10
  %12 = xor i32 %8, %11
  %13 = add i32 %8, %12
  %14 = shl i32 %12, 15
  %15 = lshr i32 %12, 17
  %16 = or i32 %14, %15
  %17 = xor i32 %13, %16
  %18 = add i32 %13, %17
  %19 = shl i32 %17, 26
  %20 = lshr i32 %17, 6
  %21 = or i32 %19, %20
  %22 = xor i32 %18, %21
  %23 = add i32 %18, %22
  %24 = add i32 %23, %6
  %25 = shl i32 %22, 6
  %26 = lshr i32 %22, 26
  %27 = or i32 %25, %26
  %28 = xor i32 %23, %27
  %29 = xor i32 %4, %6
  %30 = xor i32 %29, 466688986
  %31 = add i32 %28, %30
  %32 = add i32 %31, 1
  %33 = add i32 %24, %32
  %34 = shl i32 %32, 17
  %35 = lshr i32 %32, 15
  %36 = or i32 %34, %35
  %37 = xor i32 %33, %36
  %38 = add i32 %33, %37
  %39 = shl i32 %37, 29
  %40 = lshr i32 %37, 3
  %41 = or i32 %39, %40
  %42 = xor i32 %38, %41
  %43 = add i32 %38, %42
  %44 = shl i32 %42, 16
  %45 = lshr i32 %42, 16
  %46 = or i32 %44, %45
  %47 = xor i32 %43, %46
  %48 = add i32 %43, %47
  %49 = add i32 %48, %30
  %50 = shl i32 %47, 24
  %51 = lshr i32 %47, 8
  %52 = or i32 %50, %51
  %53 = xor i32 %48, %52
  %54 = add i32 %53, %4
  %55 = add i32 %54, 2
  %56 = add i32 %49, %55
  %57 = shl i32 %55, 13
  %58 = lshr i32 %55, 19
  %59 = or i32 %57, %58
  %60 = xor i32 %56, %59
  %61 = add i32 %56, %60
  %62 = shl i32 %60, 15
  %63 = lshr i32 %60, 17
  %64 = or i32 %62, %63
  %65 = xor i32 %61, %64
  %66 = add i32 %61, %65
  %67 = shl i32 %65, 26
  %68 = lshr i32 %65, 6
  %69 = or i32 %67, %68
  %70 = xor i32 %66, %69
  %71 = add i32 %66, %70
  %72 = add i32 %71, %4
  %73 = shl i32 %70, 6
  %74 = lshr i32 %70, 26
  %75 = or i32 %73, %74
  %76 = xor i32 %71, %75
  %77 = add i32 %76, %6
  %78 = add i32 %77, 3
  %79 = add i32 %72, %78
  %80 = shl i32 %78, 17
  %81 = lshr i32 %78, 15
  %82 = or i32 %80, %81
  %83 = xor i32 %79, %82
  %84 = add i32 %79, %83
  %85 = shl i32 %83, 29
  %86 = lshr i32 %83, 3
  %87 = or i32 %85, %86
  %88 = xor i32 %84, %87
  %89 = add i32 %84, %88
  %90 = shl i32 %88, 16
  %91 = lshr i32 %88, 16
  %92 = or i32 %90, %91
  %93 = xor i32 %89, %92
  %94 = add i32 %89, %93
  %95 = add i32 %94, %6
  %96 = shl i32 %93, 24
  %97 = lshr i32 %93, 8
  %98 = or i32 %96, %97
  %99 = xor i32 %94, %98
  %100 = add i32 %99, %30
  %101 = add i32 %100, 4
  %102 = add i32 %95, %101
  %103 = shl i32 %101, 13
  %104 = lshr i32 %101, 19
  %105 = or i32 %103, %104
  %106 = xor i32 %102, %105
  %107 = add i32 %102, %106
  %108 = shl i32 %106, 15
  %109 = lshr i32 %106, 17
  %110 = or i32 %108, %109
  %111 = xor i32 %107, %110
  %112 = add i32 %107, %111
  %113 = shl i32 %111, 26
  %114 = lshr i32 %111, 6
  %115 = or i32 %113, %114
  %116 = xor i32 %112, %115
  %117 = add i32 %112, %116
  %118 = add i32 %117, %30
  %119 = call i32 @fused_concatenate__epilogue__concatenate_167_1(ptr %0, ptr %1, i32 0, i32 %118)
  store i32 %119, ptr %2, align 4
  %120 = shl i32 %116, 6
  %121 = lshr i32 %116, 26
  %122 = or i32 %120, %121
  %123 = xor i32 %117, %122
  %124 = add i32 %123, %4
  %125 = add i32 %124, 5
  %126 = call i32 @fused_concatenate__epilogue__concatenate_167_1(ptr %0, ptr %1, i32 1, i32 %125)
  %127 = getelementptr inbounds i32, ptr %2, i32 1
  store i32 %126, ptr %127, align 4
  ret void
}

define internal i32 @fused_concatenate_slice_7_83(ptr noalias %0, ptr noalias %1, i32 %2) {
  %4 = getelementptr inbounds i32, ptr %0, i32 %2
  %5 = load i32, ptr %4, align 4, !invariant.load !1
  ret i32 %5
}

define internal i32 @fused_concatenate_slice_9_99(ptr noalias %0, ptr noalias %1, i32 %2) {
  %4 = getelementptr inbounds i32, ptr %0, i32 1
  %5 = load i32, ptr %4, align 4, !invariant.load !1
  ret i32 %5
}

define internal i32 @fused_concatenate__epilogue__concatenate_167_1(ptr noalias %0, ptr noalias %1, i32 %2, i32 %3) {
  ret i32 %3
}

attributes #0 = { "nvvm.reqntid"="1,1,1" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{}
