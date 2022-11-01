# CUTLASS Profiler 使用指南

CUTLASS中提供了许多kernel。[CUTLASS Profiler](https://github.com/NVIDIA/cutlass/blob/master/media/docs/profiler.md)可以帮助我们根据硬件和形状选出最优的kernel。

* 编译Profiler

  首先clone cutlass，并进入build目录
  
  ```shell
  git clone https://github.com/NVIDIA/cutlass.git
  mkdir build
  cd build
  ```
  
  
  
  Profiler的编译需要比较长的时间。我们可以通过以下命令来选择只编译GEMM kernel：
  
  ```shell
  cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=*gemm*  -DCUTLASS_UNITY_BUILD_ENABLED=ON 
  make cutlass_profiler -j
  ```
  
  其中，```-DCUTLASS_NVCC_ARCHS``` 可以指定要支持的最低版本架构。
  
  若要编译全部kernel，可以查看官方文档。
  
* 使用Profiler

  Profiler不支持输出为行主序的格式。所以，若要选择形状为[m, n, k]的A: row x B: row -> C: row的最佳kernel，需要跑形状为[n, m, k]的B': col x A': col -> C': col的profile。

  例如，要选择[m, n, k]=[4000, 64, 32]的A: row x B: row -> C: row的最佳kernel，可以通过如下命令(f16可以替换成其它数据格式，如f32、f64)：

  ```shell
  ./tools/profiler/cutlass_profiler --operation=gemm --m=64 --n=4000 --k=32 --A=f16:col --B=f16:col --C=f16:col --alpha=1 --beta=1 --output=tune.csv
  ```

  profile跑完后，打开输出的tune.gemm.csv文件，根据Runtime进行排序，就可以选出最快的kernel。

  | Problem | Provider | OperationKind | Operation                                        | Disposition | Status  | gemm_kind | split_k_mode | m    | n    | k    | A          | B          | C          | alpha | beta | split_k_slices | batch_count | op_class | accum | cta_m | cta_n | cta_k | stages | warps_m | warps_n | warps_k | inst_m | inst_n | inst_k | min_cc | max_cc | Bytes   | Flops    | Flops/Byte | Runtime    | GB/s    | GFLOPs  |
  | ------- | -------- | ------------- | ------------------------------------------------ | ----------- | ------- | --------- | ------------ | ---- | ---- | ---- | ---------- | ---------- | ---------- | ----- | ---- | -------------- | ----------- | -------- | ----- | ----- | ----- | ----- | ------ | ------- | ------- | ------- | ------ | ------ | ------ | ------ | ------ | ------- | -------- | ---------- | ---------- | ------- | ------- |
  | 1       | CUTLASS  | gemm          | cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8  | passed      | success | universal | serial       | 64   | 4000 | 32   | f16:column | f16:column | f16:column | 1     | 1    | 1              | 1           | tensorop | f16   | 64    | 64    | 32    | 2      | 2       | 2       | 1       | 16     | 8      | 8      | 75     | 1024   | 1284096 | 16896000 | 13         | 0.0073728  | 162.205 | 2291.67 |
  | 1       | CUTLASS  | gemm          | cutlass_tensorop_h1688gemm_64x64_32x2_nn_align4  | passed      | success | universal | serial       | 64   | 4000 | 32   | f16:column | f16:column | f16:column | 1     | 1    | 1              | 1           | tensorop | f16   | 64    | 64    | 32    | 2      | 2       | 2       | 1       | 16     | 8      | 8      | 75     | 1024   | 1284096 | 16896000 | 13         | 0.00740256 | 161.553 | 2282.45 |
  | 1       | CUTLASS  | gemm          | cutlass_tensorop_h1688gemm_64x128_32x2_nn_align8 | passed      | success | universal | serial       | 64   | 4000 | 32   | f16:column | f16:column | f16:column | 1     | 1    | 1              | 1           | tensorop | f16   | 64    | 128   | 32    | 2      | 2       | 2       | 1       | 16     | 8      | 8      | 75     | 1024   | 1284096 | 16896000 | 13         | 0.00756736 | 158.035 | 2232.75 |
  
  Operation一栏为kernel名字。我们在```cutlass/build/tools/library/generated/gemm/```目录下搜索同名文件，就可以看到最优的kernel设置。
  
  ```c++
  using cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8_base = 
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
      cutlass::half_t,//data type of A 
      cutlass::layout::RowMajor,//layout of A
      cutlass::ComplexTransform::kNone,//Enumeraed type of A
      8,//alignmentA   
      cutlass::half_t,//data type of B
      cutlass::layout::RowMajor,//layout of B
      cutlass::ComplexTransform::kNone,//Enumeraed type of B
      8,//alignmentB  
      cutlass::half_t,//data type of C
      cutlass::layout::RowMajor,//layout of C
      cutlass::half_t,//accumulation type
      cutlass::arch::OpClassTensorOp,//MMAOp
      cutlass::arch::Sm75,//CUDA SM arch
      cutlass::gemm::GemmShape<64, 64, 32>,//ThreadBlock tile size
      cutlass::gemm::GemmShape<32, 32, 32>,//Warp tile size
      cutlass::gemm::GemmShape<16, 8, 8>,//MMA OP size
      
      cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,//alignmentC
        cutlass::half_t,
        cutlass::half_t
      > //EpilogueOp
  ,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,//Stage num
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;
  ```
  
  可以发现cutlass在profiler里的列主序实际上是用行主序实现的，所以我们直接在GemmUniversal的Kernel中使用该设置即可。各设置的对应见注释。
  
  ```c++
  struct cutlass_tensorop_h1688gemm_64x64_32x2_nn_align8 {
    using Gemm = cutlass::gemm::device::GemmUniversal<
        cutlass::half_t,//data type of A
        cutlass::layout::RowMajor,//layout of A
        cutlass::half_t,//data type of B
        cutlass::layout::RowMajor,//layout of B
        cutlass::half_t,//data type of C
        cutlass::layout::RowMajor,//layout of C
        cutlass::half_t,//accumulation type
        cutlass::arch::OpClassTensorOp,//MMAOp
        cutlass::arch::Sm75,//CUDA SM arch
        cutlass::gemm::GemmShape<64, 64, 32>,//ThreadBlock tile size
        cutlass::gemm::GemmShape<32, 32, 32>,//Warp tile size
        cutlass::gemm::GemmShape<16, 8, 8>,//MMA OP size
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t,
                                                     8,//alignmentC
                                                     cutlass::half_t,
                                                     cutlass::half_t>,//EpilogueOp
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        2,//Stage num
        8,//alignmentA
        8,//alignmentB
        cutlass::arch::OpMultiplyAdd,
        cutlass::ComplexTransform::kNone,//Enumeraed type of A
        cutlass::ComplexTransform::kNone,//Enumeraed type of B
        true,//GatherA
        false,//GatherB
        true//ScatterC
        >;
  };
  ```
  
  使用kernel时有几点需要注意：
  
  1. cutlass没有为simt kernel实现gather-gemm-scatter的接口。所以搜出来的kernel中，只有op_class为tensorop的kernel才能用来做融合。
  2. 输入A:row x B: row + C:row -> D:row需要满足kernel中的对齐要求。lda要是alignmentA的倍数，ldb要是alignmentB的倍数，ldc和ldd要是alignmentC的倍数。
  3. 对于fp32和fp16，align1的kernel在计算gather-gemm-scatter时会出现地址不对齐的问题，原因未知。
  4. kernel中的```CUDA SM arch```指运行该kernel要求的最低的sm架构。对于大部分kernel，高于kernel要求的sm架构的硬件都可以正常运行。某些kernel的额外要求可以在```cutlass/tools/library/scripts/generator.py```中查看。

相关issue: https://github.com/NVIDIA/cutlass/issues/663
