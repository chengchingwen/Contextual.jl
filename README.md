# Contextual

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chengchingwen.github.io/Contextual.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chengchingwen.github.io/Contextual.jl/dev/)
[![Build Status](https://github.com/chengchingwen/Contextual.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chengchingwen/Contextual.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/chengchingwen/Contextual.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/chengchingwen/Contextual.jl)

Exploring contextual dispatch implementation in Julia

<details>
	<summary>utils.jl</summary>

```julia
using Test
macro resultshow(a, val, ex)
	expr_str = sprint(Base.show_unquoted, ex)
	expr = Symbol(expr_str)
	linfo = findfirst("=# ", expr_str)
	if !isnothing(linfo)
		expr_str = expr_str[last(linfo)+1:end]
	end
	return quote
		@time $(esc(ex))
		print($expr_str)
		print(" = ")
		$expr = collect($(esc(a)))[]
		println($expr)
		@test $expr ≈ $val
	end
end
```

</details>

```julia
using Contextual
using Contextual: withctx, Context
using CUDA

struct Sin2Cos <: Context end

foo(x) = sin(2 * x) / 2
bar(a, x) = (a[1] = foo(x); return)
baz(a, x) = (withctx(Sin2Cos(), bar, a, x); return)
qux(a, x) = (a[1] = withctx(Sin2Cos(), sin, x); return)

a_cpu = Float32[0]
a_gpu = cu(a_cpu)

println("\nbefore:")
@resultshow a_cpu sin(2 * 0.7) / 2 withctx(Sin2Cos(), bar, a_cpu, 0.7)
@resultshow a_gpu sin(2 * 0.7) / 2 @cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7))
@resultshow a_cpu sin(2 * 0.7) / 2 baz(a_cpu, 0.7)
@resultshow a_gpu sin(2 * 0.7) / 2 @cuda(baz(a_gpu, 0.7))
@resultshow a_cpu sin(0.3) qux(a_cpu, 0.3)
@resultshow a_gpu sin(0.3) @cuda(qux(a_gpu, 0.3))

Contextual.ctxcall(::Sin2Cos, ::typeof(sin), x) = cos(x)

println("\nafter:")
@resultshow a_cpu cos(2 * 0.7) / 2 withctx(Sin2Cos(), bar, a_cpu, 0.7)
@resultshow a_gpu cos(2 * 0.7) / 2 @cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7))
@resultshow a_cpu cos(2 * 0.7) / 2 baz(a_cpu, 0.7)
@resultshow a_gpu cos(2 * 0.7) / 2 @cuda(baz(a_gpu, 0.7))
@resultshow a_cpu cos(0.3) qux(a_cpu, 0.3)
@resultshow a_gpu cos(0.3) @cuda(qux(a_gpu, 0.3))

Contextual.ctxcall(::Sin2Cos, ::typeof(sin), x) = tan(x)

println("\nredefine:")
@resultshow a_cpu tan(2 * 0.7) / 2 withctx(Sin2Cos(), bar, a_cpu, 0.7)
@resultshow a_gpu tan(2 * 0.7) / 2 @cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7))
@resultshow a_cpu tan(2 * 0.7) / 2 baz(a_cpu, 0.7)
@resultshow a_gpu tan(2 * 0.7) / 2 @cuda(baz(a_gpu, 0.7))
@resultshow a_cpu tan(0.3) qux(a_cpu, 0.3)
@resultshow a_gpu tan(0.3) @cuda(qux(a_gpu, 0.3))

Base.delete_method(methods(Contextual.ctxcall, Tuple{Sin2Cos, typeof(sin), Any})[1])

println("\ndelete:")
@resultshow a_cpu sin(2 * 0.7) / 2 withctx(Sin2Cos(), bar, a_cpu, 0.7)
@resultshow a_gpu sin(2 * 0.7) / 2 @cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7))
@resultshow a_cpu sin(2 * 0.7) / 2 baz(a_cpu, 0.7)
@resultshow a_gpu sin(2 * 0.7) / 2 @cuda(baz(a_gpu, 0.7))
@resultshow a_cpu sin(0.3) qux(a_cpu, 0.3)
@resultshow a_gpu sin(0.3) @cuda(qux(a_gpu, 0.3))

Contextual.ctxcall(ctx::Sin2Cos, ::typeof(foo), x) = sin(x) + withctx(ctx, cos, x)
Contextual.ctxcall(ctx::Sin2Cos, ::typeof(cos), x) = tan(x)

println("\nafter2:")
@resultshow a_cpu sin(0.7) + tan(0.7) withctx(Sin2Cos(), bar, a_cpu, 0.7)
@resultshow a_gpu sin(0.7) + tan(0.7) @cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7))
@resultshow a_cpu sin(0.7) + tan(0.7) baz(a_cpu, 0.7)
@resultshow a_gpu sin(0.7) + tan(0.7) @cuda(baz(a_gpu, 0.7))
@resultshow a_cpu sin(0.3) qux(a_cpu, 0.3)
@resultshow a_gpu sin(0.3) @cuda(qux(a_gpu, 0.3))

Contextual.ctxcall(ctx::Sin2Cos, ::typeof(foo), x) = sin(x) + cos(x)

println("\nredefine2:")
@resultshow a_cpu sin(0.7) + cos(0.7) withctx(Sin2Cos(), bar, a_cpu, 0.7)
@resultshow a_gpu sin(0.7) + cos(0.7) @cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7))
@resultshow a_cpu sin(0.7) + cos(0.7) baz(a_cpu, 0.7)
@resultshow a_gpu sin(0.7) + cos(0.7) @cuda(baz(a_gpu, 0.7))
@resultshow a_cpu sin(0.3) qux(a_cpu, 0.3)
@resultshow a_gpu sin(0.3) @cuda(qux(a_gpu, 0.3))

#=
before:
  2.349147 seconds (3.09 M allocations: 198.860 MiB, 2.52% gc time, 100.00% compilation time)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 0.49272487
 11.416841 seconds (10.49 M allocations: 722.071 MiB, 3.19% gc time, 97.99% compilation time: 3% of which was recompilatio
n)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 0.49272487
  0.004102 seconds (435 allocations: 30.594 KiB, 99.41% compilation time)
baz(a_cpu, 0.7) = 0.49272487
  0.618518 seconds (411.41 k allocations: 29.702 MiB, 1.72% gc time, 64.10% compilation time)
@cuda(baz(a_gpu, 0.7)) = 0.49272487
  0.059568 seconds (10.79 k allocations: 403.672 KiB, 99.96% compilation time)
qux(a_cpu, 0.3) = 0.29552022
  0.372148 seconds (120.18 k allocations: 9.514 MiB, 42.41% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.29552022

after:
  0.007325 seconds (7.55 k allocations: 462.016 KiB, 99.86% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 0.08498357
  0.225906 seconds (69.83 k allocations: 5.719 MiB, 5.88% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 0.08498357
  0.005372 seconds (781 allocations: 55.516 KiB, 99.79% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 0.08498357
  0.222038 seconds (27.66 k allocations: 3.316 MiB, 2.12% compilation time)
@cuda(baz(a_gpu, 0.7)) = 0.08498357
  0.005124 seconds (693 allocations: 47.984 KiB, 99.80% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.9553365
  0.242207 seconds (15.27 k allocations: 2.534 MiB, 11.56% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.9553365

redefine:
  0.007298 seconds (7.55 k allocations: 462.016 KiB, 99.86% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 2.8989418
  0.244230 seconds (84.54 k allocations: 6.736 MiB, 11.34% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 2.8989418
  0.005374 seconds (780 allocations: 54.062 KiB, 99.84% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 2.8989418
  0.218063 seconds (27.89 k allocations: 3.334 MiB, 2.13% compilation time)
@cuda(baz(a_gpu, 0.7)) = 2.8989418
  0.005136 seconds (693 allocations: 47.984 KiB, 99.78% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.30933625
  0.216940 seconds (14.48 k allocations: 2.507 MiB, 0.52% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.30933625

delete:
  0.086327 seconds (107.95 k allocations: 5.606 MiB, 99.99% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 0.49272487
  0.227787 seconds (68.88 k allocations: 5.666 MiB, 5.80% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 0.49272487
  0.003912 seconds (423 allocations: 29.562 KiB, 99.71% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 0.49272487
  0.246610 seconds (27.47 k allocations: 3.310 MiB, 13.40% compilation time)
@cuda(baz(a_gpu, 0.7)) = 0.49272487
  0.059131 seconds (10.08 k allocations: 353.953 KiB, 99.98% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.29552022
  0.216110 seconds (13.94 k allocations: 2.481 MiB, 0.53% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.29552022

after2:
  0.006924 seconds (5.73 k allocations: 359.016 KiB, 99.84% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 1.4865061
  0.260920 seconds (54.43 k allocations: 5.056 MiB, 14.11% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 1.4865061
  0.005430 seconds (653 allocations: 45.922 KiB, 99.79% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 1.4865061
  0.228406 seconds (29.80 k allocations: 3.417 MiB, 2.01% compilation time)
@cuda(baz(a_gpu, 0.7)) = 1.4865061
  0.000006 seconds
qux(a_cpu, 0.3) = 0.29552022
  0.000075 seconds (17 allocations: 1.078 KiB)
@cuda(qux(a_gpu, 0.3)) = 0.29552022

redefine2:
  0.006634 seconds (4.53 k allocations: 282.359 KiB, 99.82% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 1.4090599
  0.228165 seconds (27.96 k allocations: 3.323 MiB, 1.93% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 1.4090599
  0.005365 seconds (649 allocations: 45.703 KiB, 99.79% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 1.4090599
  0.230428 seconds (29.13 k allocations: 3.392 MiB, 1.99% compilation time)
@cuda(baz(a_gpu, 0.7)) = 1.4090599
  0.000005 seconds
qux(a_cpu, 0.3) = 0.29552022
  0.000075 seconds (17 allocations: 1.078 KiB)
@cuda(qux(a_gpu, 0.3)) = 0.29552022
=#
```
