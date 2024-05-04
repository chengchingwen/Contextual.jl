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
  3.992558 seconds (4.77 M allocations: 298.112 MiB, 3.12% gc time, 100.00% compilation time)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 0.49272487
 11.130603 seconds (10.59 M allocations: 725.977 MiB, 1.70% gc time, 97.90% compilation time: 3% of which was recompilatio
n)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 0.49272487
  0.063235 seconds (36.53 k allocations: 1.531 MiB, 99.96% compilation time)
baz(a_cpu, 0.7) = 0.49272487
  0.629113 seconds (421.50 k allocations: 30.348 MiB, 4.26% gc time, 64.62% compilation time)
@cuda(baz(a_gpu, 0.7)) = 0.49272487
  0.063256 seconds (36.41 k allocations: 1.488 MiB, 99.96% compilation time)
qux(a_cpu, 0.3) = 0.29552022
  0.370754 seconds (120.04 k allocations: 9.468 MiB, 42.38% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.29552022

after:
  0.095906 seconds (104.55 k allocations: 5.574 MiB, 99.99% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 0.08498357
  0.439240 seconds (255.60 k allocations: 15.448 MiB, 50.76% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 0.08498357
  0.004826 seconds (996 allocations: 71.789 KiB, 99.72% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 0.08498357
  0.225488 seconds (36.04 k allocations: 3.903 MiB, 4.71% compilation time)
@cuda(baz(a_gpu, 0.7)) = 0.08498357
  0.005709 seconds (1.52 k allocations: 105.492 KiB, 99.80% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.9553365
  0.240011 seconds (17.92 k allocations: 2.688 MiB, 12.24% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.9553365

redefine:
  0.011297 seconds (9.79 k allocations: 591.172 KiB, 99.91% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 2.8989418
  0.420425 seconds (145.87 k allocations: 9.804 MiB, 20.62% gc time, 28.23% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 2.8989418
  0.004710 seconds (996 allocations: 71.789 KiB, 99.74% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 2.8989418
  0.221795 seconds (36.39 k allocations: 3.933 MiB, 4.78% compilation time)
@cuda(baz(a_gpu, 0.7)) = 2.8989418
  0.005748 seconds (1.52 k allocations: 105.492 KiB, 99.81% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.30933625
  0.213405 seconds (17.26 k allocations: 2.667 MiB, 1.15% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.30933625

delete:
  1.689796 seconds (1.60 M allocations: 79.711 MiB, 1.07% gc time, 100.00% compilation time: 100% of which was recompilati
on)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 0.49272487
  0.342328 seconds (134.11 k allocations: 8.924 MiB, 31.15% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 0.49272487
  0.063205 seconds (36.51 k allocations: 1.528 MiB, 99.98% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 0.49272487
  0.251018 seconds (37.56 k allocations: 3.951 MiB, 15.51% compilation time)
@cuda(baz(a_gpu, 0.7)) = 0.49272487
  0.063037 seconds (35.69 k allocations: 1.438 MiB, 99.98% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.29552022
  0.215557 seconds (15.10 k allocations: 2.522 MiB, 0.54% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.29552022

after2:
  0.038593 seconds (27.81 k allocations: 1.619 MiB, 99.97% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 1.4865061
  0.277985 seconds (81.35 k allocations: 6.785 MiB, 20.48% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 1.4865061
  0.004571 seconds (684 allocations: 47.664 KiB, 99.71% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 1.4865061
  0.235228 seconds (37.93 k allocations: 3.981 MiB, 4.37% compilation time)
@cuda(baz(a_gpu, 0.7)) = 1.4865061
  1.289852 seconds (1.27 M allocations: 65.629 MiB, 1.45% gc time, 100.00% compilation time: 100% of which was recompilation)
qux(a_cpu, 0.3) = 0.29552022
  0.216188 seconds (17.42 k allocations: 2.675 MiB, 1.16% compilation time)
@cuda(qux(a_gpu, 0.3)) = 0.29552022

redefine2:
  0.011833 seconds (6.44 k allocations: 377.234 KiB, 99.91% compilation time: 100% of which was recompilation)
withctx(Sin2Cos(), bar, a_cpu, 0.7) = 1.4090599
  0.236598 seconds (35.64 k allocations: 3.883 MiB, 4.20% compilation time)
@cuda(withctx(Sin2Cos(), bar, a_gpu, 0.7)) = 1.4090599
  0.004592 seconds (674 allocations: 47.070 KiB, 99.67% compilation time: 100% of which was recompilation)
baz(a_cpu, 0.7) = 1.4090599
  0.235683 seconds (36.73 k allocations: 3.938 MiB, 4.35% compilation time)
@cuda(baz(a_gpu, 0.7)) = 1.4090599
  0.000006 seconds
qux(a_cpu, 0.3) = 0.29552022
  0.000071 seconds (17 allocations: 1.078 KiB)
@cuda(qux(a_gpu, 0.3)) = 0.29552022
=#
```
