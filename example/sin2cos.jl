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
