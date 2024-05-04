module ContextualCUDAExt

using Contextual
using Contextual: Context
using CUDA

@eval CUDA.@device_override function Contextual.withctx(ctx::Context, func, args...)
    $(Expr(:meta, :generated, Contextual.WithCtxGenerator(CUDA.method_table)))
    $(Expr(:meta, :generated_only))
end

end
