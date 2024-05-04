module ContextualAMDGPUExt

using Contextual
using Contextual: Context
using AMDGPU

@eval AMDGPU.@device_override function Contextual.withctx(ctx::Context, func, args...)
    $(Expr(:meta, :generated, Contextual.WithCtxGenerator(AMDGPU.method_table)))
    $(Expr(:meta, :generated_only))
end

end
