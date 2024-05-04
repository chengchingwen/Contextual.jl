module ContextualMetalExt

using Contextual
using Contextual: Context
using Metal

@eval Metal.@device_override function Contextual.withctx(ctx::Context, func, args...)
    $(Expr(:meta, :generated, Contextual.WithCtxGenerator(Metal.method_table)))
    $(Expr(:meta, :generated_only))
end

end
