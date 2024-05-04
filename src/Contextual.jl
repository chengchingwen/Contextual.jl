module Contextual

using Core: CodeInfo, MethodTable, MethodInstance, SSAValue, SlotNumber, NewvarNode, ReturnNode, GotoNode, GotoIfNot
using Base.Meta: isexpr

create_codeinfo(argnames, body; kws...) = create_codeinfo(argnames, nothing, body; kws...)
create_codeinfo(mod::Module, argnames, body; kws...) = create_codeinfo(mod, argnames, nothing, body; kws...)
create_codeinfo(argnames, spnames, body; kws...) = create_codeinfo(@__MODULE__, argnames, spnames, body; kws...)
function create_codeinfo(mod::Module, argnames, spnames, body; inline = false)
    # argnames: `Vector{Symbol}` representing the variable names, starts with `Symbol("#self#")`.
    # spnames: the variable names in `where {...}`
    @assert isexpr(body, :block) "body should be `Expr(:block, ...)`."
    if inline # insert inline tag to body
        body = Expr(:block, Expr(:meta, :inline), body.args...)
    end
    expr = Expr(:lambda, argnames, Expr(Symbol("scope-block"), body))
    if !isnothing(spnames)
        expr = Expr(Symbol("with-static-parameters"), expr, spnames...)
    end
    ci = ccall(:jl_expand, Any, (Any, Any), expr, mod) # expand macrocall and return code_info
    ci.inlineable = true
    return ci
end

walk(fn, x, guard) = fn(x)
walk(fn, x::SSAValue, guard) = fn(x)
walk(fn, x::SlotNumber, guard) = fn(x)
walk(fn, x::NewvarNode, guard) = NewvarNode(walk(fn, x.slot, guard))
walk(fn, x::ReturnNode, guard) = ReturnNode(walk(fn, x.val, guard))
walk(fn, x::GotoNode, guard) = fn(x)
walk(fn, x::GotoIfNot, guard) = fn(x)
walk(fn, x::Expr, guard) = Expr(x.head, walk(fn, x.args, guard)...)
walk(fn, x::Vector, guard) = map(el -> walk(fn, el, guard), x)
walk(fn, x) = walk(fn, x, Val(nothing))

resolve(x) = x
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)

function _copy_ci_fields!(new_ci, ci)
    for name in propertynames(ci)
        name in (:code, :slotnames, :slotflags, :slottypes,
                 :codelocs, :ssavaluetypes, :ssaflags,
                 :method_for_inference_limit_heuristics, :inlineable, :inlining) && continue
        setproperty!(new_ci, name, getproperty(ci, name))
    end
    return new_ci
end

function _lookup_method(@nospecialize(fsig::Type), @nospecialize(mt::Union{Nothing, MethodTable}), world)
    matches = Base._methods_by_ftype(fsig, mt, -1, world)
    return !isnothing(matches) && !isempty(matches) ? only(matches) : nothing
end
function _lookup_method(@nospecialize(fsig::Type), @nospecialize(method_tables::Vector{MethodTable}), world)
    for mt in method_tables
        matches = Base._methods_by_ftype(fsig, mt, -1, world)
        !isnothing(matches) && !isempty(matches) && return only(matches)
    end
    return nothing
end
function methods_by_ftype(@nospecialize(fsig::Type), @nospecialize(mt::Union{MethodTable, Vector{MethodTable}}), world)
    meth = _lookup_method(fsig, mt, world)
    return isnothing(meth) ? only(Base._methods_by_ftype(fsig, -1, world)) : meth
end
methods_by_ftype(@nospecialize(fsig::Type), @nospecialize(::Nothing), world) = only(Base._methods_by_ftype(fsig, -1, world))

struct CallerCallback
    caller::MethodInstance
end
function (callback::CallerCallback)(replaced::MethodInstance, max_world::UInt32,
                                    seen::Set{MethodInstance}=Set{MethodInstance}())
    push!(seen, replaced)
    # run callback of caller
    caller = callback.caller
    if isdefined(caller, :callbacks)
        for cb in caller.callbacks
            cb(caller, max_world, seen)
        end
    end
    return
end

# take a function signature `fsig` and world age `world`,
# transform into codeinfo of a new function whose function
#  arguments are specified by `(fargs..., func, args)`
#  where `func` is the symbol for origin function and
#  `args` is the vararg passing to the origin function.
struct FuncTransform
    sig::Any
    meth::Method
    mi::MethodInstance
    fargs::Vector{Symbol}
    func::Symbol
    args::Symbol
    codeinfo::CodeInfo
    argssa::Int
    function FuncTransform(
        @nospecialize(fsig), fargs::Vector{Symbol}, func::Symbol, args::Symbol, world;
        inline = false, caller::Union{MethodInstance, Nothing} = nothing,
        method_tables::Union{Nothing, MethodTable, Vector{MethodTable}} = nothing
    )
        match = methods_by_ftype(fsig, method_tables, world)
        meth = match.method
        instance = Core.Compiler.specialize_method(match)
        if Base.hasgenerator(meth)
            if Base.may_invoke_generator(instance)
                ci = ccall(:jl_code_for_staged, Any, (Any, UInt), instance, world)::CodeInfo
            else
                error("Could not expand generator for `@generated` method ", instance)
            end
        else
            ci = Base.uncompressed_ir(meth)
        end
        Meta.partially_inline!(ci.code, Any[], match.spec_types, Any[match.sparams...], 0, 0, :off)
        nfargs = meth.nargs - 1
        nargs = length(fsig.parameters) - 1
        new_fargs = [fargs..., func, args]
        n = length(new_fargs)
        # create empty codeinfo and reassign slots
        new_ci = create_codeinfo(new_fargs, Expr(:block); inline)
        _copy_ci_fields!(new_ci, ci)
        new_ci.slotnames = Any[new_fargs..., Iterators.drop(ci.slotnames, 1)...]
        new_ci.slotflags = UInt8[
            0x00, Iterators.repeated(0x08, n - 3)...,
            first(ci.slotflags), 0x08,
            (flag | 0x08 for flag in Iterators.take(Iterators.drop(ci.slotflags, 1), nfargs))...,
            Iterators.drop(ci.slotflags, nfargs + 1)...
        ]
        # make assignment for arguments of the origin function
        funcslot = SlotNumber(n - 1)
        argsslot = SlotNumber(n)
        code = Any[]
        codelocs = similar(ci.codelocs, 0)
        ssaflags = UInt8[]
        hasvararg = meth.isva
        if nargs == nfargs
            if hasvararg
                for i = 1:nargs-1
                    arg = SlotNumber(i + n)
                    v = Expr(:call, GlobalRef(Base, :getfield), argsslot, i, true)
                    push!(code, Expr(:(=), arg, v))
                end
                push!(code, Expr(:call, GlobalRef(Base, :getfield), argsslot, nargs, true))
                push!(code, Expr(:(=), SlotNumber(nargs + n), Expr(:call, GlobalRef(Core, :tuple), SSAValue(nargs))))
            else
                for i = 1:nargs
                    arg = SlotNumber(i + n)
                    v = Expr(:call, GlobalRef(Base, :getfield), argsslot, i, true)
                    push!(code, Expr(:(=), arg, v))
                end
            end
        elseif nargs < nfargs
            for i = 1:nargs
                arg = SlotNumber(i + n)
                v = Expr(:call, GlobalRef(Base, :getfield), argsslot, i, true)
                push!(code, Expr(:(=), arg, v))
            end
            @assert nargs + 1 == nfargs
            arg = SlotNumber(nargs + n + 1)
            push!(code, Expr(:(=), arg, ()))
        else
            for i = 1:nfargs - 1
                arg = SlotNumber(i + n)
                v = Expr(:call, GlobalRef(Base, :getfield), argsslot, i, true)
                push!(code, Expr(:(=), arg, v))
            end
            rest = nargs - nfargs + 1
            for i = 1:rest
                v = Expr(:call, GlobalRef(Base, :getfield), argsslot, nfargs + i - 1, true)
                push!(code, v)
            end
            arg = SlotNumber(nfargs + n)
            v = Expr(:call, GlobalRef(Core, :tuple),
                     (SSAValue(nfargs + j - 1) for j = 1:rest)...)
            push!(code, Expr(:(=), arg, v))
        end
        ssavaluetypes = length(code)
        append!(codelocs, Iterators.repeated(0, ssavaluetypes))
        append!(ssaflags, Iterators.repeated(0x0, ssavaluetypes))
        for (stmt, loc, flag) in zip(ci.code, ci.codelocs, ci.ssaflags)
            stmt = walk(stmt) do x
                if x isa SlotNumber
                    id = x.id == 1 ? n - 1 : x.id + n - 1
                    return SlotNumber(id)
                elseif x isa SSAValue
                    return SSAValue(x.id + ssavaluetypes)
                elseif x isa GotoNode
                    label = walk(var"#self#", SSAValue(x.label)).id
                    return GotoNode(label)
                elseif x isa GotoIfNot
                    cond = walk(var"#self#", x.cond)
                    dest = walk(var"#self#", SSAValue(x.dest)).id
                    return GotoIfNot(cond, dest)
                else
                    return x
                end
            end
            push!(code, stmt)
            push!(codelocs, loc)
            push!(ssaflags, flag)
        end
        new_ci.code = code
        new_ci.codelocs = codelocs
        new_ci.ssaflags = ssaflags
        new_ci.ssavaluetypes = length(code)
        new_ci.method_for_inference_limit_heuristics = meth
        if !isnothing(caller)
            # `func` is not directly called, so we need to set the backedges manually so that change of `func`
            #  trigger recompilation. However, GPUCompiler use a callback for invalidations, but the callback
            #  might not be set for `func`, so we add our callback to trigger the callback of the caller.
            ccall(:jl_method_instance_add_backedge, Cvoid, (Any, Any, Any), instance, nothing, caller)
            if isdefined(instance, :callbacks)
                push!(instance.callbacks, CallerCallback(caller))
            else
                instance.callbacks = Any[CallerCallback(caller)]
            end
        end
        return new(fsig, meth, instance, fargs, func, args, new_ci, ssavaluetypes)
    end
end

struct WithCtxGenerator{MT<:Union{Nothing, MethodTable, Vector{MethodTable}}}
    overlay_tables::MT
end

function (g::WithCtxGenerator)(world, source, self, ctx, func, args)
    fsig = Base.to_tuple_type((func, args...))
    overlay = isnothing(g.overlay_tables) ? false : !isnothing(_lookup_method(fsig, g.overlay_tables, world))
    dispatchsig = Base.to_tuple_type((typeof(ctxcall), ctx, func, args...))
    ctxdispatchmeth = _lookup_method(dispatchsig, nothing, world)
    hasctxdispatch = !isnothing(ctxdispatchmeth)
    if hasctxdispatch || overlay
        ci = create_codeinfo(
            [:withctx, :ctx, :func, :args],
            Expr(:block, Expr(:return, Expr(:call, __ctxcall__, Val(overlay), :withctx, ctxcall, :ctx, :func,
                                            (:(args[$i]) for i in 1:length(args))...)));
            inline = true)
    else
        caller = Core.Compiler.specialize_method(methods_by_ftype(Tuple{self, ctx, func, args...}, nothing, world))
        ctxvar = gensym(:__ctx__)
        funvar = gensym(:__func__)
        argvar = gensym(:__args__)
        ft = FuncTransform(
            fsig, [gensym(:withctx), ctxvar], funvar, argvar, world; caller, inline = true)
        withslot = SlotNumber(1)
        ctxslot = SlotNumber(2)
        for (ssavalue, st) in enumerate(ft.codeinfo.code)
            if ssavalue > ft.argssa && isexpr(st, :call)
                callee = resolve(first(st.args))
                new_st = Expr(:call, __ctxcall__, Val(false), withslot, ctxcall, ctxslot, st.args...)
                ft.codeinfo.code[ssavalue] = new_st
                ft.codeinfo.ssaflags[ssavalue] |= 0x01
            end
        end
        ci = ft.codeinfo
    end
    return ci
end

function __ctxcall_generator__(world, source, self, overlay, with, call, ctx_func_args)
    isoverlay = overlay.parameters[1]
    ctx, func, args... = ctx_func_args
    fsig = Base.to_tuple_type((typeof(ctxcall), ctx, func, args...))
    noctxdispatch = isnothing(_lookup_method(fsig, nothing, world))
    if noctxdispatch
        if isoverlay
            ci = create_codeinfo(
                [:__ctxcall__, :__overlay__, :withctx, :ctxcall, :ctx_func_args],
                Expr(:block, Expr(:return,
                                  Expr(:call,
                                       Expr(:call, first, Expr(:call, Base.tail, :ctx_func_args)),
                                       Expr(:..., Expr(:call, Base.tail, Expr(:call, Base.tail, :ctx_func_args))))));
                inline = true)
        else
            ci = create_codeinfo(
                [:__ctxcall__, :__overlay__, :withctx, :ctxcall, :ctx_func_args],
                Expr(:block, Expr(:return, Expr(:call, :withctx, Expr(:..., :ctx_func_args))));
                inline = true)
        end
    else
        caller = Core.Compiler.specialize_method(methods_by_ftype(Tuple{self, overlay, with, call, ctx_func_args...}, nothing, world))
        ft = FuncTransform(
            fsig, [gensym(:__ctxcall__), gensym(:__overlay__), gensym(:withctx)], :ctxcall, gensym(:__args__), world;
            caller, inline = true)
        withslot = SlotNumber(3)
        ctxslot = SlotNumber(5)
        funcslot = SlotNumber(6)
        for (ssavalue, st) in enumerate(ft.codeinfo.code)
            if ssavalue > ft.argssa
                st = walk(st) do x
                    resolve(x) isa typeof(withctx) ? withslot : x
                end
                if isexpr(st, :call)
                    callee = first(st.args)
                    if callee == withslot
                        ft.codeinfo.ssaflags[ssavalue] |= 0x01
                    end
                end
                ft.codeinfo.code[ssavalue] = st
            end
        end
        ci = ft.codeinfo
    end
    mt_edges = Core.Compiler.vect(typeof(ctxcall).name.mt, Tuple{typeof(ctxcall), ctx, Vararg{Any}})
    ci.edges = mt_edges
    return ci
end

@eval function __ctxcall__(overlay::Val, with, call, ctx_func_args...)
    $(Expr(:meta, :generated, __ctxcall_generator__))
    $(Expr(:meta, :generated_only))
end

abstract type Context end

# for dispatch only, would never be called directly
# behavior default to `withctx(ctx, func, args...)`
ctxcall(ctx::Context, func::Core.Builtin, args...) = func(args...)
ctxcall(ctx::Context, func::typeof(Core._apply_iterate), iter, f, args...) = Core._apply_iterate(iter, withctx, (ctx, f), args...)

@eval function withctx(ctx::Context, func, args...)
    $(Expr(:meta, :generated, WithCtxGenerator(nothing)))
    $(Expr(:meta, :generated_only))
end

end
