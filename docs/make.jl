using Contextual
using Documenter

DocMeta.setdocmeta!(Contextual, :DocTestSetup, :(using Contextual); recursive=true)

makedocs(;
    modules=[Contextual],
    authors="chengchingwen <chengchingwen214@gmail.com> and contributors",
    sitename="Contextual.jl",
    format=Documenter.HTML(;
        canonical="https://chengchingwen.github.io/Contextual.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chengchingwen/Contextual.jl",
    devbranch="main",
)
