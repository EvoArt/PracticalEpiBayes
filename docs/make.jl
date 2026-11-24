using Documenter, PracticalEpiBayes

makedocs(
    sitename = "PracticalEpiBayes.jl",
    modules = [PracticalEpiBayes],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = ["Home" => "index.md", "API" => "api.md"],
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(repo = "github.com/EvoArt/PracticalEpiBayes.git", devbranch = "master")
