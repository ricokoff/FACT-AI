# FACT-AI 2019-2020

## Repo structure
- SENN: original repo from paper's authors (plus additional commits), set up as 
`git submodule`

- this folder: additional work

## How to work with this repo
```
git clone --recurse-submodules git@github.com:ricokoff/FACT-AI.git
```

```
git pull --rebase --recurse-submodules origin <branch-name>
```

otherwise the SENN submodule doesn't get properly tracked.
