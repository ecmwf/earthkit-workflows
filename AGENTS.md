# Overview
* utilize `just` for command running -- `just val` in backend is the "typechecking and testing". Always run this after you make any changes to python code
* project is managed by `uv` -- utilize that for running any python-related subcommands like `uv run pytest` or `uv run ty` for typechecking
* there are two python modules -- cascade which is a low level execution engine, and earthkit.workflows which is a higher level abstraction on top of it. Each has its own subdirectory in tests
* always use type annotations, it is enforced
  * when working with a package with bad typing coverage like sqlalchemy, use ty:ignore comment
  * when ty is not powerful enough, use ty:ignore 
  * use typing.cast when the code logic is implicitly erasing the type information
* prioritize using pydantic.BaseModel or dataclasses.dataclass object for capturing contracts and interfaces.
  * ideally keep them plain, stateless, frozen, without functions -- we end up serializing those objects often over to other python processes or different languages
