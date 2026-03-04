In the src/cascade module, we `raise` exceptions at many places. They are of the following categories:
 - code has reached an unexpected state, such as encountering unexpected enum value, internal invariant not upheld, etc. This suggests a programmer error. Neither re-run, nor change of user config, are likely to help, and instead upgrade of the cascade library must happen
 - underlying infrastructure problem -- os exceptions, network exceptions, lack of memory in shm, etc. This suggests that a re-run, perhaps with a different configuration (allocated memory, chunking size, ...), may help
 - user code problem -- cascade is a runner of user-provided functions, and we may encounter issues originating from that code. Errors of this kind suggest the user must change their code or configuration first

Create a new module cascade.low.exceptions, and introduce three new classes corresponding to the three kinds of exceptions above:
 - CascadeInfrastructureError
 - CascadeInternalError
 - CascadeUserError

Each should be design as follows (use a parent class, with the three as children with `pass` as the body):
 - parent: Exception|None -- sometimes this wraps a parent exception (typically in the infrastructure and user cases, but possible in internal too)
 - description: str
 - possibly override __repr__ and __str__ -- we want those to give full context

Make an educated guess for each existing raise of which class to replace with. If we are catching an existing exception, wrap it as a parent, and put to description the `repr` of it

Be mindful that we often do `try: ...; except Exception as e: logger.exception; raise` -- there you need to wrap.
When we just `raise ValueError` or `raise TypeError`, no need to wrap, just use the cascade error directly, with description being the original text without repr.
Similarly KeyError and the like need not be wrapped.
Consider modifying the logger statements, with like 'logger.exception("failed during xxx, propagating")' -> 'logger.exception("failed during xxx, propagating as InfrastructureError")'.
