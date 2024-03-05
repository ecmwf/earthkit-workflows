import functools


def num_args(expect: int, accept_nested: bool = True):
    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def check_num_args(*args, **kwargs):
            if accept_nested and len(args) == 1:
                args = args[0]
            assert (
                len(args) == expect
            ), f"{func.__name__} expects two input arguments, got {len(args)}"
            return func(*args, **kwargs)

        return check_num_args

    return decorator


def batchable(func: callable) -> callable:
    """
    Decorator to mark a function as batchable. A method is batchable if
    it can be computed sequentially trivially by applying the same function
    to each batch and to aggregate the batches. Examples of batchable
    functions are sum, prod, min and non-batchable are mean and std.
    """
    func.batchable = True
    return func
