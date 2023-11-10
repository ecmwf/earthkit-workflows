import inspect
import importlib


class PatchModule:
    def __init__(self, module, original, patch):
        self.module = module
        self.original = importlib.import_module(original)
        self.patch = patch

    def __enter__(self):
        if self.original == self.patch:
            return

        functions = []
        module_key = None
        for key, value in self.module.__dict__.items():
            if value == self.original:
                module_key = key
            elif inspect.isfunction(value):
                functions.append(key)
        assert module_key is not None

        setattr(self.module, module_key, self.patch)
        for func in functions:
            func_source = inspect.getsource(getattr(self.module, func))
            exec(func_source, self.module.__dict__)

    def __exit__(self, type, value, traceback):
        if self.original == self.patch:
            return
        importlib.reload(self.module)
