from cascade.fluent import Payload, Node


class MockArgs:
    def __init__(self, config_path: str):
        self.config = config_path
        self.set = None
        self.recover = False


class MockPayload(Payload):
    def __init__(self, name: str):
        super().__init__(name)


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(MockPayload(name))
