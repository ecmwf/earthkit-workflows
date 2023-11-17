from datetime import datetime

from cascade.fluent import Payload, Node


class MockArgs:
    def __init__(self, config_path: str):
        self.config = config_path
        self.set = None
        self.recover = False
        self.override_input = []
        self.override_output = []


class MockClusterArgs(MockArgs):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.date = datetime.fromisoformat("2023-11-01")
        self.spread = None
        self.pca = None
        self.spread_compute = [
            "fdb:spread_z500",
            "fileset:spread_z500",
            "mars:spread_z500",
            "fdb:spread_z500_0001",
        ]
        self.ensemble = "fdb:ens_z500"
        self.deterministic = "fdb:determ_z500"
        self.clim_dir = ""
        self.ncomp_file = ""
        self.centroids = ""
        self.representative = ""
        self.output_root = ""
        self.cen_anomalies = ""
        self.rep_anomalies = ""
        self.mask = None
        self.indexes = None


class MockPayload(Payload):
    def __init__(self, name: str):
        super().__init__(name)


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(MockPayload(name))
