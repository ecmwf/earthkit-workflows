from cascade.cascade import Cascade
from cascade.graphs import ContextGraph, TaskGraph
from cascade.scheduler import AnnealingScheduler


class TestDepthFirstScheduler:

    def setup_method(self, method):
        
        self.G = TaskGraph()

        # Create 10 output nodes
        for i in range(10):
            self.G.add_task(cost=100, in_memory=1, out_memory=0, name=f"output_{i}")

        # Create 50 reader nodes, each with an additional processing node
        for i in range(50):
            self.G.add_task(cost=100, in_memory=0, out_memory=1, name=f"read_{i}")
            self.G.add_task(cost=100, in_memory=1, out_memory=1, name=f"sh2gp_{i}")
            self.G.add_comm_edge(f"read_{i}", f"sh2gp_{i}", size=0.5)

        # Create processes which require several of the inputs to produce the outputs
        for i in range(10):
            self.G.add_task(cost=30, in_memory=7, out_memory=1, name=f"mean_{i}")
            for j in range(7):
                self.G.add_comm_edge(f"sh2gp_{min(i*5+j,49)}", f"mean_{i}", size=1)
            self.G.add_comm_edge(f"mean_{i}", f"output_{i}", size=1)

        # Create a process which requires all of the means to produce the output
        self.G.add_task(cost=30, in_memory=10, out_memory=1, name="mean_all")
        for i in range(10):
            self.G.add_comm_edge(f"mean_{i}", "mean_all", size=1)
        self.G.add_task(cost=30, in_memory=1, out_memory=1, name="output_all")
        self.G.add_comm_edge("mean_all", "output_all", size=1)

        self.G.draw("test_depth_first.png")

        self.contexts = ContextGraph()
        self.contexts.add_node("gpu_1", type="GPU", speed=10, memory=10)
        self.contexts.add_node("gpu_2", type="GPU", speed=10, memory=10)
        self.contexts.add_node("gpu_3", type="GPU", speed=5, memory=10)
        self.contexts.add_node("gpu_4", type="GPU", speed=5, memory=10)
        self.contexts.add_edge("gpu_1", "gpu_2", bandwidth=0.1, latency=1)
        self.contexts.add_edge("gpu_1", "gpu_3", bandwidth=0.02, latency=3)
        self.contexts.add_edge("gpu_1", "gpu_4", bandwidth=0.02, latency=3)
        self.contexts.add_edge("gpu_2", "gpu_3", bandwidth=0.02, latency=3)
        self.contexts.add_edge("gpu_2", "gpu_4", bandwidth=0.02, latency=3)
        self.contexts.add_edge("gpu_3", "gpu_4", bandwidth=0.1, latency=1)

    def test_depth_first_scheduler(self):
        schedule = Cascade.create_schedule(self.G, self.contexts)
        print(schedule)

        execution = Cascade.simulate(schedule)
        print(execution)

    def test_annealing_scheduler(self):
        schedule = AnnealingScheduler(self.G, self.contexts).create_schedule()
        execution = Cascade.simulate(schedule)
        print(f"With Communications:", execution)

if __name__ == "__main__":
    t = TestDepthFirstScheduler()
    t.setup_method(None)
    t.test_depth_first_scheduler()
    t.test_annealing_scheduler()