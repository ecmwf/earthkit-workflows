from typing import Iterator

from dask.distributed import Client, Variable, as_completed, get_client


def producer(n: int) -> Iterator[int]:
    for i in range(n):
        print(f"whoa just shipped one fresh {i}")
        yield i


def consumer(i: int) -> None:
    print(f"whoa what an {i} did I just receive!")


def vanilla_python(n: int) -> None:
    for i in producer(n):
        consumer(i)


def futuristic_producer(n: int) -> str:
    client = get_client()
    var = Variable("N")
    fut = client.scatter(n)
    var.set(fut)
    for i, output in enumerate(producer(n)):
        fut = client.scatter(output)
        var = Variable(f"result[{i}]")
        var.set(fut)
    return f"happily produced {n}"


def futuristic_consumer() -> str:
    # client = get_client()
    var = Variable("N")
    N = var.get().result()
    for i in range(N):
        var = Variable(f"result[{i}]")
        consumer(var.get().result())
    return f"happily consumed {N}"


if __name__ == "__main__":
    N = 5
    print("vanilla python demonstration:")
    vanilla_python(N)
    print("dask future demonstration:")
    client = Client()
    prod = client.submit(futuristic_producer, N)
    cons = client.submit(futuristic_consumer)
    print("\n".join(e.result() for e in as_completed([prod, cons])))
    del prod, cons
    for k in ["N"] + [f"result[{i}]" for i in range(N)]:
        Variable(k).delete()
