from pydantic import BaseModel

from cascade.low.func import pydantic_recursive_collect


def test_pydantic_recursive_collect() -> None:

    class DaModel(BaseModel):
        a: int

        def visit(self) -> list[str]:
            if self.a % 2 == 1:
                return [f"a={self.a} is quite odd"]
            else:
                return []

    class DaBase(BaseModel):
        da_model: DaModel
        nested: list[DaModel]

    models = {
        "first": [
            DaBase(
                da_model=DaModel(a=1),
                nested=[
                    DaModel(a=2),
                    DaModel(a=3),
                ],
            ),
            DaBase(
                da_model=DaModel(a=4),
                nested=[],
            ),
        ],
        "second": DaBase(
            da_model=DaModel(a=5),
            nested=[
                DaModel(a=6),
                DaModel(a=7),
            ],
        ),
    }

    expected = [
        (".first.[0].da_model.", "a=1 is quite odd"),
        (".first.[0].nested.[1].", "a=3 is quite odd"),
        (".second.da_model.", "a=5 is quite odd"),
        (".second.nested.[1].", "a=7 is quite odd"),
    ]

    result = pydantic_recursive_collect(models, "visit")
    assert result == expected
