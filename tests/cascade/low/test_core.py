from cascade.low.core import DatasetId


def test_datasetid_serde():
    cases = [
        DatasetId(task="basic", output="0"),
        DatasetId(
            task="whoa_tricky!()(??!@#!$34--thisWouldBeABadFileNameReally\n\n\1\0\t",
            output="thiscanbetrickytoo",
        ),
    ]

    for case in cases:
        assert case == DatasetId.des(case.ser())
