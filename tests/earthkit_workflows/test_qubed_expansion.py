# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from qubed import Qube  # type: ignore

from earthkit.workflows._qubed import _convert_num_to_abc, expand_as_qube
from earthkit.workflows.fluent import from_source

SOURCE_ACTION = from_source("test", datacubes={"dim": [0]})

# ============================================================================
# Fixtures for creating test qubes
# ============================================================================


@pytest.fixture
def simple_qube():
    """Create a simple qube with one axis."""
    return Qube.from_datacube({"step": [6, 12]})


@pytest.fixture
def surface_variables_qube():
    """Create a qube representing surface variables."""
    return Qube.from_datacube(
        {
            "step": [6, 12],
            "param": ["100u", "100v", "10u", "10v", "2d", "2t"],
        }
    )


@pytest.fixture
def pressure_level_qube():
    """Create a qube representing pressure level variables."""
    return Qube.from_datacube(
        {
            "step": [6, 12],
            "param": ["q", "t", "u", "v"],
            "level": [50, 100, 150, 200, 250],
        }
    )


@pytest.fixture
def hierarchical_qube():
    """Create a hierarchical qube with two branches.

    Structure after compress:
    root
    ├── param=100u/100v/10u/10v/2d/2t, step=6/12
    └── level=50/100/150/200/250, param=q/t/u/v, step=6/12

    Both children have step dimension in the qube.
    After expansion, children should have BOTH step AND their own dims.
    """
    surface = Qube.from_datacube(
        {
            "step": [6, 12],
            "param": ["100u", "100v", "10u", "10v", "2d", "2t"],
        }
    )

    pressure = Qube.from_datacube(
        {
            "step": [6, 12],
            "param": ["q", "t", "u", "v"],
            "level": [50, 100, 150, 200, 250],
        }
    )

    qube = surface | pressure
    qube.compress()

    return qube


@pytest.fixture
def hierarchical_qube_with_drop(hierarchical_qube):
    """Create a hierarchical qube and drop an axis."""
    return hierarchical_qube.drop(["step"])


@pytest.fixture
def multi_level_qube():
    """Create a multi-level qube with multiple children at different levels.

    Structure after compress:
    root
    ├── param=a/b, step=1/2/3
    └── param=c/d
        ├── class=od, step=1/2/3
        └── level=100/200, step=1/2/3

    All children have step dimension in the qube.
    After expansion, all branches should have step dimension.
    """
    child1 = Qube.from_datacube({"step": [1, 2, 3], "param": ["a", "b"]})

    child2 = Qube.from_datacube({"step": [1, 2, 3], "param": ["c", "d"], "class": ["od"]})

    nested = Qube.from_datacube(
        {
            "step": [1, 2, 3],
            "param": ["c", "d"],
            "level": [100, 200],
        }
    )

    child2_with_nested = child2 | nested

    qube = child1 | child2_with_nested
    qube.compress()

    return qube


@pytest.fixture
def empty_qube():
    """Create an empty qube."""
    return Qube.empty()


# ============================================================================
# Parametrised tests for convert_num_to_abc
# ============================================================================


@pytest.mark.parametrize(
    "num,expected",
    [
        (0, "a"),
        (1, "b"),
        (5, "f"),
        (25, "z"),
        (26, "aa"),
        (27, "ab"),
        (51, "az"),
        (52, "ba"),
        (77, "bz"),
        (701, "zz"),
        (702, "aaa"),
    ],
)
def test_convert_num_to_abc(num, expected):
    """Test number to alphabetical conversion."""
    assert _convert_num_to_abc(num) == expected


# ============================================================================
# Tests for expand_as_qube() function - core functionality
# ============================================================================


class TestExpandAsQube:
    """Test the expand_as_qube() function - the core functionality."""

    def test_expand_simple_qube(self, simple_qube):
        """Test expanding with a simple single-axis qube."""
        result = expand_as_qube(SOURCE_ACTION, simple_qube)
        for dim, values in simple_qube.axes().items():
            assert result.qube.axes()[dim] == values

    def test_expand_multi_dimensional_no_split(self, pressure_level_qube):
        """Test expanding with a multi-dimensional qube (no hierarchy)."""
        result = expand_as_qube(SOURCE_ACTION, pressure_level_qube)
        for dim, values in pressure_level_qube.axes().items():
            assert result.qube.axes()[dim] == values

    def test_expand_hierarchical_creates_branches(self, hierarchical_qube):
        """Test that hierarchical expansion creates separate branches.

        The qube structure is:
        root
        ├── param=100u/100v/10u/10v/2d/2t, step=6/12
        └── level=50/100/150/200/250, param=q/t/u/v, step=6/12

        Each branch should have all dimensions from the qube.
        """
        result = expand_as_qube(SOURCE_ACTION, hierarchical_qube)
        for dim, values in hierarchical_qube.axes().items():
            assert result.qube.axes()[dim] == values

    def test_expand_uses_alphabetical_fallback(self):
        """Test that expansion uses alphabetical naming when metadata is missing.

        The qube structure is:
        root, step=1/2
        ├── param=a/b
        └── param=c/d

        Since children lack name metadata, they should be named /a and /b.
        Each branch should have parent's step dimension expanded on it.
        """
        qube = Qube.from_ascii("""root
├── step=1/2
│   └── param=a/b
└── step=1/2
    └── param=c/d
""")

        result = expand_as_qube(SOURCE_ACTION, qube)
        for dim, values in qube.axes().items():
            assert result.qube.axes()[dim] == values

    def test_expand_handles_nested_structure(self, multi_level_qube):
        """Test expansion with nested qube structure.

        The multi_level_qube has multiple children at the root level.
        After expansion, each branch should have the step dimension.
        """
        result = expand_as_qube(SOURCE_ACTION, multi_level_qube)
        for dim, values in multi_level_qube.axes().items():
            assert result.qube.axes()[dim] == values


# ============================================================================
# Edge cases and error conditions
# ============================================================================


def test_expansion_with_no_children_returns_early(empty_qube):
    """Test that expansion with no children returns immediately."""
    result = expand_as_qube(SOURCE_ACTION, empty_qube)

    # Action should be returned unchanged
    assert result is SOURCE_ACTION


# ============================================================================
# Integration tests for realistic usage scenarios
# ============================================================================


def test_drop_then_expand(pressure_level_qube):
    """Test dropping an axis then expanding."""
    new_qube = pressure_level_qube.drop(["step"])
    result = expand_as_qube(SOURCE_ACTION, new_qube)
    for dim, values in new_qube.axes().items():
        assert result.qube.axes()[dim] == values


def test_complex_hierarchy_expansion(multi_level_qube):
    """Test expansion with complex nested hierarchy."""
    result = expand_as_qube(SOURCE_ACTION, multi_level_qube)
    for dim, values in multi_level_qube.axes().items():
        assert result.qube.axes()[dim] == values


# ============================================================================
# Result validation tests - verify expanded action dimensions
# ============================================================================


def test_expand_verifies_correct_dimensions(surface_variables_qube):
    """Test that expansion results in correct dimensions being expanded."""
    result = expand_as_qube(SOURCE_ACTION, surface_variables_qube)
    for dim, values in surface_variables_qube.axes().items():
        assert result.qube.axes()[dim] == values


def test_expand_verifies_dimension_values(pressure_level_qube):
    """Test that expansion uses correct values for each dimension."""
    result = expand_as_qube(SOURCE_ACTION, pressure_level_qube)
    for dim, values in pressure_level_qube.axes().items():
        assert result.qube.axes()[dim] == values


def test_expand_processes_sibling_dimensions(multi_level_qube):
    """Test that expansion handles qube with multiple sibling dimensions."""
    result = expand_as_qube(SOURCE_ACTION, multi_level_qube)
    for dim, values in multi_level_qube.axes().items():
        assert result.qube.axes()[dim] == values


def test_expand_result_has_all_qube_axes(surface_variables_qube):
    """Test that after expansion, all qube axes are accounted for."""
    result = expand_as_qube(SOURCE_ACTION, surface_variables_qube)
    for dim, values in surface_variables_qube.axes().items():
        assert result.qube.axes()[dim] == values


def test_expand_correct_value_count(simple_qube):
    """Test that expansion includes all values for each dimension."""
    result = expand_as_qube(SOURCE_ACTION, simple_qube)
    for dim, values in simple_qube.axes().items():
        assert result.qube.axes()[dim] == values


# ============================================================================
# Integration test with real Action object
# ============================================================================


def test_expand_as_qube_with_real_action():
    """Test that expand_as_qube works with a real Action object."""
    from earthkit.workflows.fluent import Action

    action = from_source("test", datacubes={"dim_0": [0, 1], "dim_1": [0]})

    # Create a simple qube
    qube = Qube.from_datacube({"step": [6, 12]})

    # Expand the action using the qube
    result = expand_as_qube(action, qube)

    # Verify that the result is an Action
    assert isinstance(result, Action)

    # Verify that the action has been expanded with the step dimension
    assert "step" in result.nodeqube.dimensions()


@pytest.mark.parametrize(
    "qube_fixture",
    [
        "pressure_level_qube",
        "hierarchical_qube",
    ],
)
def test_expand_as_qube_with_real_action_post_select(qube_fixture, request):
    qube = request.getfixturevalue(qube_fixture)
    action = from_source("test", datacubes={"dim_0": [0, 1], "dim_1": [0, 1]})

    result = expand_as_qube(action, qube)
    subset = result.select(param="t")

    dims = subset.qube.axes()
    assert "step" in dims
    assert "param" in dims

    assert dims["param"] == ["t"]

    with pytest.raises(ValueError):
        subset = result.select(param="nonexistent_param")


@pytest.mark.parametrize(
    "qube_fixture",
    [
        "pressure_level_qube",
        "hierarchical_qube",
    ],
)
def test_expand_as_qube_with_real_action_post_select_level(qube_fixture, request):
    qube = request.getfixturevalue(qube_fixture)
    action = from_source("test", datacubes={"dim_0": [0, 1], "dim_1": [0, 1]})

    result = expand_as_qube(action, qube)
    subset = result.select(level=50)

    dims = subset.qube.axes()
    assert "step" in dims
    assert "level" in dims

    assert dims["level"] == [50]

    with pytest.raises(ValueError):
        subset = result.select(param="nonexistent_param")
