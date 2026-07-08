# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from qubed import Qube

from earthkit.workflows._qubed import _convert_num_to_abc, expand_as_qube
from earthkit.workflows.nodetree import nodetree_array

from .helpers import mock_action

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


# Branch names: the Rust Qube does not yet support metadata, so hierarchical
# qubes use the alphabetical fallback naming (a, b, c, ...).
# TODO: Once metadata is supported, inject ``name`` metadata and update the
# expected branch names in the tests below.
BRANCH_A = "/a"
BRANCH_B = "/b"
BRANCH_C = "/c"


@pytest.fixture
def hierarchical_qube():
    """Create a hierarchical qube with two branches.

    Structure after compress:
    root
    ├── param=100u/100v/10u/10v/2d/2t, step=6/12  (branch /a)
    └── level=50/100/150/200/250, param=q/t/u/v, step=6/12  (branch /b)

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
    ├── param=a/b, step=1/2/3 (branch /a)
    └── param=c/d (branch /b or further split)
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
        action = mock_action((1,))
        result = expand_as_qube(action, simple_qube)

        # Verify the result has the step dimension
        ds = result.nodes.to_dataset()
        assert "step" in ds.dims
        assert len(ds.step) == 2
        assert list(ds.step.values) == [6, 12]

    def test_expand_multi_dimensional_no_split(self, pressure_level_qube):
        """Test expanding with a multi-dimensional qube (no hierarchy)."""
        action = mock_action((1,))
        result = expand_as_qube(action, pressure_level_qube)

        # Verify the result has both dimensions
        ds = result.nodes.to_dataset()
        assert "step" in ds.dims
        assert "param" in ds.dims
        assert len(ds.step) == 2
        assert len(ds.param) == 4

    def test_expand_hierarchical_creates_branches(self, hierarchical_qube):
        """Test that hierarchical expansion creates separate branches.

        The qube structure is:
        root
        ├── param=100u/100v/10u/10v/2d/2t, step=6/12  (branch /a)
        └── level=50/100/150/200/250, param=q/t/u/v, step=6/12  (branch /b)

        Expected expanded action structure:
        /a: DataArray with dims (step, param)
        /b: DataArray with dims (step, param, level)

        Each branch should have all dimensions from the qube.
        """
        action = mock_action((1,))
        result = expand_as_qube(action, hierarchical_qube)

        # Verify the result has a hierarchical structure with branches
        groups = list(result.nodes.groups)
        assert BRANCH_A in groups
        assert BRANCH_B in groups

        # First branch should have step and param
        branch_a_ds = result.nodes[BRANCH_A].to_dataset()
        assert "step" in branch_a_ds.dims, "Branch /a should have step dimension"
        assert "param" in branch_a_ds.dims, "Branch /a should have param dimension"
        assert len(branch_a_ds.step) == 2, "Branch /a should have 2 step values"
        assert len(branch_a_ds.param) == 6, "Branch /a should have 6 param values"

        # Second branch should have step, param, AND level
        branch_b_ds = result.nodes[BRANCH_B].to_dataset()
        assert "step" in branch_b_ds.dims, "Branch /b should have step dimension"
        assert "param" in branch_b_ds.dims, "Branch /b should have param dimension"
        assert "level" in branch_b_ds.dims, "Branch /b should have level dimension"
        assert len(branch_b_ds.step) == 2, "Branch /b should have 2 step values"
        assert len(branch_b_ds.param) == 4, "Branch /b should have 4 param values"
        assert len(branch_b_ds.level) == 5, "Branch /b should have 5 level values"

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

        action = mock_action((1,))
        result = expand_as_qube(action, qube)

        # Verify alphabetical branch names are used (a, b for first two children)
        groups = list(result.nodes.groups)
        assert "/a" in groups, "First child should be named /a"
        assert "/b" in groups, "Second child should be named /b"

        # Each branch should have BOTH step (parent) and param (child) dimensions
        ds_a = result.nodes["/a"].to_dataset()
        assert "step" in ds_a.dims, "Branch /a should have parent's step dimension"
        assert "param" in ds_a.dims, "Branch /a should have its own param dimension"
        assert len(ds_a.step) == 2, "Branch /a should have 2 step values"
        assert len(ds_a.param) == 2, "Branch /a should have 2 param values"

        ds_b = result.nodes["/b"].to_dataset()
        assert "step" in ds_b.dims, "Branch /b should have parent's step dimension"
        assert "param" in ds_b.dims, "Branch /b should have its own param dimension"
        assert len(ds_b.step) == 2, "Branch /b should have 2 step values"
        assert len(ds_b.param) == 2, "Branch /b should have 2 param values"

    def test_expand_handles_nested_structure(self, multi_level_qube):
        """Test expansion with nested qube structure.

        The multi_level_qube has multiple children at the root level.
        After expansion, each branch should have the step dimension.
        """
        action = mock_action((1,))
        result = expand_as_qube(action, multi_level_qube)

        # Verify the result has branches (alphabetical naming)
        groups = list(result.nodes.groups)
        assert BRANCH_A in groups, f"Should have {BRANCH_A} branch"
        assert BRANCH_B in groups, f"Should have {BRANCH_B} branch"

        # First branch should have step and param
        ds_a = result.nodes[BRANCH_A].to_dataset()
        assert "step" in ds_a.dims, f"{BRANCH_A} should have step dimension"
        assert "param" in ds_a.dims, f"{BRANCH_A} should have param dimension"
        assert len(ds_a.step) == 3, f"{BRANCH_A} should have 3 step values"
        assert len(ds_a.param) == 2, f"{BRANCH_A} should have 2 param values"


# ============================================================================
# Edge cases and error conditions
# ============================================================================


def test_expansion_with_no_children_returns_early(empty_qube):
    """Test that expansion with no children returns immediately."""
    action = mock_action((1,))
    result = expand_as_qube(action, empty_qube)

    # Action should be returned unchanged
    assert result is action


# ============================================================================
# Integration tests for realistic usage scenarios
# ============================================================================


def test_drop_then_expand(pressure_level_qube):
    """Test dropping an axis then expanding."""
    action = mock_action((1,))
    new_qube = pressure_level_qube.drop(["step"])
    result = expand_as_qube(action, new_qube)

    # Verify step dimension is not present
    ds = result.nodes.to_dataset()
    assert "step" not in ds.dims
    # But other dimensions should be present
    assert "param" in ds.dims
    assert "level" in ds.dims


def test_complex_hierarchy_expansion(multi_level_qube):
    """Test expansion with complex nested hierarchy."""
    action = mock_action((1,))
    result = expand_as_qube(action, multi_level_qube)

    # Verify branches exist
    groups = list(result.nodes.groups)
    assert any(g in groups for g in [BRANCH_A, BRANCH_B, BRANCH_C])

    # Verify step dimension exists somewhere
    has_step = False
    for path in groups:
        try:
            ds = result.nodes[path].to_dataset()
            if "step" in ds.dims:
                has_step = True
                break
        except:
            pass
    assert has_step


# ============================================================================
# Result validation tests - verify expanded action dimensions
# ============================================================================


def test_expand_verifies_correct_dimensions(surface_variables_qube):
    """Test that expansion results in correct dimensions being expanded."""
    action = mock_action((1,))
    result = expand_as_qube(action, surface_variables_qube)

    ds = result.nodes.to_dataset()
    assert "step" in ds.dims
    assert "param" in ds.dims


def test_expand_verifies_dimension_values(pressure_level_qube):
    """Test that expansion uses correct values for each dimension."""
    action = mock_action((1,))
    result = expand_as_qube(action, pressure_level_qube)

    ds = result.nodes.to_dataset()

    # Check dimension values
    assert "step" in ds.dims
    assert 6 in ds.step.values and 12 in ds.step.values

    assert "param" in ds.dims
    assert "q" in ds.param.values

    assert "level" in ds.dims
    assert 50 in ds.level.values and 250 in ds.level.values


def test_expand_hierarchy_creates_correct_paths(hierarchical_qube):
    """Test that hierarchical expansion creates correct path structure."""
    action = mock_action((1,))
    result = expand_as_qube(action, hierarchical_qube)

    # Verify branch paths exist
    groups = list(result.nodes.groups)
    assert BRANCH_A in groups
    assert BRANCH_B in groups


def test_expand_hierarchy_dropped_creates_correct_paths(hierarchical_qube_with_drop):
    """Test that hierarchical expansion creates correct path structure."""
    action = mock_action((1,))
    result = expand_as_qube(action, hierarchical_qube_with_drop)

    # Verify branch paths exist
    groups = list(result.nodes.groups)
    assert BRANCH_A in groups
    assert BRANCH_B in groups

    # Verify step is not in dimensions
    branch_a_ds = result.nodes[BRANCH_A].to_dataset()
    assert "step" not in branch_a_ds.dims


def test_expand_processes_sibling_dimensions(multi_level_qube):
    """Test that expansion handles qube with multiple sibling dimensions."""
    action = mock_action((1,))
    result = expand_as_qube(action, multi_level_qube)

    # Check for nested level dimension
    found_level = False
    for path in result.nodes.groups:
        try:
            ds = result.nodes[path].to_dataset()
            if "level" in ds.dims:
                found_level = True
                break
        except:
            pass

    assert found_level


def test_expand_result_has_all_qube_axes(surface_variables_qube):
    """Test that after expansion, all qube axes are accounted for."""
    action = mock_action((1,))
    original_axes = surface_variables_qube.axes()  # native Rust method

    result = expand_as_qube(action, surface_variables_qube)
    ds = result.nodes.to_dataset()

    for axis in original_axes:
        assert axis in ds.dims, f"Axis {axis} was not expanded"


def test_expand_correct_value_count(simple_qube):
    """Test that expansion includes all values for each dimension."""
    action = mock_action((1,))
    result = expand_as_qube(action, simple_qube)

    ds = result.nodes.to_dataset()
    assert len(ds.step) == 2
    assert 6 in ds.step.values
    assert 12 in ds.step.values


# ============================================================================
# Integration test with real Action object
# ============================================================================


def test_expand_as_qube_with_real_action():
    """Test that expand_as_qube works with a real Action object."""
    from earthkit.workflows.fluent import Action

    action = mock_action(shape=(2, 1))

    # Create a simple qube
    qube = Qube.from_datacube({"step": [6, 12]})

    # Expand the action using the qube
    result = expand_as_qube(action, qube)

    # Verify that the result is an Action
    assert isinstance(result, Action)

    # Verify that the action has been expanded with the step dimension
    assert "step" in result.nodes.to_dataset().dims


@pytest.mark.parametrize(
    "qube_fixture",
    [
        "pressure_level_qube",
        "hierarchical_qube",
    ],
)
def test_expand_as_qube_with_real_action_post_select(qube_fixture, request):
    qube = request.getfixturevalue(qube_fixture)
    action = mock_action(shape=(2, 2))

    result = expand_as_qube(action, qube)
    subset = result.select(param="t")

    da = nodetree_array(subset.nodes)
    assert "step" in da.dims
    assert "param" not in da.dims
    assert "param" in da.coords

    assert da.param == "t"

    with pytest.raises(IndexError):
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
    action = mock_action(shape=(2, 2))

    result = expand_as_qube(action, qube)
    subset = result.select(level=50)

    da = nodetree_array(subset.nodes)
    assert "step" in da.dims
    assert "level" not in da.dims
    assert "level" in da.coords

    assert da.level == 50

    with pytest.raises(IndexError):
        subset = result.select(param="nonexistent_param")
