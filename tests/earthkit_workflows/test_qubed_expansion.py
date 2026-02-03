# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from unittest.mock import MagicMock

import pytest
from qubed import Qube

from earthkit.workflows._qubed import _convert_num_to_abc, expand_as_qube

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
    """Create a hierarchical qube with named children (surface and pressure)."""
    surface = Qube.from_datacube(
        {
            "param": ["100u", "100v", "10u", "10v", "2d", "2t"],
        }
    )
    surface.add_metadata({"name": "surface"})

    pressure = Qube.from_datacube(
        {
            "param": ["q", "t", "u", "v"],
            "level": [50, 100, 150, 200, 250],
        }
    )
    pressure.add_metadata({"name": "pressure"})

    parent = Qube.from_datacube({"step": [6, 12]})
    combined = surface | pressure
    parent_with_children = parent | combined

    return parent_with_children


@pytest.fixture
def hierarchical_qube_with_drop(hierarchical_qube):
    """Create a hierarchical qube and drop an axis."""
    return hierarchical_qube.remove_by_key("step")


@pytest.fixture
def multi_level_qube():
    """Create a multi-level qube with multiple children at different levels."""
    child1 = Qube.from_datacube({"param": ["a", "b"]})
    child1.add_metadata({"name": "group1"})

    child2 = Qube.from_datacube({"param": ["c", "d"]})
    child2.add_metadata({"name": "group2"})

    nested = Qube.from_datacube({"level": [100, 200]})
    nested.add_metadata({"name": "nested"})

    child2_with_nested = child2 | nested
    parent = Qube.from_datacube({"step": [1, 2, 3]})
    combined = parent | (child1 | child2_with_nested)

    return combined


@pytest.fixture
def single_child_qube():
    """Create a qube with truly a single child by having one branch."""
    qube = Qube.from_datacube(
        {
            "step": [6, 12],
            "param": ["t", "q"],
        }
    )
    return qube


@pytest.fixture
def empty_qube():
    """Create an empty qube."""
    return Qube.empty()


@pytest.fixture
def mock_action():
    """Create a mock Action object with necessary methods."""
    action = MagicMock()
    action.expand.return_value = action
    action.split.return_value = action

    action.expand_calls = []
    action.split_calls = []

    def track_expand(*args, **kwargs):
        action.expand_calls.append((args, kwargs))
        return action

    def track_split(*args, **kwargs):
        action.split_calls.append((args, kwargs))
        return action

    action.expand.side_effect = track_expand
    action.split.side_effect = track_split

    return action


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

    def test_expand_simple_qube(self, simple_qube, mock_action):
        """Test expanding with a simple single-axis qube."""
        expand_as_qube(mock_action, simple_qube)

        assert len(mock_action.expand_calls) == 1
        args, kwargs = mock_action.expand_calls[0]
        assert args[0] == ("step", [6, 12])
        assert args[1] == ("step", [6, 12])
        assert kwargs["path"] == ""
        assert len(mock_action.split_calls) == 0

    def test_expand_multi_dimensional_no_split(self, single_child_qube, mock_action):
        """Test expanding with a multi-dimensional qube (no hierarchy)."""
        expand_as_qube(mock_action, single_child_qube)
        assert len(mock_action.expand_calls) >= 1
        assert len(mock_action.split_calls) == 0

    def test_expand_uses_child_names(self, hierarchical_qube, mock_action):
        """Test that expansion uses child metadata names in split paths."""
        expand_as_qube(mock_action, hierarchical_qube)

        if len(mock_action.split_calls) > 0:
            split_dict = mock_action.split_calls[0][0][0]
            paths = list(split_dict.keys())
            assert any("surface" in path for path in paths) or any("pressure" in path for path in paths)

    def test_expand_uses_alphabetical_fallback(self, mock_action):
        """Test that expansion uses alphabetical naming when metadata is missing."""
        child1 = Qube.from_datacube({"param": ["a", "b"]})
        child2 = Qube.from_datacube({"param": ["c", "d"]})
        parent = Qube.from_datacube({"step": [1, 2]})
        qube = parent | (child1 | child2)

        expand_as_qube(mock_action, qube)

        if len(mock_action.split_calls) > 0:
            split_dict = mock_action.split_calls[0][0][0]
            paths = list(split_dict.keys())
            assert any(path.endswith("a") for path in paths) or any(path.endswith("b") for path in paths)

    def test_expand_handles_nested_structure(self, multi_level_qube, mock_action):
        """Test expansion with nested qube structure."""
        result = expand_as_qube(mock_action, multi_level_qube)
        assert len(mock_action.expand_calls) > 0
        assert result == mock_action


# ============================================================================
# Edge cases and error conditions
# ============================================================================


def test_expansion_with_no_children_returns_early(mock_action):
    """Test that expansion with no children returns immediately."""
    empty_qube = Qube.empty()
    result = expand_as_qube(mock_action, empty_qube)
    assert result == mock_action
    assert len(mock_action.expand_calls) == 0
    assert len(mock_action.split_calls) == 0


# ============================================================================
# Integration tests for realistic usage scenarios
# ============================================================================


def test_drop_then_expand(pressure_level_qube, mock_action):
    """Test dropping an axis then expanding."""
    new_qube = pressure_level_qube.remove_by_key("step")
    expand_as_qube(mock_action, new_qube)

    for args, _kwargs in mock_action.expand_calls:
        assert args[0][0] != "step"


def test_complex_hierarchy_expansion(multi_level_qube, mock_action):
    """Test expansion with complex nested hierarchy."""
    result = expand_as_qube(mock_action, multi_level_qube)
    assert result == mock_action
    assert len(mock_action.expand_calls) >= 1


# ============================================================================
# Result validation tests - verify expanded action dimensions
# ============================================================================


def test_expand_verifies_correct_dimensions(surface_variables_qube, mock_action):
    """Test that expansion results in correct dimensions being expanded."""
    expand_as_qube(mock_action, surface_variables_qube)

    expanded_dims = set()
    for args, _kwargs in mock_action.expand_calls:
        dim_name = args[0][0]
        expanded_dims.add(dim_name)

    # Should have expanded at least one dimension
    assert len(expanded_dims) >= 1
    # The first dimension should be step or param
    first_dim = list(expanded_dims)[0] if expanded_dims else None
    assert first_dim in ["step", "param"]


def test_expand_verifies_dimension_values(pressure_level_qube, mock_action):
    """Test that expansion uses correct values for each dimension."""
    expand_as_qube(mock_action, pressure_level_qube)

    dim_values = {}
    for args, _kwargs in mock_action.expand_calls:
        dim_name = args[0][0]
        dim_vals = args[0][1]
        dim_values[dim_name] = dim_vals

    # Should have expanded the dimensions (at least one)
    assert len(dim_values) >= 1
    # Check for expected dimensions
    for dim in ["step", "param", "level"]:
        if dim in dim_values:
            if dim == "step":
                assert 6 in dim_values["step"] and 12 in dim_values["step"]
            elif dim == "param":
                assert "q" in dim_values["param"]
            elif dim == "level":
                assert 50 in dim_values["level"] and 250 in dim_values["level"]


def test_expand_hierarchy_creates_correct_paths(hierarchical_qube, mock_action):
    """Test that hierarchical expansion creates correct path structure."""
    expand_as_qube(mock_action, hierarchical_qube)

    assert len(mock_action.split_calls) == 1
    assert len(mock_action.expand_calls) == 4

    all_paths = []
    for args, _kwargs in mock_action.split_calls:
        split_dict = args[0]
        all_paths.extend(split_dict.keys())

    path_strings = " ".join(all_paths)
    assert "surface" in path_strings or "pressure" in path_strings


def test_expand_hierarchy_dropped_creates_correct_paths(hierarchical_qube_with_drop, mock_action):
    """Test that hierarchical expansion creates correct path structure."""
    expand_as_qube(mock_action, hierarchical_qube_with_drop)

    assert len(mock_action.split_calls) == 1
    assert len(mock_action.expand_calls) == 3
    all_paths = []
    for args, _kwargs in mock_action.split_calls:
        split_dict = args[0]
        all_paths.extend(split_dict.keys())

    path_strings = " ".join(all_paths)
    assert "surface" in path_strings or "pressure" in path_strings


def test_expand_processes_sibling_dimensions(multi_level_qube, mock_action):
    """Test that expansion handles qube with multiple sibling dimensions."""
    expand_as_qube(mock_action, multi_level_qube)

    expanded_dims = set()
    for args, _kwargs in mock_action.expand_calls:
        dim_name = args[0][0]
        expanded_dims.add(dim_name)

    assert len(expanded_dims) >= 1
    assert "level" in expanded_dims


def test_expand_result_has_all_qube_axes(surface_variables_qube):
    """Test that after expansion, all qube axes are accounted for."""
    original_axes = surface_variables_qube.axes()

    mock_action = MagicMock()
    expanded_keys = set()

    def track_expand(*args, **kwargs):
        if len(args) > 0:
            expanded_keys.add(args[0][0])
        return mock_action

    mock_action.expand.side_effect = track_expand
    mock_action.split.return_value = mock_action

    expand_as_qube(mock_action, surface_variables_qube)

    for axis in original_axes:
        assert axis in expanded_keys, f"Axis {axis} was not expanded"


def test_expand_correct_value_count(simple_qube, mock_action):
    """Test that expansion includes all values for each dimension."""
    expand_as_qube(mock_action, simple_qube)

    # For simple qube, should expand step dimension with 2 values
    assert len(mock_action.expand_calls) >= 1
    args, _kwargs = mock_action.expand_calls[0]
    dim_values = args[0][1]
    assert len(dim_values) == 2
    assert 6 in dim_values
    assert 12 in dim_values


# ============================================================================
# Integration test with real Action object
# ============================================================================


def test_expand_as_qube_with_real_action():
    """Test that expand_as_qube works with a real Action object."""
    from earthkit.workflows.fluent import Action
    from .helpers import mock_action 

    action = mock_action(shape=(1, 1))

    # Create a simple qube
    qube = Qube.from_datacube({"step": [6, 12]})

    # Expand the action using the qube
    result = expand_as_qube(action, qube)

    # Verify that the result is an Action
    assert isinstance(result, Action)

    # Verify that the action has been expanded with the step dimension
    assert "step" in result.nodes.to_dataset().dims
