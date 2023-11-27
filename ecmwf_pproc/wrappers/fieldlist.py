import os
import importlib

from earthkit.data import FieldList as EKFieldList
from earthkit.data.sources.numpy_list import NumpyFieldList


class ArrayFieldList(NumpyFieldList):
    def __init__(self, array, metadata, *args, **kwargs):
        self._array = array
        self._metadata = metadata

        if not isinstance(self._metadata, list):
            self._metadata = [self._metadata]

        if isinstance(self._array, np.ndarray):
            if self._array.shape[0] != len(self._metadata):
                # we have a single array and a single metadata
                if len(self._metadata) == 1 and self._shape_match(
                    self._array.shape, self._metadata[0].geography.shape()
                ):
                    self._array = np.array([self._array])
                else:
                    raise ValueError(
                        (
                            f"first array dimension ({self._array.shape[0]}) differs "
                            f"from number of metadata objects ({len(self._metadata)})"
                        )
                    )
        elif isinstance(self._array, list):
            if len(self._array) != len(self._metadata):
                raise ValueError(
                    (
                        f"array len ({len(self._array)}) differs "
                        f"from number of metadata objects ({len(self._metadata)})"
                    )
                )

            for i, a in enumerate(self._array):
                if not isinstance(a, np.ndarray):
                    raise ValueError(
                        f"All array element must be an ndarray. Type at position={i} is {type(a)}"
                    )

        else:
            raise TypeError("array must be an ndarray or a list of ndarrays")

        # hide internal metadata related to values
        self._metadata = [md._hide_internal_keys() for md in self._metadata]

        super().__init__(*args, **kwargs)

    def _shape_match(self, shape1, shape2):
        if shape1 == shape2:
            return True
        if len(shape1) == 1 and shape1[0] == np.prod(shape2):
            return True
        return False


class FieldList(EKFieldList):
    def from_numpy(array, metadata):
        xp = importlib.import_module(os.getenv("CASCADE_ARRAY_MODULE", "numpy"))

        return ArrayFieldList(xp.asarray(array), metadata)
