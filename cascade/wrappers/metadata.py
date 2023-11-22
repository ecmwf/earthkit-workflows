import copy

from earthkit.data.core.metadata import RawMetadata
from earthkit.data.readers.grib.metadata import GribMetadata, GribFieldGeography
from earthkit.data.readers.grib.memory import GribMessageMemoryReader
from earthkit.data.readers.grib.codes import GribCodesHandle


class GribBufferMetaData(RawMetadata):
    def __init__(self, grib_metadata: GribMetadata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        metadata = grib_metadata._handle.copy()
        metadata.set_array("values", metadata.get_array("values").shape)
        self.buffer = metadata.get_buffer()

    @property
    def geography(self):
        return GribFieldGeography(self.buffer_to_metadata())

    def override(self, *args, **kwargs):
        new_metadata = copy.deepcopy(self)
        new_metadata._d.update(*args, **kwargs)
        return new_metadata

    def buffer_to_metadata(self) -> GribMetadata:
        return GribMetadata(
            GribCodesHandle(
                GribMessageMemoryReader(self.buffer)._next_handle(), None, None
            )
        )
