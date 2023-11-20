from earthkit.data.sources.mars import MarsRetriever


class MarsRetrieverWithCache(MarsRetriever):
    @classmethod
    def _reverse_request_formatting(cls, request: dict):
        new_request = request.copy()
        new_request["param"] = (
            request["param"][0] if len(request["param"]) == 1 else request["param"]
        )
        new_request["date"] = [d.replace("-", "") for d in request["date"]]
        if len(new_request["date"]) == 1:
            new_request["date"] = new_request["date"][0]
        return new_request

    def _retrieve(self, request):
        cache_path = request.pop("cache", None)
        if cache_path is None:
            return super()._retrieve(request)

        cache = (
            None
            if cache_path is None
            else cache_path.format_map(self._reverse_request_formatting(request))
        )
        self.service().execute(request, cache)
        return cache
