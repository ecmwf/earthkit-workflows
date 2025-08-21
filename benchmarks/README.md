At the moment just an adhoc collection of scripts, utils and scenarios.

# Scenario: SHM Throughput
It seems that `AF_UNIX` + `SOCK_STREAM` and `AF_INET` + `SOCK_DGRAM` are, performance-wise, indistinguishable at the `shm` scale.
That leads us to prefer the former, because:
 - `SOCK_STREAM` is more reliable in general (though should not be noticeable in the localhost case),
 - it is easier to pick a random file name than a random *non-occupied* port number, and it's easier to free it after usage.
