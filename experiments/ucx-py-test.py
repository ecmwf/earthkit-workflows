import asyncio
import time
import ucp
import numpy as np
import sys

port = 13337
n_bytes = 2**30
host = ucp.get_address(ifname='eth0')

async def send(ep):
    # recv buffer
    arr = np.empty(n_bytes, dtype='u1')
    await ep.recv(arr)
    assert np.count_nonzero(arr) == np.array(0, dtype=np.int64)
    print("Received NumPy array")

    # increment array and send back
    arr += 1
    print("Sending incremented NumPy array")
    await ep.send(arr)

    await ep.close()
    lf.close()

async def server_main():
    global lf
    lf = ucp.create_listener(send, port)

    while not lf.closed():
        await asyncio.sleep(0.1)

async def client_main():
    host = ucp.get_address(ifname='eth0')  # ethernet device name
    ep = await ucp.create_endpoint(host, port)
    msg = np.zeros(n_bytes, dtype='u1') # create some data to send

    # send message
    print("Send Original NumPy array")
    await ep.send(msg)  # send the real message

    # recv response
    print("Receive Incremented NumPy arrays")
    resp = np.empty_like(msg)
    await ep.recv(resp)  # receive the echo
    await ep.close()
    np.testing.assert_array_equal(msg + 1, resp)


if __name__ == '__main__':

    if sys.argv[1] == "server":
        asyncio.run(server_main())
    else:
        asyncio.run(client_main())