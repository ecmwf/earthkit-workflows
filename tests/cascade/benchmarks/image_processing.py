# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cascade.cascade import Cascade
from cascade.fluent import Fluent, Payload  # type: ignore
from cascade.visualise import visualise


def mandelbrot(c, max_iter):
    """Calculate the color value for a point in the complex plane based on the Mandelbrot set."""
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    if n == max_iter:
        return max_iter
    return n + 1 - np.log(np.log2(abs(z)))


def generate_fractal_image(
    xmin=0, xmax=100, ymin=0, ymax=100, width=200, height=200, max_iter=5
):
    """Generate a fractal image of the Mandelbrot set."""
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return np.array([[mandelbrot(complex(r, i), max_iter) for r in r1] for i in r2])


def average_images(images):
    # Check if input is a 3D array
    if images.ndim != 3:
        raise ValueError("The input array must be 3-dimensional.")
    # Compute the pixel-wise mean along the first dimension (image dimension)
    average_image = np.mean(images, axis=0)
    return average_image


def plot_image(image, cmap="gray"):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    png = buf.getvalue()
    buf.close()
    return png


def plot_multiple_images(images, cmap="gray"):
    _, axes = plt.subplots(1, len(images), figsize=(15, 5))

    if len(images) == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for ax, png in zip(axes, images):
        img = Image.open(io.BytesIO(png))
        ax.imshow(img)
        ax.axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()


def test_cascade_graph():
    graph = (
        Fluent()
        .source(generate_fractal_image, np.ndarray((2, 3)))
        .reduce(Payload(average_images), "dim_0")
        .map(Payload(plot_image))
        .reduce(Payload(plot_multiple_images))
    )

    visualise(graph.graph(), "mandelbrot.html")

    schedule = Cascade.schedule(graph)

    Cascade.execute(schedule)


test_cascade_graph()
