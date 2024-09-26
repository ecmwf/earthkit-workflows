"""
Low Level representation of Cascade graphs -- not expected to be user facing.

Used to stabilise contract between Cascade graphs and Schedulers and Executors.

Works on atomic level: a single callable with all necessary information to be
executed in an isolated process.
"""
