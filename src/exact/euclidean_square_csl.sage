#!/usr/bin/env sage
# Exact Sage entry point; the implementation uses only rational operations and
# is shared with the Python validation runner.
from exact.euclidean_square_csl import exact_certificate

print(exact_certificate())

