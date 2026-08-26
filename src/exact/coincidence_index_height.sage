#!/usr/bin/env sage
from exact.coincidence_index_height import local_factor_record

print([local_factor_record(j) for j in range(1, 101)])

