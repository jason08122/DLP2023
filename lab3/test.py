from typing import Dict, List, Tuple
from functools import reduce

filters = (25, 50, 100, 200)

# for idx, num_of_filters in enumerate(filters[:-1], start=1):
#     print(f'idx: {idx}, num_of_filters: {num_of_filters} \n')

flatten_size = filters[-1] * reduce(lambda x, _: round((x - 4) / 2), filters[:-1], 373)

print(flatten_size)