import xxhash
from functools import lru_cache

#@lru_cache(maxsize=None)
def fast_hash(s):
    return xxhash.xxh3_64_intdigest(s)

