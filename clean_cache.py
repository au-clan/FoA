from diskcache import Cache
from tqdm import tqdm

def clean_cache(cache_filepath):
    cache = Cache(cache_filepath, size_limit=int(2e10))
    # Iterate over the cache keys
    for key in tqdm(cache.iterkeys()):
        # Get the list associated with the key
        value = cache.get(key)
        
        # Remove tuples where the first element is None
        modified_list = [item for item in value if item[0] is not None]
        
        # Update the cache with the modified list
        cache.set(key, modified_list)
        assert False, "This is a test"
        exit()

    # Close the cache
    cache.close()

def main():
    clean_cache("caches/crosswords")
    clean_cache("caches/gameof24")

if __name__ == "__main__":
    main()