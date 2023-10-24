


import cProfile
from io import TextIOWrapper
import pstats
from typing import Callable, Optional, Text

def profile_and_stats(callable_func: Callable, output_path: Optional[Text] = None):
    profile = cProfile.Profile()
    result = profile.runcall(callable_func)
    stream: Optional[TextIOWrapper] = open(output_path, 'w') if output_path is not None else None
    ps = pstats.Stats(profile, stream=stream)
    ps.sort_stats('cumtime') 
    ps.print_stats()
    return result

# Example usage:
# profile_and_stats(lambda: main(), "profile_stats.txt")

# def short_main(timeout: int = 120):
#     utils.run_with_timeout(main, timeout=timeout)
# def safe_short_main():
#     utils.catch_and_print(short_main)  
# # Save output in file with current date and time
# output_path=f"analyse_layer_bottleneck_{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.txt"
# bottleneck.profile_and_stats(safe_short_main, output_path=output_path)