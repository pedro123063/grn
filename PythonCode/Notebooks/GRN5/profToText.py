import pstats

# Load your profile file
p = pstats.Stats('28_05_2026_13_09_GRN5_DE_Parallel_1000_generations_results.prof')

# Sort by cumulative time to see the biggest bottlenecks
print("--- TOP CUMULATIVE TIME ---")
p.strip_dirs().sort_stats('cumulative').print_stats(40)

# Sort by internal time (tottime) to see heavy individual functions
print("\n--- TOP TOTAL TIME (EXCLUDING SUB-CALLS) ---")
p.sort_stats('tottime').print_stats(40)