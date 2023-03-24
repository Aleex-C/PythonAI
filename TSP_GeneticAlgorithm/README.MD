# TSP
---
Own implementation for the TSP using:
- fitness-proportional selection ("roulette-wheel")
- ordered crossover
- swapping-index mutation (taking consideration of the TSP rules (i.e all cities must be present in a chromosome)

The population is a list of routes, each route being a sequence of traversal, city-by-city. Data is taken from the TSPLIB95 database.