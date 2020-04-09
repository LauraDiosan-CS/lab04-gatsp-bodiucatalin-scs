import os
import numpy as np, random, operator
import pandas as pd

class Fitness:
    def __init__(self, route, mat):
        self.route = route
        self.mat = mat
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += self.mat[fromCity][toCity]
            self.distance = pathDistance
            print("Distance:",self.distance)
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
            print("Fitnes:", self.fitness)
        return self.fitness




def createRoute(cityList):
    cities = []
    for i in range(cityList):
        cities.append(i)
    route = random.sample(cities, cityList)
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population, mat):
    fitnessResults = {}
    for i in range(0,len(population)):
        print("Routes", population[i])
        fitnessResults[i] = Fitness(population[i], mat).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate, matrice):
    popRanked = rankRoutes(currentGen, matrice)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations, dim):
    pop = initialPopulation(popSize, dim)

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate, population)

    bestRouteIndex = rankRoutes(pop, population)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def readFromFile(fileName):
    with open(fileName) as file:
        content = [line.strip() for line in file.readlines()]
        dim = int(content.pop(0))
        mat = [[int(number) for number in line.split(',')] for line in content]
        return dim, mat


def main():
    fileName = "mediumF.txt"
    currentDir = os.getcwd()
    path = os.path.join(currentDir, fileName)


    dim, mat = readFromFile(path)


    cityList = mat;


    bestRoute = geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500, dim=dim)

    print(" Final:", bestRoute)

main()