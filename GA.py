import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
inf = sys.maxsize #INT_MAX
TOURNAMENT_SIZE = 2
# seed = random.randint(1,100)
# random.seed(9)


class Board:
    
    max_clashes = 28  # nC2 = 8C2 = 8*(8-1)/2 = 56/2 = 28

    def __init__(self, state):

        self.state = state
        self.fitness = self.getFitness(state)

    def getFitness(self, state):
        s = state
        n = len(s)
        num_of_clashes = 0

        row_clashes = 0
        # calc row_clashes
        for i in range(n):
            j = i+1
            while(j < n):
                if(s[i] == s[j]):
                    row_clashes = row_clashes + 1
                j = j+1

        diagonal_clashes = 0
        # calculate diagonal clashes
        for i in range(n):
            for j in range(n):
                if (i != j):
                    dy = abs(int(s[i]) - int(s[j]))
                    dx = abs(i-j)
                    if(dx == dy):
                        diagonal_clashes += 1

        diagonal_clashes = diagonal_clashes//2
        num_of_clashes = row_clashes + diagonal_clashes
        # print(row_clashes, diagonal_clashes, num_of_clashes) #debug
        return 1 + self.max_clashes - num_of_clashes  

class BasicEightQueens:
    def __init__(self):
        self.main(20)

    def randomSelectFitParent(self, population):        
        weights = [x.fitness for x in population]
        fit_parents = random.choices(population, weights = weights, k=2)
        return fit_parents[0], fit_parents[1]


    def reproduce(self, x,y):
        n = len(x.state)
        i = random.randint(0,n-1)
        child_state = x.state[:i] + y.state[i:]
        child = Board(child_state)
        return child



    def mutate(self, child):
        n = len(child.state)
        i = random.randint(0,n-1)
        new_val = random.randint(1,n)
        new_state = ''
        if i < n-1:
            new_state =  child.state[:i] + str(new_val) + child.state[i+1:]
        else:
            new_state =  child.state[:i] + str(new_val)
        
        return Board(new_state)
    
    # def mutate(self, child):
        # n = len(child.state) #improvement-2 #swap mutation
        # i = random.randint(0,n-1)
        # j = random.randint(0,n-1)
        # temp = list(child.state)
        # t = temp[i]
        # temp[i] = temp[j]
        # temp[j] = t
        # new_state = ''
        # for i in temp:
        #     new_state += str(i)
        # return Board(new_state)

    def getBestFitness(self, population):
        return max([x.fitness for x in population])        

    
   

    def getAverageFitness(self, population):
        n = len(population)
        total_fitness = 0
        
        for x in population:
            total_fitness += x.fitness
        
        return total_fitness/n

    def init_population(self, population, k):
        # all states are same 
        # each state has queen on same row
        # state=''.join([str(random.randint(1,8))]*8)
        # population = [Board(state)]*k
        state = str(random.randint(1,8))*8
        for x in range(k):
            population.append(Board(state))       

        return population
    
    def getBestIndividual(self, population):
        bestState = sorted(population, key= lambda x: x.fitness, reverse=True)[0]
        return bestState.state
   

    def main(self, k):
        population = []
        max_fitness = 29
        generation = 1

        x_generation = []
        y_best_fitness = []

        #Initialise Population
        population = self.init_population(population,k)
        

        while not max_fitness in [x.fitness for x in population]:
           
            print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
            # print("Average Fitness: ", self.getAverageFitness(population))
            # print("Fitness Values of Population:")
            # print([x.fitness for x in population],'\n') #debug

            new_population = []

            for i in population:
                #select 2 probable fit parents
                x, y = self.randomSelectFitParent(population)
                # print("Parent 1:",x.state, x.fitness,"Parent 2:",y.state, y.fitness) #debug
                
                # crossover
                child = self.reproduce(x,y)

                #mutate this child #todo 0-1 probability
                mutate_flag = random.choices([0,1], weights = [95,5], k=1)
                if(mutate_flag[0]):
                    child = self.mutate(child)
                
                new_population.append(child)
            
    
            x_generation.append(generation)
            y_best_fitness.append(self.getBestFitness(population))
            generation += 1
            population = new_population
            
        
        
        print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
        print("Average Fitness: ", self.getAverageFitness(population))
        # print("Fitness Values of Population:")
        # print([x.fitness for x in population],'\n') #debug
        
        x_generation.append(generation)
        y_best_fitness.append(self.getBestFitness(population))
        plt.title('GA for Eight Queens Problem')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.plot(x_generation, y_best_fitness, label = 'Basic')
        plt.legend()
        # plt.show()
        print("Best Solution: ", self.getBestIndividual(population))



class ImprovedEightQueens:

    def __init__(self):
        k = 20
        self.main(k)    


    def randomSelectFitParent(self, population):
        weights = [x.fitness for x in population]
        fit_parents = random.choices(population, weights=weights, k=2)
        return fit_parents[0], fit_parents[1]

    def reproduce(self, x, y):
        n = len(x.state)
        i = random.randint(0, n-1)
        child_state = x.state[:i] + y.state[i:]
        child = Board(child_state)

        if(child.fitness > max(x.fitness, y.fitness)):  # improvement-1
            return child
        else:
            if(x.fitness > y.fitness):
                return x
            else:
                return y


#     # return child
# #     # contest_chromosomes = [child, x, y] #improvement-2
# #     # weights = [x.fitness for x in contest_chromosomes]

# #     # chosen_chromosomes = random.choices(contest_chromosomes, weights = weights, k=2)

# #     # new_x = chosen_chromosomes[0]
# #     # new_y = chosen_chromosomes[1]

# #     # i = random.randint(0,n-1)
# #     # child_state = new_x.state[:i] + new_y.state[i:]
# #     # child = Board(child_state)
# #     # return child
    # def reproduce(x,y): # improvement-3
        # #strip crossover 
        # n = len(x.state)
        # i = random.randint(0,n-2)
        # j = random.randint(i+1,n-1)
        # select = list(y.state[i:j])
        # x_list = list(x.state)
        # new_state = ''.join(x_list[:i] + select + x_list[j:])
        # child = Board(new_state)
        # return child

    # def reproduce(x,y): # improvement-2
    #     members = [x,y]
    #     n = len(x.state) 
    #     i = random.randint(0,n-1)
    #     child1_state = x.state[:i] + y.state[i:]
    #     child1 = Board(child1_state)
    #     members.append(child1)

    #     child2_state = y.state[:i] + x.state[i:]
    #     child2 = Board(child2_state)
    #     members.append(child2)

    #     i = random.randint(0,n-1)
    #     child3_state = x.state[:i] + y.state[i:]
    #     child3 = Board(child3_state)
    #     members.append(child3)

    #     child4_state = y.state[:i] + x.state[i:]
    #     child4 = Board(child4_state)
    #     members.append(child4)

    #     sorted_members = sorted(members, key= lambda x: x.fitness, reverse=True)
    #     return sorted_members[0]




    def mutate(self, child):  # improvement -3
        n = len(child.state)
        max_fitness = child.fitness
        max_child = child

        for i in range(n-1):
            for j in range(1, 9):
                temp_state = Board(child.state[:i] + str(j) + child.state[i+1:])
                if(max_fitness < temp_state.fitness):
                    max_fitness = temp_state.fitness
                    max_child = temp_state

        #for last index
        for j in range(1, 9):
            temp_state = Board(child.state[:n-1] + str(j))
            if(max_fitness < temp_state.fitness):
                max_fitness = temp_state.fitness
                max_child = temp_state

        if(max_child == child):
            n = len(child.state)
            i = random.randint(0, n-1)
            new_val = random.randint(1, n)
            new_state = ''
            if i < n-1:
                new_state = child.state[:i] + str(new_val) + child.state[i+1:]
            else:
                new_state = child.state[:i] + str(new_val)
            max_child = Board(new_state)

        return max_child


    # def mutate(self, child):


    #     n = len(child.state)
    #     i = random.randint(0,n-1)
    #     new_val = random.randint(1,n)
    #     new_state = ''
    #     if i < n-1:
    #         new_state =  child.state[:i] + str(new_val) + child.state[i+1:]
    #     else:
    #         new_state =  child.state[:i] + str(new_val)

    #     return Board(new_state)

    # def mutate(self, child): #swap_mutation #improvement-2
    #     n = len(child.state)
    #     i = random.randint(0,n-1)
    #     j = random.randint(0,n-1)
    #     temp = list(child.state)
    #     t = temp[i]
    #     temp[i] = temp[j]
    #     temp[j] = t
    #     new_state = ''
    #     for i in temp:
    #         new_state += str(i)
    #     return Board(new_state)


    def getBestFitness(self, population):
        return max([x.fitness for x in population])

#check

    def getAverageFitness(self, population):
        n = len(population)
        total_fitness = 0

        for x in population:
            total_fitness += x.fitness

        return total_fitness/n

    def init_population(self, population, k):
        state = str(random.randint(1, 8))*8
        for x in range(k):
            population.append(Board(state))

        return population
 
#check
    def selectNewPopulation(self, population, new_population):

        n = len(population)

        # print([x.fitness for x in newlist])
        n1 = math.floor((0.80)*n) + 1  # n1 best n2 probability wise
        n2 = n-n1
        sorted_new_population = sorted(
            new_population, key=lambda x: x.fitness, reverse=True)
        list1 = sorted_new_population[:n1]
        list2 = random.choices(population, k=n2)
        return (list1 + list2)

    # def selectNewPopulation(population, new_population):
    #     # print([x.fitness for x in population])
    #     # print([x.fitness for x in new_population])
    #     total_population = population + new_population
    #     # print([x.fitness for x in total_population])
    #     n = len(population)
    #     newlist = sorted(total_population, key= lambda x: x.fitness, reverse=True)
    #     # print([x.fitness for x in newlist])
    #     n1 = math.floor((0.80)*n) + 1 #n1 best n2 probability wise
    #     n2 = n-n1
    #     list1 = newlist[:n1]
    #     list2 = newlist[n1:]
    #     list2 = random.choices(list2, k=n2)
    #     return (list1 + list2)

        # weights = [x.fitness for x in total_population]
        # new_population = random.choices(total_population, weights = weights, k = n)
    
    def getBestIndividual(self, population):
        bestState = sorted(population, key= lambda x: x.fitness, reverse=True)[0]
        return bestState.state

#check
    def main(self, k):
        population = []
        max_fitness = 29
        generation = 1
        mutation_probability = 5
        x_generation = []
        y_best_fitness = []
        
        
        #Initialise Population
        population =  self.init_population(population, k)


        currentBestFitness = self.getBestFitness(population)
        previousBestFitness = currentBestFitness

        

        while not max_fitness in [x.fitness for x in population]:

            
            print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
            # print("Average Fitness: ", self.getAverageFitness(population))
            # print("Fitness Values of Population:")
            # print([x.fitness for x in population],'\n') #debug
            # print(generation, [x.fitness for x in population], self.getAverageFitness(population), self.getBestFitness(population), mutation_probability)  # debug

            if(previousBestFitness == currentBestFitness):
                mutation_probability += 1
            else:
                mutation_probability = 5

            new_population = []
            

            for i in population:
                #select 2 fit parents randomly based on probability with fitness as weights
                x, y = self.randomSelectFitParent(population)

                # crossover
                child = self.reproduce(x, y)

                #mutate this child #todo 0-1 probability
                # mutate_flag = random.choices([0,1], weights = [1/generation, generation], k=1)

                mutate_flag = random.choices([0, 1], weights=[100-mutation_probability, mutation_probability], k=1)
                if(mutate_flag[0]):
                    child = self.mutate(child)

                new_population.append(child)

            x_generation.append(generation)
            y_best_fitness.append(self.getBestFitness(population))

            previousBestFitness = currentBestFitness
            population = self.selectNewPopulation(population, new_population)
            # population = new_population
            currentBestFitness = self.getBestFitness(population)

            generation += 1

        
        print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
        # print("Average Fitness: ", self.getAverageFitness(population))
        # print("Fitness Values of Population:")
        # print([x.fitness for x in population],'\n') #debug
        # # print(generation, [x.fitness for x in population], mutation_probability) #debug
        x_generation.append(generation)
        y_best_fitness.append(self.getBestFitness(population))
        plt.title('GA Performance Graph for Eight Queens Problem')
        plt.xlabel('Number of Generations')
        plt.ylabel('Best Fitness Value')
        plt.plot(x_generation, y_best_fitness,label = "Improved")
        plt.legend()
        # plt.show()
        print("Best Solution: ", self.getBestIndividual(population))
        




class Tour():
        
    data = np.array([
        [0, inf, inf, inf, inf, inf, 0.15, inf, inf, 0.2, inf, 0.12, inf, inf], #A
        [inf, 0, inf, inf, inf, inf, inf, 0.19, 0.4, inf, inf, inf, inf, 0.13], #B
        [inf, inf, 0, 0.6, 0.22, 0.4, inf, inf, 0.2, inf, inf, inf, inf, inf], #C
        [inf, inf, 0.6, 0, inf, 0.21, inf, inf, inf, inf, 0.3, inf, inf, inf], #D
        [inf, inf, 0.22, inf, 0, inf, inf, inf, 0.18, inf, inf, inf, inf, inf], #E
        [inf, inf, 0.4, 0.21, inf, 0, inf, inf, inf, inf, 0.37, 0.6, 0.26, 0.9], #F
        [0.15, inf, inf, inf, inf, inf, 0, inf, inf, inf, 0.55, 0.18, inf, inf], #G
        [inf, 0.19, inf, inf, inf, inf, inf, 0, inf, 0.56, inf, inf, inf, 0.17], #H
        [inf, 0.4, 0.2, inf, 0.18, inf, inf, inf, 0, inf, inf, inf, inf, 0.6], #I
        [0.2, inf, inf, inf, inf, inf, inf, 0.56, inf, 0, inf, 0.16, inf, 0.5], #J
        [inf, inf, inf, 0.3, inf, 0.37, 0.55, inf, inf, inf, 0, inf, 0.24, inf], #K
        [0.12, inf, inf, inf, inf, 0.6, 0.18, inf, inf, 0.16, inf, 0, 0.4, inf], #L
        [inf, inf, inf, inf, inf, 0.26, inf, inf, inf, inf, 0.24, 0.4, 0, inf], #M
        [inf, 0.13, inf, inf, inf, 0.9, inf, 0.17, 0.6, 0.5, inf, inf, inf, 0] #N
    ]
    )

    distance = pd.DataFrame(data, index = list("ABCDEFGHIJKLMN"), columns = list("ABCDEFGHIJKLMN"))
    # print(distance) #debug

    def __init__(self, state):
        self.state = state
        self.fitness = self.getFitness()

    def getFitness(self):
        s = self.state
        n = len(s)
        totalDistance = 0

        for i in range(n):
            fromCity = s[i]
            toCity = ''
            if i == n-1:
                toCity = s[0]
            else:
                toCity = s[i+1]

            totalDistance += self.distance[fromCity][toCity]
        
        return (1/totalDistance)


class BasicTSP:
    def __init__(self):
        self.main(20)
            
    def init_population(self, k):
        return [Tour("ABCDEFGHIJKLMN")]*k

    def randomSelectFitParent(self, population):
        total_fitness = 0
        
        for x in population:
            total_fitness += x.fitness
        # np.random.choice(a, size=None, replace=True, p=)
        weights = [x.fitness/total_fitness for x in population]

        fit_parents =  np.random.choice(population, size=2, replace=False, p=weights)
        # fit_parents = random.choices(population, weights = weights, k=2)
        return fit_parents[0], fit_parents[1]

    def reproduce(self,x,y):
        n = len(x.state)
        i = random.randint(0,n-1)
        j = random.randint(i+1,n)
        new_state = [None]*n

        new_state[i:j] = list(x.state[i:j]) #copy subtour from x's state/path

        for i in y.state:
            if i not in new_state: #if y's city not present in new state then find position and copy there
                for j in range(len(new_state)):
                    if new_state[j] == None:
                        new_state[j] = i
                        break

        new_state = ''.join(new_state)
        child = Tour(new_state)
        return child

    def mutate(self, child):
        n = len(child.state) #swap-mutation
        i = random.randint(0,n-1)
        j = random.randint(0,n-1)
        temp = list(child.state)
        t = temp[i]
        temp[i] = temp[j]
        temp[j] = t
        new_state = ''
        for i in temp:
            new_state += str(i)
        return Tour(new_state)  


    def getBestFitness(self, population):
        return max([x.fitness for x in population])        

        
    def getAverageFitness(self, population):
        n = len(population)
        total_fitness = 0
        
        for x in population:
            total_fitness += x.fitness
        
        return total_fitness/n       

    # def selectNewPopulation(population, new_population):
    #     # print([x.fitness for x in population])
    #     # print([x.fitness for x in new_population])
    #     total_population = population + new_population
    #     # print([x.fitness for x in total_population])
    #     n = len(population)
    #     newlist = sorted(total_population, key= lambda x: x.fitness, reverse=True)
    #     # print([x.fitness for x in newlist])
    #     n1 = math.floor((0.80)*n) + 1 #n1 best n2 probability wise
    #     n2 = n-n1
    #     list1 = newlist[:n1]
    #     list2 = newlist[n1:] 
    #     list2 = random.choices(list2, k=n2)
    #     return (list1 + list2)
        
        # weights = [x.fitness for x in total_population]
        # new_population = random.choices(total_population, weights = weights, k = n)
    def getBestIndividual(self, population):
        bestState = sorted(population, key= lambda x: x.fitness, reverse=True)[0]
        return bestState.state  
    
    def main(self, k):
        population = []
        generation  = 1
        max_generation = 10000
        x_generation = []
        y_best_fitness = []

        population = self.init_population(k)

        while generation != max_generation:
            
         
            print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
            # print("Average Fitness: ", self.getAverageFitness(population))
            # print("Fitness Values of Population:")
            # print([x.fitness for x in population],'\n') #debug

            new_population = []
            for i in range(k):
                #randomly select parents
                x, y = self.randomSelectFitParent(population)

                #cross-over
                child = self.reproduce(x,y)

                #mutate this child #todo 0-1 probability
                mutate_flag = random.choices([0,1], weights = [95,5], k=1)
                if(mutate_flag[0]):
                    child = self.mutate(child)
                
                new_population.append(child)
            
            # print([x.state for x in population], getAverageFitness(population), getBestFitness(population))
            # print([x.fitness for x in population]) #debug
            x_generation.append(generation)
            y_best_fitness.append(self.getBestFitness(population))
            population = new_population
            generation +=1
        
        

        print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
        # print("Average Fitness: ", self.getAverageFitness(population))
        # print("Fitness Values of Population:")
        # print([x.fitness for x in population],'\n') #debug
        
        x_generation.append(generation)
        y_best_fitness.append(self.getBestFitness(population))
        plt.title('GA for TSP')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.plot(x_generation, y_best_fitness, label = "Basic")
        plt.legend()
        # plt.show()
        print("Best Solution: ", self.getBestIndividual(population))

        
        







class ImprovedTSP:
    def __init__(self):
        self.main(20)

    def init_population(self, k):
        return [Tour("ABCDEFGHIJKLMN")]*k

    # def randomSelectFitParent(population):
    #     total_fitness = 0
        
    #     for x in population:
    #         total_fitness += x.fitness
        
    #     weights = [x.fitness for x in population]
    #     fit_parents = random.choices(population, weights = weights, k=2)
    #     return fit_parents[0], fit_parents[1]

    def randomSelectFitParent(self, population):
        participants = random.sample(population, TOURNAMENT_SIZE)
        # return fit_parents[0], fit_parents[1]
        parents = sorted(participants, key= lambda x: x.fitness, reverse=True)
        return parents[0], parents[1]

    # def randomSelectFitParent(self, population):
    #     total_fitness = 0
        
    #     for x in population:
    #         total_fitness += x.fitness
    #     # np.random.choice(a, size=None, replace=True, p=)
    #     weights = [x.fitness/total_fitness for x in population]

    #     fit_parents =  np.random.choice(population, size=2, replace=False, p=weights)
    #     # fit_parents = random.choices(population, weights = weights, k=2)
    #     return fit_parents[0], fit_parents[1]
            
    def reproduce(self, x,y):
        n = len(x.state)
        i = random.randint(0,n-1)
        j = random.randint(i+1,n)
        new_state = [None]*n

        new_state[i:j] = list(x.state[i:j]) #copy subtour from x's state/path

        for i in y.state:
            if i not in new_state: #if y's city not present in new state then find position and copy there
                for j in range(len(new_state)):
                    if new_state[j] == None:
                        new_state[j] = i
                        break

        new_state = ''.join(new_state)
        child = Tour(new_state)
        # return child
        if(child.fitness > max(x.fitness,y.fitness)): #improvement-1
            return child
        else:
            if(x.fitness > y.fitness):
                return x
            else:
                return y
        

    # def mutate(self, child):
    #     n = len(child.state) #swap-mutation
    #     i = random.randint(0,n-1)
    #     j = random.randint(0,n-1)
    #     temp = list(child.state)
    #     t = temp[i]
    #     temp[i] = temp[j]
    #     temp[j] = t
    #     new_state = ''
    #     for i in temp:
    #         new_state += str(i)
    #     return Tour(new_state)  

    def swapped_child(self, child, i , j):
        n = len(child.state) #swap-mutation
        temp = list(child.state)
        t = temp[i]
        temp[i] = temp[j]
        temp[j] = t
        new_state = ''
        for i in temp:
            new_state += str(i)
        return Tour(new_state)  





    def mutate(self, child):
        max_child = child
        n = len(child.state) #swap-mutation
        max_fitness = child.fitness

        for i in range(n):
            j = i+1
            while(j < n):
                temp_child = self.swapped_child(child, i, j)        
                if temp_child.fitness > max_fitness:
                    max_fitness = temp_child.fitness
                    max_child = temp_child

                j+=1

        if max_child == child:
            i = random.randint(0,n-1)
            j = random.randint(0,n-1)
            temp = list(child.state)
            t = temp[i]
            temp[i] = temp[j]
            temp[j] = t

            # i = random.randint(0,n-1)
            # j = random.randint(0,n-1)
            # t = temp[i]
            # temp[i] = temp[j]
            # temp[j] = t
            new_state = ''
            for i in temp:
                new_state += str(i)
            return Tour(new_state)

        return max_child  

    def getBestFitness(self, population):
        return max([x.fitness for x in population])        

        
    def getAverageFitness(self, population):
        n = len(population)
        total_fitness = 0
        
        for x in population:
            total_fitness += x.fitness
        
        return total_fitness/n       

    # def selectNewPopulation(population, new_population):
    #     total_population = population + new_population
    #     n = len(population)
    #     newlist = sorted(total_population, key= lambda x: x.fitness, reverse=True)
    #     n1 = math.floor((0.80)*n) + 1 #n1 best n2 probability wise
    #     n2 = n-n1
    #     list1 = newlist[:n1]
    #     list2 = newlist[n1:] 
    #     list2 = random.choices(list2, k=n2)
    #     return (list1 + list2)
        
        # weights = [x.fitness for x in total_population]
        # new_population = random.choices(total_population, weights = weights, k = n)

    def selectNewPopulation(self, population, new_population):
        
        n = len(population)
        
        # print([x.fitness for x in newlist])
        n1 = math.floor((0.80)*n) + 1 #n1 best n2 probability wise
        n2 = n-n1
        sorted_new_population = sorted(new_population, key= lambda x: x.fitness, reverse=True)
        list1 = sorted_new_population[:n1]
        list2 = random.choices(population, k=n2)
        return (list1 + list2)

    def getBestIndividual(self, population):
        bestState = sorted(population, key= lambda x: x.fitness, reverse=True)[0]
        return bestState.state

    def main(self, k):
        population = []
        generation  = 1
        max_generation = 8000
        x_generation = []
        y_best_fitness = [] #index = gen-2
        count = 0
        mutation_probability = 5

        population = self.init_population(k)
        x_generation.append(generation)
        y_best_fitness.append(self.getBestFitness(population))

        while max_generation != generation and count != 100:

            if(y_best_fitness[generation-1] > 1e-6 and (y_best_fitness[generation-1] == y_best_fitness[generation-2])):
                count += 1
            else:
                count = 0

            if(y_best_fitness[generation-1] < 1e-6 or (y_best_fitness[generation-1] == y_best_fitness[generation-2])):
                if(mutation_probability<= 98):
                    mutation_probability += 1
            else:
                mutation_probability = 5

            
            print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
            # print("Average Fitness: ", self.getAverageFitness(population))
            # print("Mutation Probabiity: ", mutation_probability)
            # print("Fitness Values of Population:")
            # print([x.fitness for x in population],'\n') #debug
            # # print(generation ,[x.state for x in population], self.getAverageFitness(population), self.getBestFitness(population), mutation_probability, count) #debug

            new_population = []

            for i in range(k):
                #randomly select parents
                x, y = self.randomSelectFitParent(population)

                #cross-over
                child = self.reproduce(x,y)

                #mutate this child #todo 0-1 probability
                mutate_flag = random.choices([0,1], weights = [100-mutation_probability,mutation_probability], k=1)
                if(mutate_flag[0]):
                    child = self.mutate(child)
                
                new_population.append(child)
            
            
            # print([x.fitness for x in population]) #debug
            population = self.selectNewPopulation(population, new_population)
            # population = new_population
            generation +=1
            x_generation.append(generation)
            y_best_fitness.append(self.getBestFitness(population))
        
        
        
        print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
        # print("Average Fitness: ", self.getAverageFitness(population))
        # print("Mutation Probabiity: ", mutation_probability)
        # print("Fitness Values of Population:")
        # print([x.fitness for x in population],'\n') #debug
        
        x_generation.append(generation)
        y_best_fitness.append(self.getBestFitness(population))
        plt.title('Improved GA Performance Graph for TSP')
        plt.xlabel('Number of Generations')
        plt.ylabel('Best Fitness Value')
        plt.plot(x_generation, y_best_fitness, label="Improved")
        plt.legend()
        # plt.show()
        print("Best Solution: ", self.getBestIndividual(population))
            
        
# class ImprovedTSP:
#     def __init__(self):
#         self.main(20)

#     def init_population(self, k):
#         return [Tour("ABCDEFGHIJKLMN")]*k

#     # def randomSelectFitParent(population):
#     #     total_fitness = 0
        
#     #     for x in population:
#     #         total_fitness += x.fitness
        
#     #     weights = [x.fitness for x in population]
#     #     fit_parents = random.choices(population, weights = weights, k=2)
#     #     return fit_parents[0], fit_parents[1]


#     def randomSelectFitParent(self, population):
#         participants = random.sample(population, TOURNAMENT_SIZE)
#         # return fit_parents[0], fit_parents[1]
#         parents = sorted(participants, key= lambda x: x.fitness, reverse=True)
#         return parents[0]

#     def reproduce(self, x,y):
#         n = len(x.state)
#         i = random.randint(0,n-1)
#         j = random.randint(i+1,n)
#         new_state = [None]*n

#         new_state[i:j] = list(x.state[i:j]) #copy subtour from x's state/path

#         for i in y.state:
#             if i not in new_state: #if y's city not present in new state then find position and copy there
#                 for j in range(len(new_state)):
#                     if new_state[j] == None:
#                         new_state[j] = i
#                         break

#         new_state = ''.join(new_state)
#         child = Tour(new_state)
#         return child
#         # if(child.fitness > max(x.fitness,y.fitness)): #improvement-1
#         #     return child
#         # else:
#         #     if(x.fitness > y.fitness):
#         #         return x
#         #     else:
#         #         return y
        

#     # def mutate(self, child):
#     #     n = len(child.state) #swap-mutation
#     #     i = random.randint(0,n-1)
#     #     j = random.randint(0,n-1)
#     #     temp = list(child.state)
#     #     t = temp[i]
#     #     temp[i] = temp[j]
#     #     temp[j] = t
#     #     new_state = ''
#     #     for i in temp:
#     #         new_state += str(i)
#     #     return Tour(new_state)  

#     # def swapped_child(self, child, i , j):
#     #     n = len(child.state) #swap-mutation
#     #     temp = list(child.state)
#     #     t = temp[i]
#     #     temp[i] = temp[j]
#     #     temp[j] = t
#     #     new_state = ''
#     #     for i in temp:
#     #         new_state += str(i)
#     #     return Tour(new_state)  





#     def mutate(self, child):
#         n = len(child.state) #swap-mutation
#         i = random.randint(0,n-1)
#         j = random.randint(0,n-1)
#         temp = list(child.state)
#         t = temp[i]
#         temp[i] = temp[j]
#         temp[j] = t
#         new_state = ''
#         for i in temp:
#             new_state += str(i)
#         return Tour(new_state)  

#     def getBestFitness(self, population):
#         return max([x.fitness for x in population])        

        
#     def getAverageFitness(self, population):
#         n = len(population)
#         total_fitness = 0
        
#         for x in population:
#             total_fitness += x.fitness
        
#         return total_fitness/n       

#     # def selectNewPopulation(population, new_population):
#     #     total_population = population + new_population
#     #     n = len(population)
#     #     newlist = sorted(total_population, key= lambda x: x.fitness, reverse=True)
#     #     n1 = math.floor((0.80)*n) + 1 #n1 best n2 probability wise
#     #     n2 = n-n1
#     #     list1 = newlist[:n1]
#     #     list2 = newlist[n1:] 
#     #     list2 = random.choices(list2, k=n2)
#     #     return (list1 + list2)
        
#         # weights = [x.fitness for x in total_population]
#         # new_population = random.choices(total_population, weights = weights, k = n)

#     def selectNewPopulation(self, population, new_population):
        
#         n = len(population)
        
#         # print([x.fitness for x in newlist])
#         n1 = math.floor((0.80)*n) + 1 #n1 best n2 probability wise
#         n2 = n-n1
#         sorted_new_population = sorted(new_population, key= lambda x: x.fitness, reverse=True)
#         list1 = sorted_new_population[:n1]
#         list2 = random.choices(population, k=n2)
#         return (list1 + list2)

#     def getBestIndividual(self, population):
#         bestState = sorted(population, key= lambda x: x.fitness, reverse=True)[0]
#         return bestState.state

#     def main(self, k):
#         population = []
#         generation  = 1
#         max_generation = 8000
#         x_generation = []
#         y_best_fitness = [] #index = gen-2
#         # count = 0
#         mutation_probability = 20

#         population = self.init_population(k)
#         x_generation.append(generation)
#         y_best_fitness.append(self.getBestFitness(population))

#         while max_generation != generation:

            
#             # if(y_best_fitness[generation-1] < 1e-6 or (y_best_fitness[generation-1] == y_best_fitness[generation-2])):
#             #     if(mutation_probability<= 98):
#             #         mutation_probability += 1
#             # # elif y_best_fitness[generation-1] > 1e-6:
#             # #     mutation_probability += 0.5
#             # else:
#             #     mutation_probability = 5

            
#             print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
#             print("Average Fitness: ", self.getAverageFitness(population))
#             print("Mutation Probabiity: ", mutation_probability)
#             print("Fitness Values of Population:")
#             print([x.fitness for x in population],'\n') #debug
#             # print(generation ,[x.state for x in population], self.getAverageFitness(population), self.getBestFitness(population), mutation_probability, count) #debug

#             new_population = []

#             for i in range(k):
#                 #randomly select parents
#                 x = self.randomSelectFitParent(population)
#                 y = self.randomSelectFitParent(population)

#                 #cross-over
#                 child = self.reproduce(x,y)

#                 #mutate this child #todo 0-1 probability
#                 mutate_flag = random.choices([0,1], weights = [100-mutation_probability,mutation_probability], k=1)
#                 if(mutate_flag[0]):
#                     child = self.mutate(child)
                
#                 new_population.append(child)
            
            
#             # print([x.fitness for x in population]) #debug
#             # population = self.selectNewPopulation(population, new_population)
#             population = new_population
#             generation +=1
#             x_generation.append(generation)
#             y_best_fitness.append(self.getBestFitness(population))
        
        
        
#         print("Generation: ", generation, " Best Fitness: ", self.getBestFitness(population))
#         print("Average Fitness: ", self.getAverageFitness(population))
#         print("Mutation Probabiity: ", mutation_probability)
#         print("Fitness Values of Population:")
#         print([x.fitness for x in population],'\n') #debug
        
#         x_generation.append(generation)
#         y_best_fitness.append(self.getBestFitness(population))
#         plt.title('Improved GA performance for TSP')
#         plt.xlabel('Number of Generations')
#         plt.ylabel('Best Fitness Value')
#         plt.plot(x_generation, y_best_fitness, label="Improved")
#         plt.legend()
#         # plt.show()
#         return generation, self.getBestFitness(population), self.getBestIndividual(population)
            
        

    


if __name__ == '__main__':
    problem = input("Select algorithm for execution:\n")
    if(problem == 'EightQueens'):
        solution = BasicEightQueens()
        # solution = ImprovedEightQueens()
        plt.show()
    elif(problem == 'TSP'):
        solution = BasicTSP()
        # solution = ImprovedTSP()
        # print(seed)
        plt.show()

    else:
        print("Please select correct algorithm\n")



####################################################
#change tsp stopping criterion to num of generations
