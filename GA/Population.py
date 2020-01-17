import itertools
import numpy as np
from GA.Individual import Individual


class Population():
    def __init__(self, pop_size, idv_size, scalerX, scalerY):
        self.pop = self._get_initial_population(pop_size, idv_size)
        self.pop_size = pop_size
        self.idv_size = idv_size
        self.scalerX = scalerX
        self.scalerY = scalerY

    def cal_fitness(self, dnn):
        confrontations = itertools.permutations(self.pop, 2)
        for confrontation in confrontations:
            self._check_confrontation(confrontation[0], confrontation[1], dnn)
        self.pop.sort(key=lambda idv: idv.fitness, reverse=True)

    def _check_confrontation(self, idv1, idv2, dnn):
        if np.sum(idv1.genome) < 1 and np.sum(idv2.genome) < 1:
            genome1 = np.append(idv1.genome, 1 - np.sum(idv1.genome))
            genome2 = np.append(idv2.genome, 1 - np.sum(idv2.genome))
            capital_change = (genome2 / genome1) - 1
            scaled_capital_change = self.scalerX.transform(capital_change.reshape((1,  55)))
            scaled_capital_change = scaled_capital_change.reshape((1, 55))
            result = dnn.predict(scaled_capital_change)
            scaled_result = self.scalerY.inverse_transform(result)
            if scaled_result > 0:
                idv2.fitness += 1
            elif scaled_result < 0:
                idv1.fitness += 1
        if np.sum(idv1.genome) >= 1:
            idv1.fitness -=  1
        if np.sum(idv2.genome) >= 1:
            idv1.fitness -=  1

    def _get_initial_population(self, pop_size, idv_size):
        pop = []
        for _ in range(pop_size):
            pop.append(Individual(idv_size=idv_size))
        return pop

    def cal_probability(self, alpha):
        # If lowest fitness is a negative number shift all fitness this difference
        if self.pop[-1].fitness < 0:
            for idv in self.pop:
                idv.fitness += abs(self.pop[-1].fitness)
        fitness_sum = 0
        for idv in self.pop:
            fitness_sum += idv.fitness
        fitness_avg = fitness_sum / len(self.pop)
        self._normalize_fitness(fitness_avg, alpha)
        fitness_sum = 0
        for idv in self.pop:
            fitness_sum += idv.fitness
        if fitness_sum != 0:
            for idv in self.pop:
                idv.probability = idv.fitness / fitness_sum
        else:
            for idv in self.pop:
                idv.probability = 1 / len(self.pop)

    def _normalize_fitness(self, fitness_avg, alpha):
        delta = self.pop[0].fitness - fitness_avg
        if delta == 0:
            delta = 1
        a = (fitness_avg*(alpha - 1)) / delta
        b = fitness_avg * (1 - a)
        for idv in self.pop:
            idv.fitness = a*idv.fitness + b

    def get_new_pop(self):
        children = []
        while len(children) < self.pop_size:
            choice = np.random.uniform(0, 1, 2)
            parent_1 = None
            parent_1_call = True
            parent_2 = None
            parent_2_call = True
            prob = 0
            for idv in self.pop:
                prob += idv.probability
                if choice[0] <= prob and parent_1_call:
                    parent_1 = idv
                    parent_1_call = False
                if choice[1] <= prob and parent_2_call:
                    parent_2 = idv
                    parent_2_call = False
                if not parent_1_call and not parent_2_call:
                    break
            if parent_1 is parent_2:
                continue
            else:
                self._add_child_offset(children, parent_1, parent_2)
        self.pop = children

    def _add_child_offset(self, children, parent_1, parent_2):
        offset = np.random.randint(0, len(parent_1.genome[0]))
        childgenome1 = np.concatenate((parent_1.genome[0][:offset], parent_2.genome[0][offset:]))
        child1 = Individual(self.idv_size)
        child1.genome = np.reshape(childgenome1, parent_1.genome.shape)
        children.append(child1)

        childgenome2 = np.concatenate((parent_2.genome[0][:offset], parent_1.genome[0][offset:]))
        child2 = Individual(self.idv_size)
        child2.genome = np.reshape(childgenome2, parent_2.genome.shape)
        children.append(child2)

    def mutate_population(self, mutation_probability):
        for idv in self.pop:
            self._mutate_idv(idv, mutation_probability)

    def _mutate_idv(self, idv, mutation_probability):
        gene_vector = idv.genome
        gene_prob = mutation_probability / len(gene_vector)
        for i in range(len(gene_vector)):
            chance = np.random.uniform(0, 1)
            if chance <= gene_prob:
                gene_vector[i] = np.random.uniform(0, 1)
        idv.genome = gene_vector
