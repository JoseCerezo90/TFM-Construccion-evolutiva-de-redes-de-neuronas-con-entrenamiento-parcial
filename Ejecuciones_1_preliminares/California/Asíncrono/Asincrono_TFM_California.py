import copy
import time
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randrange
from tensorflow import keras
from collections import Counter
from numpy import savetxt, loadtxt

total_epochs = 0

####################
# CLASS INDIVIDUAL #
####################

class Individual:
    # Constructor
    def __init__(self, axiom, grammar, non_terminals, max_depth, current_depth, use_GUPI, starting, nn_data):
        self.axiom = axiom
        self.grammar = grammar
        self.non_terminals = non_terminals
        self.initial_depth = 0
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.representation = []
        self.nn_data = nn_data
        if use_GUPI:
            self.representation = self.__generate_derivation_GUPI(self.axiom, self.max_depth, self.current_depth, starting)
        else:
            while self.representation == []:
                self.representation = self.__generate_derivation()
        self.decoded = self.__decode_individual()
        self.neural_model = FFNN(encoded_network=self.decoded, nn_data=self.nn_data)
        self.fitness, _ = self.neural_model.run_and_test(1)
        self.mature = False
        
    # Given the current derivation tree, find the next non terminal (the first leftmost)
    def __find_next_non_terminal(self, derivation_tree):
        # Splits the derivation tree symbol by symbol
        derivation_tree = [character for character in derivation_tree]
        for symbol in derivation_tree:
            if symbol in self.non_terminals:
                return symbol
        # There are no non_terminals
        return ''

    # Modifies the derivation tree applying the new derivation
    def __apply_derivation(self, derivation_tree, new_derivation, non_terminal): 
        new_derivation_tree = ''
        derivation_done = False
        for i in range(len(derivation_tree)):
            if derivation_tree[i] == non_terminal and not derivation_done:
                new_derivation_tree += new_derivation
                derivation_done = True
            else:
                new_derivation_tree += derivation_tree[i]
        return new_derivation_tree

    ##########################################################
    # Grammatically uniform population initialization (GUPI) #
    ##########################################################

    def __has_non_terminal_symbols(self, derivation):
        for non_terminal in self.non_terminals:
            if non_terminal in derivation:
                return True
        return False
    
    def __is_recursive(self, non_terminal, derivation):
        return non_terminal in derivation
    
    def __get_non_terminals(self, derivation):
        list = []
        for i in range(len(derivation)):
            if derivation[i] in self.non_terminals:
                list.append(derivation[i])
        return list

    def __get_rule_using_probs(self, probs):
        r = random.uniform(0, 1)
        accum_probs = 0
        for i in range(len(probs)):
            accum_probs += probs[i]
            if r <= accum_probs:
                return i

    # Helper function of "get_non_terminal_length" and "get_rule_length"
    def __obtain_number_combinations(self, combinations):
        exit = True
        while exit:
            new_combinations = []
            for i in range(len(combinations)):
                non_terminals_to_derivate = self.__get_non_terminals_with_pos(combinations[i][0])
                if non_terminals_to_derivate == []:
                    new_combinations.append(combinations[i])
                else:
                    for (j, pos) in non_terminals_to_derivate:
                        for k in range(len(self.grammar.get(j))):
                            derivation = self.__apply_derivation_with_pos(combinations[i][0], self.grammar.get(j)[k], j, pos)
                            if self.__is_recursive(j, self.grammar.get(j)[k]):
                                # If it does not exceed rho, add
                                if (combinations[i][1]-1) >= 0:
                                    new_combinations.append((derivation, combinations[i][1]-1))
                            else:
                                new_combinations.append((derivation, combinations[i][1]))
            # No more derivations available
            if set(new_combinations) == set(combinations): exit = False
            else: combinations = copy.copy(list(set(new_combinations)))
        return len(combinations)

    # Helper function of "get_non_terminal_length"
    def __get_non_terminals_with_pos(self, derivation):
        list = []
        for i in range(len(derivation)):
            if derivation[i] in self.non_terminals:
                list.append((derivation[i], i))
        return list

    # Helper function of "get_non_terminal_length"
    def __apply_derivation_with_pos(self, derivation_tree, new_derivation, non_terminal, pos): 
        new_derivation_tree = ''
        derivation_done = False
        for i in range(len(derivation_tree)):
            if derivation_tree[i] == non_terminal and not derivation_done and i == pos:
                new_derivation_tree += new_derivation
                derivation_done = True
            else:
                new_derivation_tree += derivation_tree[i]
        return new_derivation_tree 

    # Calculates the length of a non-terminal (corresponds to the denominator of formula 4 of the GUPI paper)
    def __get_non_terminal_length(self, non_terminal, rho):
        # 'H' can only result in '1H' or '1', being the first recursive, therefore, the length of 'H' 
        # is given by the number of recursions (1H) that it can perform (rho) + apply the non-recursive
        if non_terminal == 'H':
            return rho+1
        # 'L' can lead to 'H0L' and 'H', the former being recursive
        else:
            # If recursion is 0, the only possible derivation is: L -> H; H -> 1
            if rho == 0:
                return 1
            # Get all possible derivations taking into account rho
            else:
                combinations = [('H0L', rho-1), ('H', rho)]
                return self.__obtain_number_combinations(combinations)

    # Calculates the length of a derivation (corresponds to the numerator of formula 4 of the GUPI paper)
    def __get_rule_length(self, derivation, rho):
        if derivation == '1':
            return 1
        elif derivation == 'H':
            return rho+2
        else:
            return self.__obtain_number_combinations([(derivation, rho)])

    # Uses the grammar to generate a derivation tree 
    def __generate_derivation_GUPI(self, non_terminal, rho, current_depth, starting):
        # The neural network must have at least input and output layer (so the first derivation is: L -> H0L)
        if starting:
            return [(0, current_depth)] + self.__generate_derivation_GUPI('H', rho-1, current_depth+1, False) + \
            self.__generate_derivation_GUPI('L', rho-1, current_depth+1, False)
        else:
            # Line 5 & 6 (GUPI paper pseudocode)
            if non_terminal not in self.non_terminals:
                return []
            # Line 7 (GUPI paper pseudocode)
            probs = []
            denominator = self.__get_non_terminal_length(non_terminal, rho)
            for i in range(len(self.grammar.get(non_terminal))):
                if rho == 0 and self.__is_recursive(non_terminal, self.grammar.get(non_terminal)[i]):
                    probs.append(0) 
                else:
                    prob = self.__get_rule_length(self.grammar.get(non_terminal)[i], rho-1) / denominator
                    probs.append(prob)
            # Line 8 (GUPI paper pseudocode)
            n_rule = self.__get_rule_using_probs(probs)
            derivation = self.grammar.get(non_terminal)[n_rule]
            # Line 9 & 10 (GUPI paper pseudocode) 
            if not self.__has_non_terminal_symbols(derivation):
                return [(n_rule, current_depth)] + self.__generate_derivation_GUPI('', rho, current_depth+1, False)
            # Line 11 & 12 (GUPI paper pseudocode) 
            if self.__is_recursive(non_terminal, derivation):
                rho -= 1
            # Line 13-18 (GUPI paper pseudocode) 
            B = self.__get_non_terminals(derivation)
            if len(B) == 1:
                return [(n_rule, current_depth)] + self.__generate_derivation_GUPI(B[0], rho, current_depth+1, False)
            elif len(B) == 2:
                return [(n_rule, current_depth)] + self.__generate_derivation_GUPI(B[0], rho, current_depth+1, False) + \
                    self.__generate_derivation_GUPI(B[1], rho, current_depth+1, False)

    #########################
    # Random inicialization #
    #########################

    # Uses the grammar to generate a derivation tree 
    def __generate_derivation(self):
        it = 0
        representation = []
        derivation_tree = self.axiom
        current_depth = self.initial_depth
        depth_stack = []
        non_terminal = self.__find_next_non_terminal(derivation_tree)
        while it < self.max_depth:
            # Next derivation to be applied to the current non_terminal symbol (I force there to be at least two layers (input and output))
            if it == 0:
                derivation_number = 0
            else:
                derivation_number = randrange(len(self.grammar.get(non_terminal)))
            # The representation resulted the depth of the node in the derivation tree
            representation.append((derivation_number, current_depth))
            current_depth += 1
            new_derivation = self.grammar.get(non_terminal)[derivation_number]
            # If the new derivation contains more than one nonterminal, 
            # the current depth is stored for when the next nonterminal is derived.
            if new_derivation == 'H0L':
                depth_stack.append(current_depth)
            # If the new derivation only contains terminals, we will continue deriving non-terminals 
            # that could not be derived before (case above), so we must recover the depth in which they were found
            if new_derivation == '1' and len(depth_stack) > 0:
                current_depth = depth_stack.pop()
            derivation_tree = self.__apply_derivation(derivation_tree, new_derivation, non_terminal)
            non_terminal =  self.__find_next_non_terminal(derivation_tree)
            it += 1
            if non_terminal == '':
                return representation
        return []

    # Decoding an individual means getting its leaf nodes, i.e. when there are no nonterminals left
    def __decode_individual(self):
        derivation_tree = self.axiom
        non_terminal = self.__find_next_non_terminal(derivation_tree)
        for (derivation_number, _) in self.representation:
            new_derivation = self.grammar.get(non_terminal)[derivation_number]
            derivation_tree = self.__apply_derivation(derivation_tree, new_derivation, non_terminal)
            non_terminal =  self.__find_next_non_terminal(derivation_tree)
            if non_terminal == '':
                return derivation_tree

    def set_representation(self, new_representation):
        self.mature = False
        self.representation = new_representation
        self.decoded = self.__decode_individual()
        self.neural_model = FFNN(encoded_network=self.decoded, nn_data=self.nn_data)
        self.fitness, _ = self.neural_model.run_and_test(1)

###################
# CLASS EVOLUTION #
###################

class Evolution:
    # Constructor
    def __init__(self, axiom, grammar, non_terminals, max_depth, current_depth, population_size, selection_percentage, 
    crossover_prob, mutation_prob, use_GUPI, max_iterations, nn_data, use_elitist_repl):
        self.max_iterations = max_iterations
        self.use_GUPI = use_GUPI
        self.use_elitist_repl = use_elitist_repl
        self.nn_data = nn_data
        self.axiom = axiom
        self.grammar = grammar
        self.non_terminals = non_terminals
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.population = []
        self.population_size = population_size
        self.individuals_to_select = round(population_size*selection_percentage)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.it_without_improve = 0
        self.sons = []
        _, OUTPUTS, _, _, _, _, _, _ = self.nn_data
        self.OUTPUTS = OUTPUTS
        # Variables to store data and generate graphs
        self.array_best_fitness = []
        self.array_average_fitness = []

    def __debug(self, it, debug_metric):
        best_individual = self.get_best_individual()
        fitness_average = self.__get_fitness_average()

        self.array_best_fitness.append(best_individual.fitness)
        self.array_average_fitness.append(fitness_average)

        print("Iteration " + str(it) + ". Best individual: " + str(best_individual.decoded) + ". Best fitness: " + \
                str(best_individual.fitness) + ". Population average: " + str(fitness_average) + debug_metric)
        """print("### Iteration " + str(it) + " ###")
        for i in range(self.population_size):
            print("Individual " + str(i) + ": " + str(self.population[i].decoded) + ". Fitness: " + str(self.population[i].fitness))
        print("** Best individual: " + str(best_individual.decoded) + ". Best fitness: " + \
                str(best_individual.fitness) + ". Population average: " + str(self.__get_fitness_average()) + debug_metric)
        print("")"""
        return fitness_average

    # Main function that will carry out the entire evolutionary process
    def evolve(self):
        if self.OUTPUTS == 1: debug_metric = " (MSE)"
        else: debug_metric = " (Accuracy)"
        self.__generate_random_population()
        it = 0
        last_fitness_average = 0
        fitness_average = self.__debug(it, debug_metric)
        while not self.__stop_condition(it, fitness_average, last_fitness_average):
            last_fitness_average = fitness_average
            it += 1
            self.sons = []
            # Selection
            selected = self.__roulette_wheel_selection()
            # Crossover
            for i in range(int(len(selected)/2)):
                if random.uniform(0, 1) <= self.crossover_prob:
                    # Find 2 parents among those selected
                    selected_index = randrange(len(selected))
                    parent_1 = selected[selected_index]
                    selected.pop(selected_index)
                    selected_index = randrange(len(selected))
                    parent_2 = selected[selected_index]
                    selected.pop(selected_index)
                    # Make the crossing
                    individual_son_1, individual_son_2 = self.__crossover(parent_1, parent_2)
                    if individual_son_1 != [] and individual_son_2 != []:
                        self.sons.append(individual_son_1)
                        self.sons.append(individual_son_2)
            # Mutation
            self.__mutation()
            # Replacement
            self.__replacement_SSGA()
            
            # Training cycle of the entire population
            for i in range(self.population_size):
                # Train if it does not reach the stop condition
                if self.population[i].neural_model.current_epochs < self.population[i].neural_model.n_epochs:
                    last_fitness = self.population[i].fitness
                    self.population[i].fitness, mature = self.population[i].neural_model.run_and_test(last_fitness)
                    if mature: self.population[i].mature = True
                
            fitness_average = self.__debug(it, debug_metric)

    # Generates a "population_size" of random individuals
    def __generate_random_population(self):
        while len(self.population) < self.population_size:
            i = Individual(axiom=self.axiom, grammar=self.grammar, non_terminals=self.non_terminals, max_depth=self.max_depth, 
                          current_depth=self.current_depth, use_GUPI=self.use_GUPI, starting=True, nn_data=self.nn_data)
            self.population.append(i)

    # Create a dummy son and insert a representation (encoding) into it
    def __create_son(self, son_representation):
        son = Individual(axiom=self.axiom, grammar=self.grammar, non_terminals=self.non_terminals, max_depth=1, 
                          current_depth=self.current_depth, use_GUPI=self.use_GUPI, starting=True, nn_data=self.nn_data)
        son.set_representation(son_representation)
        return son

    def __get_matures(self, mat):
        matures = []
        for i in range(self.population_size):
            # Matures (Selection operator)
            if self.population[i].mature and mat: matures.append(self.population[i])
            # Immatures (Reemplacement operator)
            if not self.population[i].mature and not mat: matures.append(self.population[i])
        return matures
        
    # Selection operator
    def __roulette_wheel_selection(self):
        matures = self.__get_matures(True)
        selected = []
        fitness_list = []
        sum_fitness = 0
        # Probabilities 
        # Regression Problem: Reward the LOWER fitness (MSE) you have
        if self.OUTPUTS == 1:
            for i in range(len(matures)):
                sum_fitness += (1 / matures[i].fitness)
            for i in range(len(matures)):
                fitness_list.append((1/matures[i].fitness) / sum_fitness)
        # Classification Problem: Reward the MORE fitness (accuracy) you have
        else:
            for i in range(len(matures)):
                sum_fitness += matures[i].fitness
            for i in range(len(matures)):
                fitness_list.append(matures[i].fitness / sum_fitness)
        # Roulette partitions
        probs = []
        # Check if there are enough mature individuals to select
        if self.individuals_to_select > len(matures): to_select = len(matures)
        else: to_select = self.individuals_to_select
        for i in range(to_select):
            probs.append((i+1)/(to_select+1))
        # Select the individual that falls in each partition
        accum_fitness = 0
        past_accum_fitness = 0
        for i in range(len(matures)):
            accum_fitness += fitness_list[i]
            for j in range(len(probs)):
                if accum_fitness >= probs[j] and past_accum_fitness < probs[j]:
                    selected.append(matures[i])
            past_accum_fitness += fitness_list[i]
        return selected

    # Given the current derivation tree, find the next non terminal (the first leftmost)
    def __find_next_non_terminal(self, derivation_tree):
        # Splits the derivation tree symbol by symbol
        derivation_tree = [character for character in derivation_tree]
        for symbol in derivation_tree:
            if symbol in self.non_terminals:
                return symbol
        # There are no non_terminals
        return ''

    # Modifies the derivation tree applying the new derivation
    def __apply_derivation(self, derivation_tree, new_derivation, non_terminal): 
        new_derivation_tree = ''
        derivation_done = False
        for i in range(len(derivation_tree)):
            if derivation_tree[i] == non_terminal and not derivation_done:
                new_derivation_tree += new_derivation
                derivation_done = True
            else:
                new_derivation_tree += derivation_tree[i]
        return new_derivation_tree

    # Function used in the crossover and mutation function
    # Extracts the set of all derivations of an individual and its nonterminals
    def __get_non_terminal_nodes(self, individual):
        all_derivations = []
        non_terminals_and_rule = []
        derivation_tree = self.axiom
        non_terminal = self.__find_next_non_terminal(derivation_tree)
        for i in range(len(individual)):
            new_derivation = self.grammar.get(non_terminal)[individual[i][0]]
            derivation_tree = self.__apply_derivation(derivation_tree, new_derivation, non_terminal)
            all_derivations.append(derivation_tree)
            non_terminal = self.__find_next_non_terminal(derivation_tree)
            if (i+1) < len(individual):
                non_terminals_and_rule.append((non_terminal, individual[i+1][0]))
        return non_terminals_and_rule

    def __get_subtree(self, parent, place_parent):
        parent = self.__recalculate_depths(parent)
        end_subtree = -1
        # Start place of subtree in raw string (original individual)
        subtree_start = place_parent + 1
        initial_depth = parent[place_parent+1][1] - 1
        flag = True
        for i in range(place_parent+2, len(parent)):
            if flag and initial_depth >= (parent[i][1]-1):
                end_subtree = i - 1
                flag = False
        # If no end has been found, it is because it reaches the end of the individual
        if end_subtree == -1:
            end_subtree = len(parent)-1
        return subtree_start, end_subtree

    # Function used in the crossover function
    # Extract the set of nonterminal symbols that can be crossed
    def __crossing_nodes(self, non_terminals_1, non_terminals_2):
        no_t_1 = []
        no_t_2 = []
        for i in range(len(non_terminals_1)):
            no_t_1.append(non_terminals_1[i][0])
        for i in range(len(non_terminals_2)):
            no_t_2.append(non_terminals_2[i][0])
        no_t_1 = set(no_t_1)
        no_t_2 = set(no_t_2)
        # Return the intersection of the non-terminals (that can be crossed)
        return no_t_1.intersection(no_t_2)

    def __search_crossing_position(self, symbol_to_cross, non_terminals):
        crossing_position = -1
        while crossing_position == -1:
            ind = randrange(len(non_terminals))
            if non_terminals[ind][0] == symbol_to_cross:
                crossing_position = ind
                return crossing_position

    def __swap_subtrees(self, parent_1, parent_2, subtree_start_1, end_subtree_1, subtree_start_2, end_subtree_2):
        subtree_1 = parent_1[subtree_start_1:end_subtree_1+1]
        subtree_2 = parent_2[subtree_start_2:end_subtree_2+1]
        son_1 = parent_1[:subtree_start_1] + subtree_2 + parent_1[end_subtree_1+1:]
        son_2 = parent_2[:subtree_start_2] + subtree_1 + parent_2[end_subtree_2+1:]
        return self.__recalculate_depths(son_1), self.__recalculate_depths(son_2)

    # It is necessary to recalculate depths when a crossing is made, since the nodes to be crossed may not be at the same height
    def __recalculate_depths(self, individual):
        representation = []
        derivation_tree = self.axiom
        current_depth = 0
        depth_stack = []
        non_terminal = self.__find_next_non_terminal(derivation_tree)
        for (derivation_number, _) in individual:
            representation.append((derivation_number, current_depth))
            current_depth += 1
            new_derivation = self.grammar.get(non_terminal)[derivation_number]
            if new_derivation == 'H0L':
                depth_stack.append(current_depth)
            if new_derivation == '1' and len(depth_stack) > 0:
                current_depth = depth_stack.pop()
            derivation_tree = self.__apply_derivation(derivation_tree, new_derivation, non_terminal)
            non_terminal =  self.__find_next_non_terminal(derivation_tree)
        return representation

    # Whigham crossover
    def __crossover(self, parent_1, parent_2):
        non_terminals_1 = self.__get_non_terminal_nodes(parent_1.representation)
        non_terminals_2 = self.__get_non_terminal_nodes(parent_2.representation)
        # Get the candidate crossing nodes for Whigham
        crossing_nodes = self.__crossing_nodes(non_terminals_1, non_terminals_2)
        # If intersection is empty return null children (crossing not possible)
        if crossing_nodes == set(): return [], []
        # From the intersection set, randomly extract a nonterminal crossing symbol
        symbol_to_cross = list(crossing_nodes)[randrange(len(crossing_nodes))]
        # Find symbol_to_cross in parent_1 & parent_2 (if there are several, get a random one)
        parent_position_1 = self.__search_crossing_position(symbol_to_cross, non_terminals_1) 
        parent_position_2 = self.__search_crossing_position(symbol_to_cross, non_terminals_2)
        # See where each subtree of the junction node starts and ends in each parent
        subtree_start_1, end_subtree_1 = self.__get_subtree(parent_1.representation, parent_position_1)
        subtree_start_2, end_subtree_2 = self.__get_subtree(parent_2.representation, parent_position_2)
        # Perform the crossing
        son_1, son_2 = self.__swap_subtrees(parent_1.representation, parent_2.representation, subtree_start_1, end_subtree_1, subtree_start_2, end_subtree_2)
        # Create the sons (Individual objects) with their representation
        individual_son_1 = self.__create_son(son_1)
        individual_son_2 = self.__create_son(son_2)
        return individual_son_1, individual_son_2

    # Brute force mutation (random)
    def __mutation(self):
        for i in range(len(self.sons)):
            # Mutation occurs
            if random.uniform(0, 1) <= self.mutation_prob:
                # Look for a node on which to perform the mutation (random)
                non_terminals = self.__get_non_terminal_nodes(self.sons[i].representation)
                position_node_to_mutate = randrange(len(self.sons[i].representation)-1)
                # Get the subtree
                subtree_start, end_subtree = self.__get_subtree(self.sons[i].representation, position_node_to_mutate)
                # Generate the subtree from the node to mutate
                axiom = non_terminals[position_node_to_mutate][0]
                current_depth = self.sons[i].representation[position_node_to_mutate][1]+1
                new_max_depth = round(self.max_depth/2)
                mutated_i = Individual(axiom=axiom, grammar=self.grammar, non_terminals=self.non_terminals, max_depth=new_max_depth, 
                                      current_depth=current_depth, use_GUPI=self.use_GUPI, starting=False, nn_data=self.nn_data)
                # Generate the mutated individual
                mutated_representation = self.sons[i].representation[:subtree_start] + mutated_i.representation + \
                    self.sons[i].representation[end_subtree+1:]
                self.sons[i].set_representation(mutated_representation)

    # SSGA
    def __replacement_SSGA(self):
        # Immature individuals cannot be replaced
        new_population = self.__get_matures(False)
        # Children are not replaced either
        for i in range(len(self.sons)):
            new_population.append(self.sons[i])
        remaining_individuals = self.population_size - len(new_population)
        matures = self.__get_matures(True)
        population_eval = []
        # Evaluate mature individuals (they can be replaced)
        for i in range(len(matures)):
            population_eval.append((matures[i].fitness, matures[i]))
        # Sort the population based on fitness (Regression Problem reward the LOWER fitness, but classification Problem reward the MORE fitness)
        if self.OUTPUTS == 1: sorted_population = sorted(population_eval, key=lambda x: x[0], reverse=False)
        else: sorted_population = sorted(population_eval, key=lambda x: x[0], reverse=True)
        # Keep the best individuals
        best_individuals = sorted_population[:remaining_individuals]
        # Add the best individuals
        for i in range(len(best_individuals)):
            new_population.append(best_individuals[i][1])
        self.population = new_population

    # If all individuals are equals, then the population has converged
    def __population_has_converged(self):
        for i in range(1, self.population_size):
            if self.population[i].decoded != self.population[0].decoded:
                return False
        return True
    
    def __get_fitness_average(self):
        accum = 0
        for i in range(self.population_size):
            accum += self.population[i].fitness
        return accum / self.population_size

    def get_best_individual(self):
        best_i = 0
        for i in range(1, self.population_size):
            # Regression Problem: Reward the LOWER fitness (MSE) you have
            if self.OUTPUTS == 1:
                if self.population[i].fitness < self.population[best_i].fitness:
                    best_i = i
            # Classification Problem: Reward the MORE fitness (accuracy) you have
            else:
                if self.population[i].fitness > self.population[best_i].fitness:
                    best_i = i
        return  self.population[best_i]

    def __not_improving(self, fitness_average, last_fitness_average):
        if self.OUTPUTS == 1:
            if fitness_average > last_fitness_average: self.it_without_improve += 1
            else: self.it_without_improve = 0
        else:
            if fitness_average < last_fitness_average: self.it_without_improve += 1
            else: self.it_without_improve = 0
        return self.it_without_improve > 10

    def __stop_condition(self, iteration, fitness_average, last_fitness_average):
        return (iteration >= self.max_iterations) or self.__population_has_converged() or self.__not_improving(fitness_average, last_fitness_average)


########################
# CLASS NEURAL NETWORK #
########################

class FFNN:
    # Constructor
    def __init__(self, encoded_network, nn_data):
        # FFNN parameters
        self.n_epochs = 30
        self.asyn_epochs = 3
        self.current_epochs = 0
        self.batch_size = 32
        self.learning_rate = 0.01
        # Neural network data
        INPUTS, OUTPUTS, x_train, t_train, x_dev, t_dev, x_test, t_test = nn_data
        self.INPUTS = INPUTS
        self.OUTPUTS = OUTPUTS
        self.x_train = x_train
        self.t_train = t_train 
        self.x_dev = x_dev
        self.t_dev = t_dev
        self.x_test = x_test
        self.t_test = t_test
        self.topology = self.__get_topology(encoded_network)
        self.model = self.__build_model()
        
    # Get the number of neurons for each hidden layer
    # "encoded_network" is the decoded string of an individual
    def __get_topology(self, encoded_network):
        topology = []
        num_neurons = 0
        for i in range(len(encoded_network)):
            if encoded_network[i] == '1': 
                num_neurons += 1
            elif encoded_network[i] == '0':
                topology.append(num_neurons)
                num_neurons = 0
        topology.append(num_neurons)
        return topology

    def __build_model(self):
        model = keras.Sequential(name="my_model")
        # Input layer
        model.add(keras.layers.InputLayer(input_shape=(self.INPUTS,)))
        # Hidden layers
        for neurons in self.topology:
            model.add(keras.layers.Dense(neurons, activation="relu"))
        # Regression problem
        if self.OUTPUTS == 1:
            # Output layer
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            # Optimizer
            model.compile(loss='mse',
                        optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True),
                        metrics=['mae', 'mse'])
        # Classification problem
        else:
            # Output layer
            model.add(keras.layers.Dense(self.OUTPUTS, "softmax"))
            # Optimizer
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999),
                        metrics=['categorical_accuracy'])
        return model

    # Using ECO, you have to have control over the number of epochs to train
    def run_and_test(self, last_fitness):
        global total_epochs
        total_epochs += self.asyn_epochs

        self.current_epochs += self.asyn_epochs
        history = self.model.fit(self.x_train, self.t_train, 
                    batch_size = self.batch_size, 
                    epochs = self.asyn_epochs, 
                    verbose = 0, 
                    validation_data = (self.x_dev, self.t_dev))
        # Regression problem
        if self.OUTPUTS == 1: 
            new_fitness = history.history['val_loss'][len(history.history['val_loss'])-1]
            improvement_percentage = ((last_fitness-new_fitness) / last_fitness) * 100
        # Classification problem
        else: 
            new_fitness = history.history['val_categorical_accuracy'][len(history.history['val_categorical_accuracy'])-1]
            improvement_percentage = ((new_fitness-last_fitness) / last_fitness) * 100
        # 5% learning threshold. If the improvement is less, the individual is considered to have matured (slow learning process starts)
        return new_fitness, improvement_percentage < 5
            

########
# MAIN #
########

def prepare_datasets(INPUT_FILE_NAME_1, INPUT_FILE_NAME_2):
    # Network input data
    dataset_attributes = pd.read_csv(INPUT_FILE_NAME_1, sep=",")
    # Network output data
    dataset_classes = pd.read_csv(INPUT_FILE_NAME_2, sep=",")
    # 80% of the data for train, 10% for development and 10% for testing.
    TRAIN_RATIO = 0.8
    n_instances = dataset_attributes.shape[0]
    n_train = int(n_instances*TRAIN_RATIO)
    n_dev = int((n_instances - n_train)/2)
    # Prepare the data sets
    x_train = dataset_attributes.values[:n_train]
    t_train = dataset_classes.values[:n_train]
    x_dev = dataset_attributes.values[n_train:n_train + n_dev]
    t_dev = dataset_classes.values[n_train:n_train + n_dev]
    x_test = dataset_attributes.values[n_train + n_dev:]
    t_test = dataset_classes.values[n_train + n_dev:]
    # Calculate the number of input and output neurons
    INPUTS = x_train.shape[1]
    OUTPUTS = t_train.shape[1]
    return (INPUTS, OUTPUTS, x_train, t_train, x_dev, t_dev, x_test, t_test)

if __name__ == "__main__":
    time_1 = time.time()
    # Grammar used
    axiom = 'L'
    non_terminals = ['L','H']
    grammar = {
        'L':['H0L', 'H'],
        'H':['1H','1']
    }

    # Evolution parameters (the parameters of the neural networks are in the FFNN class)
    max_iterations = 100            	# Maximum iterations allowed for the evolutionary process
    population_size = 40                # Number of individuals
    max_depth = 4                       # GUPI paper recursion (rho)
    current_depth = 0                   # Initial depth from which the trees (individuals) are generated
    selection_percentage = 0.4          # Percentage of individuals, with respect to the population size, that will be selected to subsequently outcrossing
    crossover_prob = 0.9                # Probability of applying the crossover operator
    mutation_prob = 0.1                 # Probability of applying the mutation operator

    # Classification problem
    #file_1 = "./Máster/TFM/Datos/FIFA/Attributes.csv"
    #file_2 = "./Máster/TFM/Datos/FIFA/OneHotEncodedClasses.csv"
    # Classification problem
    #file_1 = "./Máster/TFM/Datos/Iris/Attributes.csv"
    #file_2 = "./Máster/TFM/Datos/Iris/OneHotEncodedClasses.csv"
    # Regression problem
    file_1 = "../../../Máster/TFM/Datos/MedianHouseValue/Attributes.csv"
    file_2 = "../../../Máster/TFM/Datos/MedianHouseValue/ContinuousOutput.csv"
    # Neural networks data
    nn_data = prepare_datasets(file_1, file_2)

    # Generates the initial population (GUPI)
    evo = Evolution(axiom=axiom, grammar=grammar, non_terminals=non_terminals, max_depth=max_depth, current_depth=current_depth, population_size=population_size, 
                    selection_percentage=selection_percentage, crossover_prob=crossover_prob, mutation_prob=mutation_prob, use_GUPI=True, 
                    max_iterations=max_iterations, nn_data=nn_data, use_elitist_repl=True)
    evo.evolve()

    time_2 = time.time()
    
    n_file_1 = './MedianHouseValue_Async_Best_Fit_10.csv'
    n_file_2 = './MedianHouseValue_Async_Aver_Fit_10.csv'
    txt_file = 'MedianHouseValue_Async_10.txt'

    savetxt(n_file_1, evo.array_best_fitness, delimiter=',')
    savetxt(n_file_2, evo.array_average_fitness, delimiter=',')
    # Reading the saved csv
    #data = loadtxt('./data.csv', delimiter=',')
    #new_d = []
    #for i in range(len(data)):
    #    new_d.append(data[i])
    
    cad1 = "Tiempo total ejecución: " + str(datetime.timedelta(seconds=time_2-time_1))
    cad2 = "Épocas totales de entrenamiento: " + str(total_epochs)
    cad3 = "Mejor individuo: " + str(evo.get_best_individual().decoded)
    lines = [cad1, cad2, cad3]
    with open(txt_file, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    print(cad1)
    print(cad2)
    print(cad3)
    