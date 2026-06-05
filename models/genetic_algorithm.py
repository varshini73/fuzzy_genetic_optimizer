"""
Advanced Genetic Algorithm for Pricing Parameter Optimization
Optimizes 8 key parameters for maximum profitability
"""

import numpy as np
import pandas as pd
from collections import defaultdict

class GeneticOptimizer:
    """Genetic Algorithm for price optimization parameters"""
    
    def __init__(self, population_size=100, generations=100, 
                 mutation_rate=0.1, crossover_rate=0.8, elite_size=5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # 8 Parameters to optimize with their bounds
        self.param_bounds = {
            'price_multiplier': (0.7, 1.5),
            'demand_elasticity': (-2.0, -0.1),
            'satisfaction_weight': (0.1, 0.5),
            'competition_factor': (0.5, 1.5),
            'seasonal_adjustment': (0.8, 1.3),
            'profit_margin_target': (0.15, 0.60),
            'risk_tolerance': (0.1, 0.9),
            'volume_sensitivity': (0.5, 2.0)
        }
        
        self.param_names = list(self.param_bounds.keys())
        self.history = {'best_fitness': [], 'avg_fitness': [], 'best_params': []}
        
    def initialize_population(self):
        """Create initial random population"""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.param_bounds.items():
                individual[param] = np.random.uniform(low, high)
            population.append(individual)
        
        return population
    
    def fitness_function(self, individual, data, fuzzy_results):
        """Calculate fitness score for an individual"""
        total_score = 0
        n_samples = len(data)
        
        mult = individual['price_multiplier']
        elasticity = individual['demand_elasticity']
        sat_weight = individual['satisfaction_weight']
        comp_factor = individual['competition_factor']
        target_margin = individual['profit_margin_target']
        risk_tol = individual['risk_tolerance']
        
        for idx, row in data.iterrows():
            try:
                fuzzy_adj = next((r['price_adjustment_percent'] 
                                 for r in fuzzy_results['individual_results'] 
                                 if r['index'] == idx), 0)
                
                current_price = row.get('price', row.get('selling_price', 100))
                current_cost = row.get('cost', row.get('cost_price', 70))
                quantity = row.get('quantity', row.get('units_sold', 100))
                satisfaction = row.get('satisfaction_score', 75)
                
                base_adjustment = fuzzy_adj / 100
                satisfaction_effect = sat_weight * (satisfaction / 100)
                competition = 1 - row.get('price_competitiveness', 0.5)
                comp_effect = (1 - competition) * comp_factor
                
                total_adjustment = base_adjustment + satisfaction_effect - comp_effect
                risk_adjusted = total_adjustment * (1 - risk_tol) + base_adjustment * risk_tol
                optimized_price = current_price * mult * (1 + risk_adjusted)
                
                expected_demand = quantity * (1 + elasticity * (optimized_price/current_price - 1))
                expected_revenue = optimized_price * expected_demand
                expected_cost = current_cost * expected_demand
                expected_profit = expected_revenue - expected_cost
                expected_margin = expected_profit / expected_revenue if expected_revenue > 0 else 0
                
                profit_score = expected_profit / (current_price * quantity) if current_price * quantity > 0 else 0
                margin_score = expected_margin / target_margin if target_margin > 0 else 0
                satisfaction_score = satisfaction / 100
                
                fitness = 0.5 * min(profit_score, 2.0) + 0.3 * min(margin_score, 1.5) + 0.2 * satisfaction_score
                total_score += fitness
                
            except Exception:
                total_score += 0
            
        return total_score / n_samples if n_samples > 0 else 0
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select parent using tournament selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        crossover_point = np.random.randint(1, len(self.param_names))
        
        for i, param in enumerate(self.param_names):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutate(self, individual):
        """Gaussian mutation"""
        mutated = individual.copy()
        
        for param in self.param_names:
            if np.random.random() < self.mutation_rate:
                current = mutated[param]
                low, high = self.param_bounds[param]
                mutation_strength = (high - low) * 0.1
                new_value = current + np.random.normal(0, mutation_strength)
                mutated[param] = np.clip(new_value, low, high)
        
        return mutated
    
    def optimize(self, data, fuzzy_results):
        """Main optimization loop"""
        population = self.initialize_population()
        
        for generation in range(self.generations):
            fitness_scores = [self.fitness_function(ind, data, fuzzy_results) 
                            for ind in population]
            
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx].copy()
            
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(np.mean(fitness_scores))
            self.history['best_params'].append(best_individual)
            
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Fill rest
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
            
            if generation > 20 and np.std(self.history['best_fitness'][-10:]) < 0.001:
                break
        
        final_fitness = [self.fitness_function(ind, data, fuzzy_results) 
                        for ind in population]
        best_idx = np.argmax(final_fitness)
        
        return {
            'optimal_parameters': population[best_idx],
            'best_fitness': final_fitness[best_idx],
            'optimization_history': self.history,
            'generations_executed': len(self.history['best_fitness']),
            'parameter_importance': self._analyze_parameter_importance()
        }
    
    def _analyze_parameter_importance(self):
        """Analyze importance of each parameter"""
        importance = {}
        
        for param in self.param_names:
            param_values = [ind[param] for ind in self.history['best_params'][-20:]]
            importance[param] = np.std(param_values)
        
        total = sum(importance.values())
        if total > 0:
            for param in importance:
                importance[param] = importance[param] / total
        
        return importance