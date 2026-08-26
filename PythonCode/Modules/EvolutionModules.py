import sys
import os
sys.path.append(os.path.abspath("../..")) 

import random
import numpy as np
import copy
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Modules.Plotters import Plotter
from Modules.Helpers import Helper
from Modules.Equations import *

# Representa um coeficiente com valor e limites
class Coefficient:
    def __init__(self, bounds):
        self.val = np.random.uniform(*bounds)  # Inicializa com valor aleatório dentro dos limites
        self.bounds = bounds
    
    def __repr__(self):
        return f"val={self.val}"

# Representa um coeficiente usado no CMA-ES com limites.
class CMACoefficient:
    def __init__(self, val, bounds):
        self.bounds = bounds
        self.val = self.limit_val(val)  # Ajusta o valor aos limites

    # Garante que o valor esteja dentro dos limites
    def limit_val(self, val):
        return max(self.bounds[0], min(val, self.bounds[1]))

    def __repr__(self):
        return f"val={self.val}"
    
    
    
# Representa um indivíduo contendo coeficientes e funções para manipulação
class Individual:
    def __init__(self, model):
        self.model = model
        self.fitness = np.inf # Fitness inicializado como infinito
        self.coeffs = copy.deepcopy(self.model.coeffs)
        
        
    # def solve_ivp(self, solver='RK45'):
    #     return integrate.solve_ivp(self.model.system, self.model.t_span, self.model.initial_conditions, method=solver, t_eval=self.model.t_eval, args=(self, self.equation), min_step=0.001).y
    
    def solve_ivp(self, test=False, solver='RK45'):
        if test:
            initial_conditions = self.model.test_initial_conditions
            t_eval = self.model.t_test
            t_span = self.model.test_t_span
            print('teste: ', self.model.t_test, self.model.t_eval)
        else:   
            initial_conditions = self.model.initial_conditions
            t_eval = self.model.t_eval
            t_span = self.model.t_span
        
        
        if solver.upper() == 'ODEINT':
            sol = odeint(
                self.model.system,
                # lambda y, t: self.model.system(t, y, self, self.equation),  # Wrap system for odeint (t first)
                initial_conditions,
                t_eval,
                args=(self.model.max_data,self.model.MatrixInd,self.model.encodedLabels),
                tfirst=True,  # Important: tells odeint the function is (t, y) instead of (y, t)
                hmin=0.001
            )
            
            return sol.T
        else:
            return integrate.solve_ivp(
                self.model.system,
                t_span,
                initial_conditions,
                method=solver,
                t_eval=t_eval,
                args=(self.model.max_data,self.model.MatrixInd,self.model.encodedLabels)#,
                #min_step=0.001
                
            ).y
    
    def ind_to_list(self):
        ind_list = []

        for label, label_dict in self.coeffs.items():

            if not self.model.is_excluded(label, 'tau'):
                ind_list.append(label_dict['tau'].val)

            for target, coeffs in label_dict.items():
                if target == 'tau':
                    continue

                if not self.model.is_excluded(label, target, 'n'):
                    ind_list.append(coeffs['n'].val)

                if not self.model.is_excluded(label, target, 'k'):
                    ind_list.append(coeffs['k'].val)

        return ind_list
        
    # model.original_train vira o argumento data   
    def calculate_fitness(self, test=False, solver='RK45', error='SQUARED'):
        if test:
            data = self.model.original_test
        else:
            data = self.model.original_train
        try:
            y = self.solve_ivp(test=test, solver=solver)
            self.fitness = Helper.calculate_error(data, y, error)
            self.fitness = min(self.fitness, 1e6)
        except Exception as e:
            # Trata exceções relacionadas ao solver
            print(f"Error msg on EvolutionModules/Individual/calculate_fitness:{e}")
            self.fitness = 1e6
            
    def calc_all_fitness(self, test=False, solver='RK45'):
        if test:
            data = self.model.original_test
        else:
            data = self.model.original_train
        
        
        y = self.solve_ivp(test=test, solver=solver)
        fitness_dict = {}
        for error, error_func in Helper.errors_dict().items():
              fitness_dict[error] = error_func(data, y,)
       
        return fitness_dict
       
    def initialize_ind(self, solver='RK45', error='SQUARED'):
        for label, label_dict in self.coeffs.items():

            if self.model.is_excluded(label, 'tau'):
                label_dict['tau'] = Coefficient((1, 1))
                label_dict['tau'].val = 1
            else:
                label_dict['tau'] = Coefficient(self.model.bounds['tau'])

            for target, coeffs in label_dict.items():
                if target == 'tau':
                    continue

                if self.model.is_excluded(label, target, 'n'):
                    coeffs['n'] = Coefficient((1, 1))
                    coeffs['n'].val = 1
                else:
                    coeffs['n'] = Coefficient(self.model.bounds['n'])

                if self.model.is_excluded(label, target, 'k'):
                    coeffs['k'] = Coefficient((0, 0))
                    coeffs['k'].val = 0
                else:
                    coeffs['k'] = Coefficient(self.model.bounds['k'])

        self.calculate_fitness(solver=solver, error=error)
            
     
    @property
    def equation(self):
        return Equation(self.numerical_coeffs, self.model.labels)
      
    @property
    def numerical_coeffs(self):
        numerical_coeffs = copy.deepcopy(self.coeffs)
        for key, label in numerical_coeffs.items():
            label['tau'] = label['tau'].val
            for key, coeffs in label.items():
                if key != 'tau':
                    coeffs['n'] = int(coeffs['n'].val)
                    coeffs['k'] = coeffs['k'].val
                    
        return numerical_coeffs

            
    @staticmethod
    def initialize_average_bounds(model):
        array = np.zeros(model.IND_SIZE)
        i = 0

        for label, label_dict in model.coeffs.items():

            # tau
            if not model.is_excluded(label, 'tau'):
                array[i] = np.mean(model.bounds['tau'])
                i += 1

            for target, coeffs in label_dict.items():
                if target == 'tau':
                    continue

                # n
                if not model.is_excluded(label, target, 'n'):
                    array[i] = np.mean(model.bounds['n'])
                    i += 1

                # k
                if not model.is_excluded(label, target, 'k'):
                    array[i] = np.mean(model.bounds['k'])
                    i += 1

        return array
            
    
    @staticmethod
    def apply_bounds(population, model):
        for ind in population:
            list_ind = Individual.list_to_ind(ind, model)
            ind[:] = Individual.ind_to_list(list_ind)
    
    @staticmethod    
    def cma_evaluate(list_ind, model, solver='RK45', error='SQUARED'):
        ind = Individual.list_to_ind(list_ind, model)
        ind.calculate_fitness(solver=solver, error=error)
        return ind.fitness,
    
    @staticmethod
    def list_to_ind(list_ind, model): # could be costly
        i = 0
        ind = Individual(model=model)

        for label, label_dict in ind.coeffs.items():

            if model.is_excluded(label, 'tau'):
                label_dict['tau'] = CMACoefficient(1, (1, 1))
            else:
                label_dict['tau'] = CMACoefficient(list_ind[i], model.bounds['tau'])
                i += 1

            for target, coeffs in label_dict.items():
                if target == 'tau':
                    continue

                if model.is_excluded(label, target, 'n'):
                    coeffs['n'] = CMACoefficient(1, (1, 1))
                else:
                    coeffs['n'] = CMACoefficient(list_ind[i], model.bounds['n'])
                    i += 1

                if model.is_excluded(label, target, 'k'):
                    coeffs['k'] = CMACoefficient(0, (0, 0))
                else:
                    coeffs['k'] = CMACoefficient(list_ind[i], model.bounds['k'])
                    i += 1

        return ind
        
    
    
        
    def __repr__(self):
        coeffs_repr = {k: v for k, v in self.coeffs.items()}
        return f"Individual(fitness={self.fitness}, coeffs={coeffs_repr}, ind_size={self.model.IND_SIZE})"