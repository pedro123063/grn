import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math
import time
import pandas as pd
import Modules.Equations as Eq
import Modules.Encode as Enc


# Importação de módulos personalizados
from Modules.Helpers import Helper
from Modules.Solvers import Solvers

from Modules.Encode import *
from Modules.TrainTest import TemporalTrainTest

class Model:
    def __init__(self, coeffs, bounds, system, labels, datapath,\
                  name, train_percentage,encodedLabels,encodingdict,encodedBounds,idxMatrix,coeffVet,tauVet,\
                  excluded_coeffs={},
                    \
                ):
        self.coeffs = coeffs
        self.excluded_coeffs = excluded_coeffs
        self.bounds = bounds
        self.system = system
        self.IND_SIZE = self.count_coeffs()
        self.labels = labels
        self.datapath = datapath
        self.name = name
        self.train_percentage = train_percentage
        self.resolve_data(train_percentage)
        self.encodingDict=encodingdict
        self.encodedBounds=encodedBounds
        self.idxMatrix=idxMatrix
        self.coeffVet=coeffVet
        self.tauVet=tauVet
        self.encodedLabels=encodedLabels
    # def resolve_data(self):
    #     self.df, self.max_data = Helper.load_data(filename=self.datapath, labels=self.labels)
    #     self.initial_conditions = np.array([self.df[label].iloc[0] for label in self.labels])
    #     self.t_span = (self.df['t'].iloc[0], self.df['t'].iloc[-1])  # Intervalo de tempo para simulações
    #     self.t_eval = np.array(self.df['t'])  # Ponto de avaliação dos dados temporais
    #     self.original = np.array(self.df[self.labels]).T  # Dados originais para cálculo de erro

    def is_excluded(self, *keys):
        ref = self.excluded_coeffs
        for k in keys:
            if not isinstance(ref, dict) or k not in ref:
                return False
            ref = ref[k]
        return True
    
    def resolve_data(self, train_percentage=1.0):
        df, _ = Helper.load_data(self.datapath, self.labels)

        t = np.array(df['t'])
        y = np.array(df[self.labels]).T

        n = len(t)
        self.n_train = int(np.ceil(train_percentage * n))

        self.t_train = t[:self.n_train]

        self.original_train = y[:, :self.n_train]

        # max_data calculado a partir dos dados de treino
        self.max_data =np.array(\
            [np.max(self.original_train[i, :])for i, label in enumerate(self.labels)]
        ,dtype=np.double)

        # Configuração padrão do modelo (treino)
        self.t_eval = self.t_train
        self.original = self.original_train
        self.initial_conditions = self.original_train[:, 0]
        self.t_span = (self.t_train[0], self.t_train[-1])
        
        if self.train_percentage < 1.0:
            self.t_test  = t[self.n_train:]
            self.original_test  = y[:, self.n_train:] if self.n_train < n else None
            self.test_initial_conditions = self.original_test[:, 0] if self.original_test is not None else None
            self.test_t_span = (self.t_test[0], self.t_test[-1])
        
    def count_coeffs(self):
        return self.count_coeffs_aux(self.coeffs, self.excluded_coeffs)
        
    def count_coeffs_aux(self, sub_dict, excl_sub={}):
        count = 0

        for key, value in sub_dict.items():
            if key == '-':
                continue

            if isinstance(value, dict):
                count += self.count_coeffs_aux(
                    value,
                    excl_sub.get(key, {})
                )
            else:
                if key not in excl_sub:
                    count += 1

        return count
    
    def summarize_coeffs(self, coeffs, indent=2, level=0):
        lines = []
        prefix = ' ' * (indent * level)
        for key, value in coeffs.items():
            if key == '-':
                continue
            lines.append(f"{prefix}{key}")
            if isinstance(value, dict):
                lines.append(self.summarize_coeffs(value, indent, level + 1))
        return '\n'.join(lines)
    
        
    def bounds_list(self):
        bounds_list = []
        for key, label in self.coeffs.items():
            bounds_list.append(self.bounds['tau'])
            for key, coeffs in label.items():
                if key != 'tau':
                    bounds_list.append(self.bounds['n'])
                    bounds_list.append(self.bounds['k'])
                    
        return bounds_list
    

    def __repr__(self):
        coeff_summary = self.summarize_coeffs(self.coeffs)
        return (
            f"<Model Summary>\n"
            f"System: {self.system}\n"
            f"Labels: {', '.join(self.labels)}\n"
            f"Data Path: {self.datapath}\n"
            f"Number of Coefficients: {self.IND_SIZE}\n"
            f"Coefficient Structure:\n{coeff_summary}\n"
            f"Bounds: \n{self.bounds}\n"
        )
    
class ModelWrapper:
            #simply wraps the model based on our desire of use GRN(5 or 10), ABCD or ECOLI , fowarding the infos the  the right places. Read Model for more deatil.
            #Returns a Model
    @staticmethod 
    def GRN5_system(t, y,maxData,matrixInd,encodedLabels):
        #DIFFERENTIAL EQUATIONS SYSTEM ?
        
        vals = np.empty_like(y)
        for i in range(len(y)): #list of y's after being normalized by the maxValue of their respective label 
            vals[i]=Solvers.norm_hardcoded(y[i], maxData[i])
        #(if y[0] is a value for a label[0]==A , it will be normalized by max_val of all values associated with A)
        #before the for : normalizes each value using the maximum value of their correspondent label
        #zip : creates tuples (y[index],labels[index])
        
        dA = Eq.full_eq(vals, encodedLabels[0], encodedLabels[4],matrixInd) # calculates the value of dA using vals(matrix) , 'A' and 'E'
        dB = Eq.full_eq(vals, encodedLabels[1], encodedLabels[0],matrixInd)
        dC = Eq.full_eq(vals, encodedLabels[2], encodedLabels[1],matrixInd)
        dD = Eq.full_eq(vals, encodedLabels[3], encodedLabels[2],matrixInd)
        pairs=np.array([ [ encodedLabels[1] , encodedLabels[3] ] , [ encodedLabels[3], encodedLabels[4] ] ],dtype=np.int8)
        dE = Eq.complex_eqs(vals, 4, pairs,matrixInd)
    
        return np.array([dA, dB, dC, dD, dE],dtype=np.double)
    @staticmethod
    def GRN5(train_percentage, excluded_coeffs={}): #returns a model for the GRN5
        labels = ['A', 'B', 'C', 'D', 'E']
        datapath = '../../Data/GRN5_DATA.txt'
        
        coeffs = {  #base
            'A': {  #target 
                'E': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'B': {  #target
                'A': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'C': {  #target
                'B': {'n': None, 'k': None, '-': False},
                'tau': None,
            },
            'D': {  #target
                'C': {'n': None, 'k': None, '-': False},
                'tau': None,
            },
            'E': {  #target
                'D': {'n': None, 'k': None, '-': False},
                'B': {'n': None, 'k': None, '-': False},
                'E': {'n': None, 'k': None, '-': False},
                'tau': None,
            }
        }
        
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        encodedLabels,encodingdict,encodedBounds,idxMatrix,coeffVet,tauVet=Enc.encode(labels,coeffs,bounds)
        # ind.model.max_data vira um argumento max_data
        # equação é argumento para aumentar eficiencia da função    

            
        return Model(coeffs=coeffs, bounds=bounds, system=ModelWrapper.GRN5_system, labels=labels, datapath=datapath, name='GRN5', train_percentage=train_percentage,\
                     encodedLabels=encodedLabels,encodingdict=encodingdict,\
                    encodedBounds=encodedBounds,idxMatrix=idxMatrix,coeffVet=coeffVet,tauVet=tauVet,excluded_coeffs=excluded_coeffs)
    
    
    @staticmethod
    def GRN10(train_percentage):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        datapath = '../../Data/GRN10_DATA.txt'
        
        coeffs = {
            'A': {
                'J': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'B': {
                'E': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'C': {
                'A': {'n': None, 'k': None},
                'B': {'n': None, 'k': None},
                'F': {'n': None, 'k': None},
                'tau': None
            },
            'D': {
                'F': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'E': {
                'J': {'n': None, 'k': None, '-': True},
                'tau': None
            },
            'F': {
                'A': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'G': {
                'A': {'n': None, 'k': None},
                'B': {'n': None, 'k': None},
                'F': {'n': None, 'k': None},
                'tau': None
            },
            'H': {
                'F': {'n': None, 'k': None, '-': False},
                'tau': None
            },
            'I': {
                'G': {'n': None, 'k': None},
                'H': {'n': None, 'k': None},
                'tau': None
            },
            'J': {
                'I': {'n': None, 'k': None, '-': False},
                'tau': None
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.full_eq(vals, 'A', 'J')
            dB = equation.full_eq(vals, 'B', 'E')
            dC = equation.complex_eqs(vals, 'C', [
                        ['+B', '-F', '-A'], 
                        ['-B', '+F', '-A'],
                        ['-B', '-F', '+A'],
                        ['+B', '-F', '+A'],
                        ['-B', '+F', '+A'],
                        ['+B', '+F', '+A'],
            ])
            dD = equation.full_eq(vals, 'D', 'F')
            dE = equation.full_eq(vals, 'E', 'J')
            dF = equation.full_eq(vals, 'F', 'A')
            dG = equation.complex_eqs(vals, 'G', [
                        ['+B', '-F', '-A'],
                        ['-B', '+F', '-A'],
                        ['-B', '-F', '+A'],
                        ['+B', '-F', '+A'],
                        ['-B', '+F', '+A'],
                        ['+B', '+F', '+A'] 
            ])
            dH = equation.full_eq(vals, 'H', 'F')
            dI = equation.complex_eqs(vals, 'I', [['+G', '+H']])
            dJ = equation.full_eq(vals, 'J', 'I')

            return [dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ]

        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='GRN10', train_percentage=train_percentage)


    @staticmethod
    def ABCD(train_percentage):
        labels = ['A', 'B', 'C', 'D']
        datapath = '../../Data/ABCD_DATA.txt'
        
        coeffs = {
            'A': {
                'A': {'n': None, 'k': None},  # nAA & kAA
                'B': {'n': None, 'k': None},  # nAB & kAB
                'D': {'n': None, 'k': None},  # nAD & kAD
                'tau': None                   # tauA
            },
            'B': {
                'C': {'n': None, 'k': None}, # nBC & kBC
                'D': {'n': None, 'k': None}, # nBD & kBD
                'tau': None                  # tauB
            },
            'C': {
                'A': {'n': None, 'k': None}, # nCA & kCA
                'D': {'n': None, 'k': None}, # nCD & kCD
                'tau': None,                 # tauC
            },
            'D': {
                'A': {'n': None, 'k': None}, # nDA & kDA
                'D': {'n': None, 'k': None}, # nDD & kDD
                'tau': None,                 # tauD
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.complex_eqs(vals, 'A', [['-A', '-D'], ['+B', '-D'], ['+A', '-B', '+D']])
            dB = equation.complex_eqs(vals, 'B', [['-C'], ['+D']])
            dC = equation.complex_eqs(vals, 'C', [['+D'], ['-A']])
            dD = equation.complex_eqs(vals, 'D', [['-A'], ['-D']])

            return [dA, dB, dC, dD]

        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='ABCD', train_percentage=train_percentage)

    @staticmethod
    def ECOLI(train_percentage):
        labels = ['A', 'B', 'C', 'D', 'E']
        datapath = '../../Data/ECOLI_DATA.txt'
        
        coeffs = {
            'A': {
                'A': {'n': None, 'k': None},  # nAA & kAA
                'C': {'n': None, 'k': None},  # nAC & kAC
                'D': {'n': None, 'k': None},  # nAD & kAD
                'E': {'n': None, 'k': None},  # nAE & kAE
                'tau': None                   # tauA
            },
            'B': {
                'A': {'n': None, 'k': None}, # nBA & kBA
                'C': {'n': None, 'k': None}, # nBC & kBC
                'D': {'n': None, 'k': None}, # nBD & kBD
                'E': {'n': None, 'k': None}, # nBE & kBE
                'tau': None                  # tauB
            },
            'C': {
                'D': {'n': None, 'k': None}, # nCD & kCD
                'E': {'n': None, 'k': None}, # nCE & kCE
                'tau': None,                 # tauC
            },
            'D': {
                'C': {'n': None, 'k': None, '-': False}, # nDC & kDC
                'tau': None,                             # tauD
            },
            'E': {
                'A': {'n': None, 'k': None}, # nEA & kEA
                'B': {'n': None, 'k': None}, # nEB & kEB
                'C': {'n': None, 'k': None}, # nEC & kEC
                'E': {'n': None, 'k': None}, # nEE & kEE
                'tau': None,                 # tauD
            }
        }
        
        bounds = {
            'tau': (0.1, 5.0),
            'k': (0.1, 2.0),
            'n': (0.1, 30.0)
        }

        # equação é argumento para aumentar eficiencia da função    
        def system(t, y, ind, equation):
            vals = [Solvers.norm_hardcoded(val, ind.model.max_data[label]) for val, label in zip(y, labels)]
            
            dA = equation.complex_eqs(vals, 'A', [['-A', '-D', '-E'], ['-A', '-C', '+E'], ['+A', '+D', '-E']])
            dB = equation.complex_eqs(vals, 'B', [['-A', '-D', '-E'], ['-A', '-C', '+E'], ['+A', '+D', '-E']])
            dC = equation.complex_eqs(vals, 'C', [['+D'], ['+E']])
            dD = equation.full_eq(vals, 'D', 'C')
            dE = equation.complex_eqs(vals, 'E', [['+A', '+B', '+C'], ['+E']])

            return [dA, dB, dC, dD, dE]

        return Model(coeffs=coeffs, bounds=bounds, system=system, labels=labels, datapath=datapath, name='ECOLI', train_percentage=train_percentage)

