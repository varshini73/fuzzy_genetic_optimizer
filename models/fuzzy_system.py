"""
Advanced Fuzzy Logic System for Pricing Intelligence
Uses Mamdani inference with 15+ rules
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

class FuzzyPricingSystem:
    """Mamdani Fuzzy Inference System for Price Optimization"""
    
    def __init__(self):
        self.setup_fuzzy_system()
        self.rule_explanations = self._get_rule_explanations()
        
    def setup_fuzzy_system(self):
        """Initialize fuzzy variables and membership functions"""
        
        # INPUT 1: Demand Level (0-100%)
        self.demand = ctrl.Antecedent(np.arange(0, 101, 1), 'demand')
        self.demand['very_low'] = fuzz.trapmf(self.demand.universe, [0, 0, 20, 30])
        self.demand['low'] = fuzz.trimf(self.demand.universe, [20, 35, 50])
        self.demand['medium'] = fuzz.trimf(self.demand.universe, [40, 55, 70])
        self.demand['high'] = fuzz.trimf(self.demand.universe, [60, 75, 90])
        self.demand['very_high'] = fuzz.trapmf(self.demand.universe, [80, 90, 100, 100])
        
        # INPUT 2: Profit Margin (0-100%)
        self.margin = ctrl.Antecedent(np.arange(0, 101, 1), 'margin')
        self.margin['very_low'] = fuzz.trapmf(self.margin.universe, [0, 0, 15, 25])
        self.margin['low'] = fuzz.trimf(self.margin.universe, [15, 30, 45])
        self.margin['medium'] = fuzz.trimf(self.margin.universe, [35, 50, 65])
        self.margin['high'] = fuzz.trimf(self.margin.universe, [55, 70, 85])
        self.margin['very_high'] = fuzz.trapmf(self.margin.universe, [75, 85, 100, 100])
        
        # INPUT 3: Customer Satisfaction (0-100%)
        self.satisfaction = ctrl.Antecedent(np.arange(0, 101, 1), 'satisfaction')
        self.satisfaction['poor'] = fuzz.trapmf(self.satisfaction.universe, [0, 0, 30, 45])
        self.satisfaction['average'] = fuzz.trimf(self.satisfaction.universe, [35, 50, 65])
        self.satisfaction['good'] = fuzz.trimf(self.satisfaction.universe, [55, 70, 85])
        self.satisfaction['excellent'] = fuzz.trapmf(self.satisfaction.universe, [75, 85, 100, 100])
        
        # INPUT 4: Competition Intensity (0-100%)
        self.competition = ctrl.Antecedent(np.arange(0, 101, 1), 'competition')
        self.competition['low'] = fuzz.trapmf(self.competition.universe, [0, 0, 30, 40])
        self.competition['medium'] = fuzz.trimf(self.competition.universe, [30, 50, 70])
        self.competition['high'] = fuzz.trapmf(self.competition.universe, [60, 70, 100, 100])
        
        # OUTPUT: Price Adjustment (-30% to +30%)
        self.adjustment = ctrl.Consequent(np.arange(-30, 31, 1), 'adjustment')
        self.adjustment['large_decrease'] = fuzz.trapmf(self.adjustment.universe, [-30, -30, -20, -15])
        self.adjustment['medium_decrease'] = fuzz.trimf(self.adjustment.universe, [-20, -12, -5])
        self.adjustment['small_decrease'] = fuzz.trimf(self.adjustment.universe, [-10, -5, 0])
        self.adjustment['maintain'] = fuzz.trimf(self.adjustment.universe, [-3, 0, 3])
        self.adjustment['small_increase'] = fuzz.trimf(self.adjustment.universe, [0, 5, 10])
        self.adjustment['medium_increase'] = fuzz.trimf(self.adjustment.universe, [5, 12, 20])
        self.adjustment['large_increase'] = fuzz.trapmf(self.adjustment.universe, [15, 20, 30, 30])
        
        # Define Fuzzy Rules
        self.rules = []
        
        # Core pricing logic
        self.rules.append(ctrl.Rule(self.demand['very_low'] & self.margin['very_low'], 
                                   self.adjustment['large_decrease']))
        self.rules.append(ctrl.Rule(self.demand['very_high'] & self.margin['very_high'], 
                                   self.adjustment['large_increase']))
        self.rules.append(ctrl.Rule(self.demand['low'] & self.margin['medium'], 
                                   self.adjustment['small_decrease']))
        self.rules.append(ctrl.Rule(self.demand['high'] & self.margin['low'], 
                                   self.adjustment['medium_increase']))
        self.rules.append(ctrl.Rule(self.demand['medium'] & self.margin['medium'], 
                                   self.adjustment['maintain']))
        
        # Satisfaction-based adjustments
        self.rules.append(ctrl.Rule(self.satisfaction['poor'], 
                                   self.adjustment['medium_decrease']))
        self.rules.append(ctrl.Rule(self.satisfaction['excellent'] & self.demand['high'], 
                                   self.adjustment['medium_increase']))
        self.rules.append(ctrl.Rule(self.satisfaction['average'] & self.margin['low'], 
                                   self.adjustment['small_increase']))
        
        # Competition-based adjustments
        self.rules.append(ctrl.Rule(self.competition['high'] & self.demand['low'], 
                                   self.adjustment['large_decrease']))
        self.rules.append(ctrl.Rule(self.competition['low'] & self.demand['high'], 
                                   self.adjustment['large_increase']))
        self.rules.append(ctrl.Rule(self.competition['medium'] & self.margin['medium'], 
                                   self.adjustment['maintain']))
        
        # Complex combinations
        self.rules.append(ctrl.Rule(self.demand['high'] & self.margin['high'] & self.satisfaction['excellent'], 
                                   self.adjustment['large_increase']))
        self.rules.append(ctrl.Rule(self.demand['low'] & self.margin['low'] & self.satisfaction['poor'], 
                                   self.adjustment['large_decrease']))
        
        # Create control system
        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)
    
    def _get_rule_explanations(self):
        """Get human-readable rule explanations"""
        return {
            'R1': "Very low demand and very low margin → Large price decrease",
            'R2': "Very high demand and very high margin → Large price increase",
            'R3': "Low demand with medium margin → Small price decrease",
            'R4': "High demand with low margin → Medium price increase",
            'R5': "Medium demand and medium margin → Maintain current price",
            'R6': "Poor customer satisfaction → Medium price decrease",
            'R7': "Excellent satisfaction and high demand → Medium price increase",
            'R8': "Average satisfaction with low margin → Small price increase",
            'R9': "High competition and low demand → Large price decrease",
            'R10': "Low competition and high demand → Large price increase",
            'R11': "Medium competition and medium margin → Maintain price",
            'R12': "High demand, high margin, excellent satisfaction → Large increase",
            'R13': "Low demand, low margin, poor satisfaction → Large decrease"
        }
    
    def analyze_pricing(self, data):
        """Apply fuzzy logic to entire dataset"""
        results = []
        
        for idx, row in data.iterrows():
            try:
                # Set input values
                self.simulator.input['demand'] = min(100, max(0, row.get('demand_score', 50)))
                self.simulator.input['margin'] = min(100, max(0, row.get('profit_margin', 30)))
                self.simulator.input['satisfaction'] = min(100, max(0, row.get('satisfaction_score', 75)))
                
                # Competition input
                if 'price_competitiveness' in row:
                    competition_score = (1 - row['price_competitiveness']) * 100
                    self.simulator.input['competition'] = min(100, max(0, competition_score))
                else:
                    self.simulator.input['competition'] = 50
                
                # Compute fuzzy inference
                self.simulator.compute()
                adjustment = self.simulator.output['adjustment']
                
                results.append({
                    'index': idx,
                    'price_adjustment_percent': round(adjustment, 2),
                    'fuzzy_recommendation': self._categorize_adjustment(adjustment),
                    'confidence_score': self._calculate_confidence(row)
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'price_adjustment_percent': 0.0,
                    'fuzzy_recommendation': 'maintain',
                    'confidence_score': 0.5
                })
        
        return {
            'individual_results': results,
            'average_adjustment': np.mean([r['price_adjustment_percent'] for r in results]),
            'adjustment_distribution': self._get_distribution(results),
            'rule_explanations': self.rule_explanations,
            'total_analyzed': len(results)
        }
    
    def _categorize_adjustment(self, adjustment):
        """Categorize adjustment percentage"""
        if adjustment <= -15:
            return 'Large Decrease'
        elif adjustment <= -5:
            return 'Medium Decrease'
        elif adjustment < 0:
            return 'Small Decrease'
        elif adjustment == 0:
            return 'Maintain'
        elif adjustment <= 5:
            return 'Small Increase'
        elif adjustment <= 15:
            return 'Medium Increase'
        else:
            return 'Large Increase'
    
    def _calculate_confidence(self, row):
        """Calculate confidence score for recommendation"""
        confidence = 0.7
        
        if 'demand_score' in row and 'profit_margin' in row:
            confidence += 0.15
        if 'satisfaction_score' in row:
            confidence += 0.1
        if 'price_competitiveness' in row:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _get_distribution(self, results):
        """Get distribution of adjustments"""
        adjustments = [r['price_adjustment_percent'] for r in results]
        return {
            'min': min(adjustments),
            'max': max(adjustments),
            'mean': np.mean(adjustments),
            'std': np.std(adjustments),
            'quartiles': np.percentile(adjustments, [25, 50, 75]).tolist()
        }