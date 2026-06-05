"""
Price Optimization Pipeline
Integrates Fuzzy Logic and Genetic Algorithm Results
"""

import numpy as np
import pandas as pd
from datetime import datetime

class PriceOptimizationPipeline:
    """Complete pricing optimization pipeline"""
    
    def __init__(self):
        self.results = None
        self.summary = None
        
    def optimize(self, data, fuzzy_results, ga_results):
        """Combine fuzzy and GA results for final optimization"""
        
        optimal_params = ga_results['optimal_parameters']
        optimized_results = []
        
        for idx, row in data.iterrows():
            fuzzy_rec = next((r for r in fuzzy_results['individual_results'] 
                             if r['index'] == idx), None)
            
            if fuzzy_rec is None:
                continue
            
            current_price = row.get('price', row.get('selling_price', 100))
            current_cost = row.get('cost', row.get('cost_price', 70))
            
            fuzzy_adj = fuzzy_rec['price_adjustment_percent'] / 100
            ga_multiplier = optimal_params['price_multiplier']
            elasticity = optimal_params['demand_elasticity']
            
            demand = row.get('demand_score', 50) / 100
            demand_adj = elasticity * (demand - 0.5)
            
            competition = 1 - row.get('price_competitiveness', 0.5)
            comp_adj = -competition * optimal_params['competition_factor']
            
            total_adj = fuzzy_adj + demand_adj + comp_adj
            
            season = row.get('seasonal_factor', 0)
            season_adj = 1 + season * optimal_params['seasonal_adjustment']
            
            optimal_price = current_price * ga_multiplier * (1 + total_adj) * season_adj
            
            # Realistic constraints
            max_price = current_price * 1.25
            min_price = current_price * 0.80
            min_profit_price = current_cost * 1.20
            
            min_price = max(min_price, min_profit_price)
            optimal_price = np.clip(optimal_price, min_price, max_price)
            
            # Competitor constraint
            competitor_price = row.get('competitor_price', None)
            if competitor_price and not pd.isna(competitor_price):
                max_vs_competitor = competitor_price * 1.15
                optimal_price = min(optimal_price, max_vs_competitor)
            
            quantity = row.get('quantity', row.get('units_sold', 100))
            price_ratio = optimal_price / current_price if current_price > 0 else 1
            expected_quantity = quantity * (1 + elasticity * (price_ratio - 1))
            expected_revenue = optimal_price * expected_quantity
            expected_profit = expected_revenue - (current_cost * expected_quantity)
            expected_margin = (optimal_price - current_cost) / optimal_price * 100 if optimal_price > 0 else 0
            
            confidence = self._calculate_dynamic_confidence(
                fuzzy_rec, price_ratio, expected_quantity, quantity
            )
            
            optimized_results.append({
                'index': idx,
                'product_id': row.get('product_id', row.get('id', f'PROD_{idx}')),
                'product_name': row.get('product_name', row.get('name', f'Product {idx}')),
                'current_price': round(current_price, 2),
                'optimal_price': round(optimal_price, 2),
                'price_change_percent': round((price_ratio - 1) * 100, 2),
                'expected_profit': round(expected_profit, 2),
                'expected_margin': round(expected_margin, 2),
                'expected_revenue': round(expected_revenue, 2),
                'fuzzy_recommendation': fuzzy_rec['fuzzy_recommendation'],
                'confidence_score': round(confidence, 1),
                'action_required': self._determine_action(price_ratio - 1)
            })
        
        self.results = optimized_results
        
        return {
            'optimized_prices': optimized_results,
            'optimal_parameters': optimal_params,
            'total_products': len(optimized_results),
            'average_price_change': np.mean([r['price_change_percent'] for r in optimized_results]),
            'total_expected_profit': sum([r['expected_profit'] for r in optimized_results]),
            'optimization_timestamp': datetime.now().isoformat()
        }

    def _calculate_dynamic_confidence(self, fuzzy_rec, price_ratio, expected_qty, current_qty):
        """Calculate realistic confidence based on multiple factors"""
        confidence = 70.0
        
        if 'confidence_score' in fuzzy_rec:
            confidence = fuzzy_rec['confidence_score'] * 100
        
        price_change = abs(price_ratio - 1)
        if price_change > 0.20:
            confidence -= 20
        elif price_change > 0.10:
            confidence -= 10
        
        qty_ratio = expected_qty / current_qty if current_qty > 0 else 1
        if qty_ratio > 1.5:
            confidence -= 15
        elif qty_ratio < 0.5:
            confidence -= 15
        
        return max(30.0, min(98.0, confidence))
    
    def _determine_action(self, price_change_ratio):
        """Determine recommended action"""
        if price_change_ratio <= -0.15:
            return 'Significant Price Cut'
        elif price_change_ratio <= -0.05:
            return 'Moderate Price Cut'
        elif price_change_ratio < 0:
            return 'Slight Price Cut'
        elif price_change_ratio == 0:
            return 'Maintain Price'
        elif price_change_ratio <= 0.05:
            return 'Slight Price Hike'
        elif price_change_ratio <= 0.15:
            return 'Moderate Price Hike'
        else:
            return 'Significant Price Hike'
    
    def generate_recommendations(self, final_results):
        """Generate actionable recommendations"""
        recommendations = []
        
        for result in final_results['optimized_prices']:
            rec = {
                'product_id': result['product_id'],
                'product_name': result['product_name'],
                'current_price': result['current_price'],
                'recommended_price': result['optimal_price'],
                'expected_impact': f"{result['price_change_percent']:+.1f}%",
                'expected_profit_impact': f"${result['expected_profit']:,.2f}",
                'priority': self._calculate_priority(result),
                'implementation_timeline': self._suggest_timeline(result['price_change_percent']),
                'risk_level': self._assess_risk(result),
                'detailed_reasoning': self._generate_reasoning(result),
                'confidence_score': result['confidence_score'],
                'fuzzy_recommendation': result['fuzzy_recommendation']
            }
            recommendations.append(rec)
        
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        return recommendations
    
    def _calculate_priority(self, result):
        """Calculate implementation priority"""
        priority = 0
        impact = abs(result['price_change_percent'])
        priority += min(impact * 2, 50)
        priority += result['confidence_score'] * 0.3
        
        if result['expected_profit'] > 1000:
            priority += 20
        elif result['expected_profit'] > 500:
            priority += 10
        
        return min(100, priority)
    
    def _suggest_timeline(self, price_change_percent):
        """Suggest implementation timeline"""
        if abs(price_change_percent) > 20:
            return 'Phased (2-3 weeks)'
        elif abs(price_change_percent) > 10:
            return 'Short-term (1-2 weeks)'
        else:
            return 'Immediate'
    
    def _assess_risk(self, result):
        """Assess risk level"""
        if result['confidence_score'] < 60:
            return 'High'
        elif result['confidence_score'] < 80:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_reasoning(self, result):
        """Generate detailed reasoning"""
        reasons = []
        
        if result['price_change_percent'] > 0:
            reasons.append("Strong demand indicators support price increase")
        elif result['price_change_percent'] < 0:
            reasons.append("Competitive pressure suggests price reduction")
        
        if result['confidence_score'] > 80:
            reasons.append("High confidence in recommendation")
        
        reasons.append(f"Based on {result['fuzzy_recommendation']} fuzzy logic classification")
        
        return '; '.join(reasons)
    
    def get_summary_stats(self, final_results):
        """Generate summary statistics"""
        prices = [r['optimal_price'] for r in final_results['optimized_prices']]
        changes = [r['price_change_percent'] for r in final_results['optimized_prices']]
        
        return {
            'total_products': len(prices),
            'avg_price': round(np.mean(prices), 2),
            'median_price': round(np.median(prices), 2),
            'avg_price_change': round(np.mean(changes), 2),
            'price_increases': sum(1 for c in changes if c > 0),
            'price_decreases': sum(1 for c in changes if c < 0),
            'price_unchanged': sum(1 for c in changes if c == 0),
            'total_expected_profit': round(final_results['total_expected_profit'], 2),
            'max_price_change': round(max(changes), 2),
            'min_price_change': round(min(changes), 2)
        }