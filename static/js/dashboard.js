// Dashboard utilities for Fuzzy-Genetic Pricing Optimizer

class DashboardManager {
    constructor() {
        this.charts = {};
        this.currentData = null;
    }
    
    async loadResults(taskId) {
        try {
            const response = await fetch(`/api/results/${taskId}`);
            this.currentData = await response.json();
            return this.currentData;
        } catch (error) {
            console.error('Error loading results:', error);
            throw error;
        }
    }
    
    createSummaryCards(summary) {
        const cards = [
            {
                title: 'Total Products',
                value: summary.total_products,
                icon: 'fa-box',
                color: 'primary'
            },
            {
                title: 'Avg Price Change',
                value: `${summary.avg_price_change.toFixed(2)}%`,
                icon: 'fa-percent',
                color: summary.avg_price_change > 0 ? 'success' : 'danger'
            },
            {
                title: 'Expected Profit',
                value: `$${summary.total_expected_profit.toLocaleString()}`,
                icon: 'fa-dollar-sign',
                color: 'success'
            },
            {
                title: 'Confidence Score',
                value: `${this.calculateAvgConfidence()}%`,
                icon: 'fa-chart-line',
                color: 'info'
            }
        ];
        
        return cards.map(card => `
            <div class="col-md-3 mb-3">
                <div class="card bg-${card.color} text-white fade-in">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="text-uppercase">${card.title}</h6>
                                <h2 class="mb-0">${card.value}</h2>
                            </div>
                            <i class="fas ${card.icon} fa-2x opacity-50"></i>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    calculateAvgConfidence() {
        if (!this.currentData) return 0;
        const results = this.currentData.final_results.optimized_prices;
        const avgConfidence = results.reduce((sum, r) => sum + r.confidence_score, 0) / results.length;
        return avgConfidence.toFixed(1);
    }
    
    createPriceDistributionChart(canvasId, data) {
        const prices = data.final_results.optimized_prices.map(r => r.optimal_price);
        const currentPrices = data.final_results.optimized_prices.map(r => r.current_price);
        
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // Create histogram bins
        const bins = 20;
        const maxPrice = Math.max(...prices, ...currentPrices);
        const minPrice = Math.min(...prices, ...currentPrices);
        const binWidth = (maxPrice - minPrice) / bins;
        
        const optimalBins = new Array(bins).fill(0);
        const currentBins = new Array(bins).fill(0);
        
        prices.forEach(price => {
            const binIndex = Math.min(Math.floor((price - minPrice) / binWidth), bins - 1);
            optimalBins[binIndex]++;
        });
        
        currentPrices.forEach(price => {
            const binIndex = Math.min(Math.floor((price - minPrice) / binWidth), bins - 1);
            currentBins[binIndex]++;
        });
        
        this.charts.priceDist = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: bins}, (_, i) => 
                    `$${(minPrice + i * binWidth).toFixed(2)}`),
                datasets: [{
                    label: 'Optimal Prices',
                    data: optimalBins,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Current Prices',
                    data: currentBins,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Price Distribution Analysis'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Products'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Price Range'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    }
    
    createParameterImportanceChart(containerId, gaResults) {
        const importance = gaResults.parameter_importance;
        const labels = Object.keys(importance).map(k => k.replace(/_/g, ' '));
        const values = Object.values(importance);
        
        const trace = {
            type: 'bar',
            x: values,
            y: labels,
            orientation: 'h',
            marker: {
                color: values.map(v => `rgba(102, 126, 234, ${v + 0.3})`)
            }
        };
        
        const layout = {
            title: 'Parameter Importance Analysis',
            margin: {l: 150, r: 20, t: 40, b: 40},
            xaxis: {
                title: 'Relative Importance',
                tickformat: ',.0%'
            }
        };
        
        Plotly.newPlot(containerId, [trace], layout);
    }
    
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    }
    
    formatPercent(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 1,
            maximumFractionDigits: 1
        }).format(value / 100);
    }
    
    exportToCSV(data, filename) {
        const recommendations = data.recommendations;
        const csv = [
            ['Product ID', 'Product Name', 'Current Price', 'Recommended Price', 
             'Expected Impact', 'Priority', 'Risk Level', 'Timeline'],
            ...recommendations.map(r => [
                r.product_id,
                r.product_name,
                r.current_price,
                r.recommended_price,
                r.expected_impact,
                r.priority + '%',
                r.risk_level,
                r.implementation_timeline
            ])
        ].map(row => row.join(',')).join('\n');
        
        const blob = new Blob([csv], {type: 'text/csv'});
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }
}

// Initialize dashboard manager
const dashboard = new DashboardManager();

// Export for global use
window.dashboard = dashboard;