"""
AI Pricing Assistant with ChatGPT Integration
Fixed version with better error handling and logging
"""

import os
import sys
from datetime import datetime

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI package not installed. Run: pip install openai")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class ChatGPTPricingAssistant:
    """ChatGPT-powered pricing assistant"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.conversation_history = []
        
        # Initialize client if API key exists
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                # Test the connection
                self.client.models.list()
                self.is_connected = True
                print(f"✅ ChatGPT connected! Using model: {self.model}")
            except Exception as e:
                print(f"⚠️ ChatGPT connection failed: {e}")
                self.client = None
                self.is_connected = False
        else:
            self.client = None
            self.is_connected = False
            if not self.api_key:
                print("⚠️ No OpenAI API key found. Using fallback responses.")
            if not OPENAI_AVAILABLE:
                print("⚠️ OpenAI package not installed. Run: pip install openai")
        
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self):
        return """You are an expert AI Pricing Assistant for FuzzyPriceAI, a system that uses Fuzzy Logic and Genetic Algorithms for retail price optimization.

Your role: Help small business owners understand pricing recommendations in simple, clear language.

Context you have expertise in:
- Fuzzy logic for handling market uncertainty
- Genetic algorithms for parameter optimization
- Retail pricing strategies and profit maximization
- Demand elasticity and customer satisfaction metrics

Guidelines:
1. Be friendly, conversational, and helpful
2. Explain complex concepts in simple terms with examples
3. Use emojis occasionally to be engaging 😊
4. Provide actionable, practical advice
5. Keep responses concise (2-4 sentences when possible)
6. If asked about a specific product, reference its data
7. Never recommend selling below cost

Format: Respond directly to the user's question without meta-commentary."""
    
    def chat(self, user_message, product_context=None, optimization_results=None):
        """Process user message and return AI response"""
        
        # Build context
        context = self._build_context(product_context, optimization_results)
        
        # If ChatGPT is available, use it
        if self.is_connected and self.client:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": context},
                    *self.conversation_history[-6:],
                    {"role": "user", "content": user_message}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=250,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                return {
                    'response': ai_response,
                    'model': self.model,
                    'fallback': False,
                    'connected': True
                }
                
            except Exception as e:
                print(f"ChatGPT error: {e}")
                return {
                    'response': self._get_smart_fallback(user_message, product_context),
                    'fallback': True,
                    'connected': False,
                    'error': str(e)
                }
        else:
            # Use fallback
            return {
                'response': self._get_smart_fallback(user_message, product_context),
                'fallback': True,
                'connected': False
            }
    
    def _build_context(self, product_context, optimization_results):
        """Build context for the AI"""
        parts = ["Current session context:"]
        
        if product_context:
            parts.append(f"Product: {product_context.get('product_name', 'Unknown')}")
            parts.append(f"Current price: ${product_context.get('current_price', 0):.2f}")
            parts.append(f"Recommended: ${product_context.get('recommended_price', 0):.2f}")
            parts.append(f"Change: {product_context.get('expected_impact', '0%')}")
            parts.append(f"Confidence: {product_context.get('priority', 0):.0f}%")
        
        if optimization_results:
            summary = optimization_results.get('summary_stats', {})
            if summary:
                parts.append(f"Total products: {summary.get('total_products', 0)}")
        
        return "\n".join(parts) if len(parts) > 1 else "No specific product context."
    
    def _get_smart_fallback(self, message, context):
        """Smart fallback responses when ChatGPT is unavailable"""
        msg = message.lower()
        
        # Pricing factors
        if 'factor' in msg or 'affect' in msg:
            if context and context.get('product_name'):
                return f"📊 For {context['product_name']}, the main factors are:\n• Demand level\n• Profit margin\n• Competition intensity\n• Customer satisfaction\n\nOur fuzzy logic system weighs these to recommend optimal prices."
            return "📊 Key pricing factors:\n• Demand level (how popular the product is)\n• Profit margin (cost vs. selling price)\n• Competition (what others charge)\n• Customer satisfaction (loyalty and reviews)\n• Seasonality (time-based demand changes)"
        
        # Confidence
        if 'confiden' in msg:
            if context:
                conf = context.get('priority', 75)
                if conf >= 80:
                    return f"✅ I'm {conf:.0f}% confident - strong data supports this recommendation."
                elif conf >= 60:
                    return f"📈 Confidence is {conf:.0f}% - moderate. Monitor results after implementation."
                else:
                    return f"⚠️ Confidence is {conf:.0f}% - consider gathering more data first."
            return "Confidence scores:\n• 80-100%: High - safe to implement\n• 60-79%: Moderate - monitor closely\n• Below 60%: Low - gather more data"
        
        # Fuzzy logic
        if 'fuzzy' in msg:
            return "🧠 Fuzzy logic handles uncertainty in pricing. Instead of 'demand is high or low', it says 'demand is 70% high'. This gives more realistic recommendations for real-world scenarios."
        
        # Genetic algorithm
        if 'genetic' in msg:
            return "🧬 Our genetic algorithm tests thousands of price combinations, keeping the best ones and combining them - like evolution! It finds optimal prices for maximum profit."
        
        # Profit
        if 'profit' in msg or 'margin' in msg:
            return "💰 The goal is finding the sweet spot where (price × quantity) maximizes total profit. Sometimes a lower price with higher volume yields more total profit than a higher price with fewer sales."
        
        # Competition
        if 'compet' in msg:
            return "🏪 When pricing against competitors, consider your unique value. Better quality or service justifies higher prices. If you're similar to competitors, stay competitive to maintain market share."
        
        # Implementation
        if 'implement' in msg or 'change' in msg or 'apply' in msg:
            return "📋 To implement price changes:\n1. Start with small adjustments (5-10%)\n2. Monitor sales for 1-2 weeks\n3. Communicate value to customers\n4. Adjust based on results"
        
        # Greeting
        if any(w in msg for w in ['hello', 'hi', 'hey', 'help']):
            return "👋 Hello! I'm your AI Pricing Assistant. I can explain:\n• Why prices are recommended\n• What factors affect pricing\n• How confidence scores work\n• Implementation strategies\n\nWhat would you like to know?"
        
        # Default
        return "I'm here to help with pricing questions! You can ask about:\n• Factors affecting prices\n• Confidence scores\n• How fuzzy logic works\n• Implementation tips\n• Profit optimization\n\nWhat specifically interests you?"
    
    def get_suggested_questions(self, product_name=None):
        """Get suggested questions"""
        if product_name:
            return [
                f"Why is {product_name}'s price changing?",
                f"How confident is the {product_name} recommendation?",
                "What factors affect this price?",
                "How should I implement this change?"
            ]
        return [
            "What factors affect pricing?",
            "How does confidence scoring work?",
            "What is fuzzy logic?",
            "How can I maximize profit?",
            "Should I match competitor prices?"
        ]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return {"status": "cleared"}
    
    def get_status(self):
        """Get connection status"""
        return {
            'connected': self.is_connected,
            'model': self.model if self.is_connected else None,
            'fallback_mode': not self.is_connected
        }