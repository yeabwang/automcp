#!/usr/bin/env python3
"""
Enhanced MCP Agent Tool Picker with better intent detection
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class QueryAnalysis:
    intent: str
    confidence: float
    parameters: Dict[str, Any]
    safety_level: str
    suggested_tools: List[str]

class EnhancedMCPAgentToolPicker:
    """Enhanced MCP Agent with better intent detection using the analysis files"""
    
    def __init__(self, analysis_dir: str = "tests/outputs"):
        self.analysis_dir = Path(analysis_dir)
        self.capabilities = self._load_capabilities()
        self.intents = self._load_intents()
        self.tools = self._load_tools()
        self.api_base_url = "https://api.example.com"
        
        # Create intent mapping from analysis files
        self.intent_patterns = self._build_intent_patterns()
        
    def _load_capabilities(self) -> Dict[str, List[str]]:
        try:
            with open(self.analysis_dir / "groq_capabilities.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"white": [], "grey": [], "black": []}
    
    def _load_intents(self) -> List[Dict[str, Any]]:
        try:
            with open(self.analysis_dir / "groq_enriched_intents.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _load_tools(self) -> List[Dict[str, Any]]:
        try:
            with open(self.analysis_dir / "groq_mcp_tools.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _build_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build intent patterns from the enriched intents file"""
        patterns = {}
        
        for intent in self.intents:
            intent_name = intent.get('name', '')
            details = intent.get('details', {})
            
            # Create pattern mapping
            patterns[intent_name] = {
                'keywords': self._extract_keywords(details.get('summary', '')),
                'domain': details.get('domain', ''),
                'complexity': details.get('complexity', 'simple'),
                'user_context': details.get('user_context', ''),
                'paths': intent.get('paths', [])
            }
        
        return patterns
    
    def _extract_keywords(self, summary: str) -> List[str]:
        """Extract keywords from intent summary"""
        # Simple keyword extraction
        keywords = []
        summary_lower = summary.lower()
        
        if 'retrieve' in summary_lower or 'get' in summary_lower:
            keywords.extend(['find', 'get', 'show', 'list', 'search'])
        if 'create' in summary_lower or 'add' in summary_lower:
            keywords.extend(['create', 'add', 'new', 'make'])
        if 'update' in summary_lower or 'modify' in summary_lower:
            keywords.extend(['update', 'modify', 'change', 'edit'])
        if 'delete' in summary_lower or 'remove' in summary_lower:
            keywords.extend(['delete', 'remove', 'cancel'])
        if 'order' in summary_lower:
            keywords.extend(['order', 'buy', 'purchase'])
        if 'product' in summary_lower:
            keywords.extend(['product', 'item', 'dress', 'clothing'])
            
        return keywords
    
    def analyze_query(self, user_query: str) -> QueryAnalysis:
        """Enhanced query analysis using intent patterns"""
        query_lower = user_query.lower()
        
        # Extract parameters
        parameters = {}
        
        # Price extraction
        price_patterns = [
            r'under\s+(\d+)',
            r'below\s+(\d+)', 
            r'less\s+than\s+(\d+)',
            r'max\s+(\d+)'
        ]
        for pattern in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parameters['max_price'] = int(match.group(1))
                break
        
        # Color extraction
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'orange', 'brown']
        for color in colors:
            if color in query_lower:
                parameters['color'] = color
                break
        
        # Category extraction
        categories = ['dress', 'dresses', 'shirt', 'shirts', 'pants', 'shoes', 'jacket', 'jackets']
        for category in categories:
            if category in query_lower:
                parameters['category'] = category.rstrip('es').rstrip('s')
                break
        
        # Match against intent patterns
        best_intent = None
        best_score = 0
        
        for intent_name, pattern_info in self.intent_patterns.items():
            score = 0
            
            # Check keyword matches
            for keyword in pattern_info['keywords']:
                if keyword in query_lower:
                    score += 1
            
            # Bonus for domain relevance
            if pattern_info['domain'] == 'ecommerce' and any(word in query_lower for word in ['product', 'order', 'buy', 'dress']):
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_intent = intent_name
        
        # Determine safety level and tools based on intent
        if best_intent:
            intent_data = next((i for i in self.intents if i['name'] == best_intent), None)
            if intent_data:
                # Map intent to tools
                method = intent_data['paths'][0]['method'] if intent_data['paths'] else 'GET'
                endpoint = intent_data['paths'][0]['endpoint'] if intent_data['paths'] else '/products'
                
                # Find matching tool
                matching_tools = [tool['name'] for tool in self.tools 
                                if tool['method'] == method and tool['path'] == endpoint]
                
                # Determine safety level
                if best_intent in self.capabilities.get('white', []):
                    safety_level = 'white'
                elif best_intent in self.capabilities.get('grey', []):
                    safety_level = 'grey'
                elif best_intent in self.capabilities.get('black', []):
                    safety_level = 'black'
                else:
                    safety_level = 'white'  # default safe
                
                return QueryAnalysis(
                    intent=best_intent,
                    confidence=min(0.95, 0.6 + (best_score * 0.1)),
                    parameters=parameters,
                    safety_level=safety_level,
                    suggested_tools=matching_tools
                )
        
        # Fallback to basic pattern matching
        if any(word in query_lower for word in ['find', 'search', 'show', 'get', 'list']):
            if 'order' in query_lower:
                return QueryAnalysis(
                    intent="GetMyOrders",
                    confidence=0.8,
                    parameters=parameters,
                    safety_level="white", 
                    suggested_tools=["Get User Orders"]
                )
            else:
                return QueryAnalysis(
                    intent="RetrieveAllProducts",
                    confidence=0.8,
                    parameters=parameters,
                    safety_level="white",
                    suggested_tools=["Get All Products"]
                )
        elif any(word in query_lower for word in ['delete', 'remove']):
            return QueryAnalysis(
                intent="delete_product", 
                confidence=0.9,
                parameters=parameters,
                safety_level="black",
                suggested_tools=["Delete Product"]
            )
        else:
            return QueryAnalysis(
                intent="RetrieveAllProducts",
                confidence=0.6,
                parameters=parameters,
                safety_level="white",
                suggested_tools=["Get All Products"]
            )
    
    def select_tool(self, analysis: QueryAnalysis) -> Optional[Dict[str, Any]]:
        """Select tool with safety checks"""
        # Safety check
        if analysis.safety_level == "black":
            return {
                "error": "Access denied - operation requires elevated permissions",
                "safety_level": "black",
                "intent": analysis.intent
            }
        
        # Find matching tools
        matching_tools = [tool for tool in self.tools if tool['name'] in analysis.suggested_tools]
        
        if not matching_tools:
            return {
                "error": f"No tools available for intent: {analysis.intent}",
                "suggested_tools": analysis.suggested_tools
            }
        
        selected_tool = matching_tools[0]
        
        return {
            "tool": selected_tool,
            "parameters": analysis.parameters,
            "safety_check": True
        }
    
    def simulate_api_call(self, tool: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate actual API call"""
        tool_name = tool['name']
        method = tool['method']
        path = tool['path']
        
        print(f"🔧 Simulating API Call:")
        print(f"   Tool: {tool_name}")
        print(f"   {method} {self.api_base_url}{path}")
        print(f"   Parameters: {parameters}")
        
        # Generate contextual response based on tool and parameters
        if "Get All Products" in tool_name:
            products = []
            
            # Generate products based on filters
            if parameters.get('color') == 'red' and parameters.get('category') == 'dress':
                products = [
                    {
                        "id": "dress_001",
                        "name": "Red Evening Dress",
                        "category": "dress",
                        "color": "red",
                        "price": 299.99,
                        "currency": "USD"
                    },
                    {
                        "id": "dress_002", 
                        "name": "Red Casual Dress",
                        "category": "dress",
                        "color": "red",
                        "price": 149.99,
                        "currency": "USD"
                    }
                ]
            else:
                products = [
                    {
                        "id": "prod_001",
                        "name": "Sample Product",
                        "price": 99.99,
                        "currency": "USD"
                    }
                ]
            
            # Apply price filter
            if parameters.get('max_price'):
                products = [p for p in products if p['price'] <= parameters['max_price']]
            
            return {
                "products": products,
                "total_count": len(products),
                "filters_applied": parameters
            }
        
        elif "Get User Orders" in tool_name:
            return {
                "orders": [
                    {
                        "order_id": "ORD_001",
                        "status": "shipped",
                        "total": 299.99,
                        "currency": "USD"
                    }
                ]
            }
        
        elif "Create" in tool_name:
            return {
                "status": "success",
                "message": f"{tool_name} completed successfully",
                "created_id": "new_123"
            }
        
        else:
            return {
                "status": "success",
                "message": f"Tool {tool_name} executed",
                "parameters": parameters
            }
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main processing method"""
        print(f"🎯 Processing: '{user_query}'")
        print()
        
        # Step 1: Analyze
        analysis = self.analyze_query(user_query)
        print(f"📋 Query Analysis:")
        print(f"   Intent: {analysis.intent}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Parameters: {analysis.parameters}")
        print(f"   Safety Level: {analysis.safety_level}")
        print(f"   Tools: {analysis.suggested_tools}")
        print()
        
        # Step 2: Select tool
        selection = self.select_tool(analysis)
        if "error" in selection:
            print(f"❌ {selection['error']}")
            return selection
        
        # Step 3: Execute
        result = self.simulate_api_call(selection['tool'], selection['parameters'])
        print(f"✅ Success!")
        print()
        
        return {
            "query": user_query,
            "analysis": {
                "intent": analysis.intent,
                "confidence": analysis.confidence,
                "safety_level": analysis.safety_level
            },
            "tool_used": selection['tool']['name'],
            "response": result
        }

def test_enhanced_agent():
    """Test the enhanced agent"""
    agent = EnhancedMCPAgentToolPicker()
    
    test_queries = [
        "Find me red dresses under 500 USD",
        "Show me my orders", 
        "Get all products",
        "Delete a product",
        "Create a new product",
        "Find blue shirts under 100 dollars"
    ]
    
    for query in test_queries:
        print("=" * 70)
        result = agent.process_query(query)
        print(f"Result: {json.dumps(result, indent=2)}")
        print()

if __name__ == "__main__":
    test_enhanced_agent()
