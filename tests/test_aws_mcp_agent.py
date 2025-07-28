#!/usr/bin/env python3
"""
Test MCP Agent Tool Picker with AWS CloudSearch API
"""

import json
from pathlib import Path

class AWSMCPAgentToolPicker:
    """MCP Agent specialized for AWS CloudSearch operations"""
    
    def __init__(self, analysis_dir: str = "outputs"):
        self.analysis_dir = Path(analysis_dir)
        self.capabilities = self._load_aws_capabilities()
        self.intents = self._load_aws_intents()
        self.tools = self._load_aws_tools()
        
        # AWS CloudSearch configuration
        self.aws_region = "us-east-1"
        self.api_base_url = f"https://cloudsearchdomain.{self.aws_region}.amazonaws.com"
        
    def _load_aws_capabilities(self):
        try:
            with open(self.analysis_dir / "aws_capabilities.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"white": [], "grey": [], "black": []}
    
    def _load_aws_intents(self):
        try:
            with open(self.analysis_dir / "aws_enriched_intents.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _load_aws_tools(self):
        try:
            with open(self.analysis_dir / "aws_mcp_tools.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def analyze_query(self, user_query: str):
        """Analyze user query for AWS CloudSearch operations"""
        query_lower = user_query.lower()
        
        # AWS CloudSearch specific patterns
        if any(word in query_lower for word in ['search', 'find', 'query', 'documents']):
            return {
                "intent": "SearchDocuments",
                "confidence": 0.9,
                "parameters": self._extract_search_params(user_query),
                "safety_level": "white",
                "suggested_tools": ["SearchDocuments"]
            }
        elif any(word in query_lower for word in ['suggest', 'autocomplete', 'completion']):
            return {
                "intent": "GetSuggestions", 
                "confidence": 0.85,
                "parameters": self._extract_suggest_params(user_query),
                "safety_level": "white",
                "suggested_tools": ["GetSuggestions"]
            }
        elif any(word in query_lower for word in ['upload', 'index', 'add documents', 'batch']):
            return {
                "intent": "UploadDocuments",
                "confidence": 0.9,
                "parameters": self._extract_upload_params(user_query),
                "safety_level": "grey",
                "suggested_tools": ["UploadDocuments"]
            }
        else:
            # Default to search
            return {
                "intent": "SearchDocuments",
                "confidence": 0.6,
                "parameters": {"q": user_query},
                "safety_level": "white", 
                "suggested_tools": ["SearchDocuments"]
            }
    
    def _extract_search_params(self, query):
        """Extract search parameters from query"""
        params = {}
        
        # Extract actual search term
        search_terms = []
        for word in ['search for', 'find', 'query', 'documents containing']:
            if word in query.lower():
                parts = query.lower().split(word)
                if len(parts) > 1:
                    search_terms.append(parts[1].strip())
        
        if search_terms:
            params['q'] = search_terms[0]
        else:
            # Use the whole query as search term
            params['q'] = query
        
        # Default parameters
        params['queryParser'] = 'simple'
        params['size'] = 10
        
        return params
    
    def _extract_suggest_params(self, query):
        """Extract suggestion parameters"""
        params = {}
        
        # Extract partial term
        if 'suggest' in query.lower():
            parts = query.lower().split('suggest')
            if len(parts) > 1:
                partial = parts[1].strip().strip('for').strip()
                params['q'] = partial
        
        # Default suggester (would be configured in real CloudSearch)
        params['suggester'] = 'default_suggester'
        params['size'] = 5
        
        return params
    
    def _extract_upload_params(self, query):
        """Extract upload parameters"""
        params = {}
        
        # Default to JSON format
        params['contentType'] = 'application/json'
        
        # In real implementation, would extract document content
        params['documents'] = '{"type": "add", "id": "doc1", "fields": {"title": "Sample Document"}}'
        
        return params
    
    def select_tool(self, analysis):
        """Select appropriate AWS CloudSearch tool"""
        if analysis['safety_level'] == 'black':
            return {
                "error": "Operation not permitted - dangerous AWS operation blocked",
                "safety_level": "black"
            }
        
        # Find matching tool
        matching_tools = [tool for tool in self.tools if tool['name'] in analysis['suggested_tools']]
        
        if not matching_tools:
            return {
                "error": f"No AWS CloudSearch tools found for intent: {analysis['intent']}"
            }
        
        selected_tool = matching_tools[0]
        
        return {
            "tool": selected_tool,
            "parameters": analysis['parameters'],
            "safety_check": True,
            "aws_region": self.aws_region
        }
    
    def simulate_aws_call(self, tool, parameters):
        """Simulate AWS CloudSearch API call"""
        tool_name = tool['name']
        method = tool['method']
        path = tool['path']
        
        print(f"🔧 Simulating AWS CloudSearch API Call:")
        print(f"   Service: AWS CloudSearch")
        print(f"   Tool: {tool_name}")
        print(f"   {method} {self.api_base_url}{path}")
        print(f"   Region: {self.aws_region}")
        print(f"   Parameters: {parameters}")
        print(f"   Authentication: AWS Signature v4")
        
        # Generate realistic AWS CloudSearch responses
        if tool_name == "SearchDocuments":
            return {
                "status": {
                    "timems": 45,
                    "rid": "search_rid_12345"
                },
                "hits": {
                    "found": 1247,
                    "start": 0,
                    "hit": [
                        {
                            "id": "doc_001",
                            "fields": {
                                "title": ["AWS CloudSearch Documentation"],
                                "content": ["Learn how to use AWS CloudSearch for document indexing..."],
                                "category": ["documentation"]
                            }
                        },
                        {
                            "id": "doc_002", 
                            "fields": {
                                "title": ["CloudSearch Best Practices"],
                                "content": ["Tips for optimizing your CloudSearch domain..."],
                                "category": ["guide"]
                            }
                        }
                    ]
                },
                "facets": {},
                "query_info": {
                    "query": parameters.get('q', ''),
                    "parser": parameters.get('queryParser', 'simple')
                }
            }
        
        elif tool_name == "GetSuggestions":
            return {
                "status": {
                    "timems": 12,
                    "rid": "suggest_rid_67890"
                },
                "suggest": {
                    "query": parameters.get('q', ''),
                    "found": 5,
                    "suggestions": [
                        {"suggestion": f"{parameters.get('q', '')}umentation", "score": 95, "id": "doc_001"},
                        {"suggestion": f"{parameters.get('q', '')}ument search", "score": 87, "id": "doc_003"},
                        {"suggestion": f"{parameters.get('q', '')}ument indexing", "score": 82, "id": "doc_004"}
                    ]
                }
            }
        
        elif tool_name == "UploadDocuments":
            return {
                "status": "success",
                "adds": 1,
                "deletes": 0,
                "warnings": [],
                "upload_info": {
                    "contentType": parameters.get('contentType', 'application/json'),
                    "processing_time": "2.3s",
                    "documents_processed": 1
                }
            }
        
        else:
            return {
                "status": "success",
                "message": f"AWS CloudSearch {tool_name} executed",
                "parameters": parameters
            }
    
    def process_query(self, user_query: str):
        """Process user query for AWS CloudSearch"""
        print(f"🎯 AWS CloudSearch Query: '{user_query}'")
        print()
        
        # Analyze
        analysis = self.analyze_query(user_query)
        print(f"📋 Analysis:")
        print(f"   Intent: {analysis['intent']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Parameters: {analysis['parameters']}")
        print(f"   Safety: {analysis['safety_level']}")
        print()
        
        # Select tool
        selection = self.select_tool(analysis)
        if "error" in selection:
            print(f"❌ {selection['error']}")
            return selection
        
        # Execute
        result = self.simulate_aws_call(selection['tool'], selection['parameters'])
        print(f"✅ AWS CloudSearch Operation Complete!")
        print()
        
        return {
            "query": user_query,
            "aws_service": "cloudsearch",
            "tool_used": selection['tool']['name'],
            "region": self.aws_region,
            "analysis": analysis,
            "response": result
        }

def test_aws_mcp_agent():
    """Test AWS MCP Agent with various queries"""
    agent = AWSMCPAgentToolPicker()
    
    test_queries = [
        "Search for documents about machine learning",
        "Find all documents containing 'AWS'",
        "Get suggestions for 'doc'",
        "Upload a batch of new documents",
        "Query the search index for recent articles"
    ]
    
    for query in test_queries:
        print("=" * 70)
        result = agent.process_query(query)
        print(f"Result Summary: {result.get('tool_used', 'Error')} - {result.get('aws_service', 'N/A')}")
        print()

if __name__ == "__main__":
    test_aws_mcp_agent()
