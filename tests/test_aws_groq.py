#!/usr/bin/env python3
"""
Process AWS CloudSearch API with Groq LLM
"""

import json
import os
import yaml
import requests
from pathlib import Path

def process_aws_with_groq():
    """Process AWS API with Groq LLM"""
    
    print("🚀 Processing AWS CloudSearch API with Groq LLM")
    print("=" * 60)
    
    # Groq configuration
    # Use environment variable for API key
    groq_api_key = os.getenv("GROQ_API_KEY", "your-api-key-here")
    groq_model = "llama3-70b-8192"
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    # Load AWS API spec
    with open("sample_data/test_api.yaml", 'r') as f:
        api_spec = yaml.safe_load(f)
    
    # AWS endpoints
    aws_endpoints = [
        {
            "method": "GET",
            "path": "/2013-01-01/search",
            "operation": "Search",
            "description": "Search documents in CloudSearch domain with various query parsers and options"
        },
        {
            "method": "GET", 
            "path": "/2013-01-01/suggest",
            "operation": "Suggest",
            "description": "Get autocomplete suggestions based on partial query strings"
        },
        {
            "method": "POST",
            "path": "/2013-01-01/documents/batch",
            "operation": "UploadDocuments", 
            "description": "Upload batch of documents to search domain for indexing"
        }
    ]
    
    enriched_intents = []
    capabilities = {"white": [], "grey": [], "black": []}
    mcp_tools = []
    
    for i, endpoint in enumerate(aws_endpoints, 1):
        print(f"🧠 Processing {i}/3: {endpoint['operation']}")
        
        # Create prompt for intent generation
        prompt = f"""
        Analyze this AWS CloudSearch API endpoint and create semantic information:
        
        Operation: {endpoint['operation']}
        Method: {endpoint['method']}
        Path: {endpoint['path']}
        Description: {endpoint['description']}
        
        This is part of AWS CloudSearch service for document search and indexing.
        
        Create a JSON response with:
        {{
            "semantic_name": "Clear semantic name (like SearchDocuments, GetSuggestions, UploadDocuments)",
            "description": "Business-friendly description",
            "business_context": "How this fits in document search workflow",
            "complexity": "simple|moderate|complex",
            "user_context": "developer|admin|end_user",
            "safety_level": "white|grey|black",
            "domain_context": "AWS CloudSearch operation for document search and management"
        }}
        
        Safety levels:
        - white: Safe read operations (search, suggest)
        - grey: Moderate write operations (upload documents)
        - black: Dangerous operations (delete, admin functions)
        """
        
        payload = {
            "model": groq_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(groq_url, headers=headers, json=payload)
            if response.status_code == 200:
                llm_response = response.json()
                content = llm_response['choices'][0]['message']['content']
                
                # Extract JSON from response
                try:
                    # Find JSON in the response
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = content[start:end]
                        intent_data = json.loads(json_str)
                        
                        # Create enriched intent
                        enriched_intent = {
                            "name": intent_data['semantic_name'],
                            "details": {
                                "summary": intent_data['description'],
                                "complexity": intent_data['complexity'],
                                "user_context": intent_data['user_context'],
                                "domain": "aws_cloudsearch",
                                "business_context": intent_data['business_context']
                            },
                            "confidence": 0.95,
                            "paths": [
                                {
                                    "method": endpoint['method'],
                                    "endpoint": endpoint['path']
                                }
                            ],
                            "requirements": {
                                "authentication": [
                                    {
                                        "type": "aws_signature_v4",
                                        "required": True
                                    }
                                ],
                                "permissions": [
                                    {
                                        "level": intent_data['safety_level'],
                                        "action": intent_data['semantic_name']
                                    }
                                ]
                            }
                        }
                        
                        enriched_intents.append(enriched_intent)
                        
                        # Add to capabilities
                        safety_level = intent_data['safety_level']
                        capabilities[safety_level].append(intent_data['semantic_name'])
                        
                        # Create MCP tool
                        mcp_tool = {
                            "name": intent_data['semantic_name'],
                            "description": intent_data['description'],
                            "method": endpoint['method'],
                            "path": endpoint['path'],
                            "safety_level": safety_level,
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query or parameter"
                                    }
                                },
                                "required": []
                            },
                            "metadata": {
                                "complexity": intent_data['complexity'],
                                "confidence": 0.95,
                                "aws_service": "cloudsearch",
                                "operation_id": endpoint['operation']
                            }
                        }
                        
                        # Add specific parameters based on operation
                        if endpoint['operation'] == 'Search':
                            mcp_tool['input_schema']['properties'].update({
                                "q": {"type": "string", "description": "Search query"},
                                "queryParser": {"type": "string", "enum": ["simple", "structured", "lucene", "dismax"]},
                                "size": {"type": "integer", "description": "Number of results"},
                                "start": {"type": "integer", "description": "Starting index"}
                            })
                            mcp_tool['input_schema']['required'] = ["q"]
                            
                        elif endpoint['operation'] == 'Suggest':
                            mcp_tool['input_schema']['properties'].update({
                                "q": {"type": "string", "description": "Partial query string"},
                                "suggester": {"type": "string", "description": "Suggester name"},
                                "size": {"type": "integer", "description": "Max suggestions"}
                            })
                            mcp_tool['input_schema']['required'] = ["q", "suggester"]
                            
                        elif endpoint['operation'] == 'UploadDocuments':
                            mcp_tool['input_schema']['properties'].update({
                                "documents": {"type": "string", "description": "JSON or XML batch of documents"},
                                "contentType": {"type": "string", "enum": ["application/json", "application/xml"]}
                            })
                            mcp_tool['input_schema']['required'] = ["documents", "contentType"]
                        
                        mcp_tools.append(mcp_tool)
                        
                        print(f"   ✅ {intent_data['semantic_name']} (Safety: {safety_level})")
                        
                    else:
                        print(f"   ⚠️ No JSON found in response for {endpoint['operation']}")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON parse error for {endpoint['operation']}: {e}")
                    
            else:
                print(f"   ❌ Groq API error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error processing {endpoint['operation']}: {e}")
    
    # Save results
    print("\n💾 Saving results...")
    
    import os
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/aws_enriched_intents.json", 'w') as f:
        json.dump(enriched_intents, f, indent=2)
    
    with open("outputs/aws_capabilities.json", 'w') as f:
        json.dump(capabilities, f, indent=2)
    
    with open("outputs/aws_mcp_tools.json", 'w') as f:
        json.dump(mcp_tools, f, indent=2)
    
    print("=" * 60)
    print("🎉 AWS CloudSearch Analysis Complete!")
    print(f"   📊 Endpoints Processed: {len(aws_endpoints)}")
    print(f"   🧠 Intents Generated: {len(enriched_intents)}")
    print(f"   🛠️ MCP Tools Created: {len(mcp_tools)}")
    print(f"   🛡️ Safety Classification:")
    print(f"      White (Safe): {len(capabilities['white'])} - {capabilities['white']}")
    print(f"      Grey (Moderate): {len(capabilities['grey'])} - {capabilities['grey']}")
    print(f"      Black (Dangerous): {len(capabilities['black'])} - {capabilities['black']}")
    
    return {
        "intents": enriched_intents,
        "capabilities": capabilities,
        "tools": mcp_tools
    }

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml
    
    result = process_aws_with_groq()
