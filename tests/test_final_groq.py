#!/usr/bin/env python3
"""
Final Working Test - Process all 7 endpoints with real Groq LLM
"""

import os
import json
import yaml
import requests
import time
from pathlib import Path
from typing import List, Dict

def process_all_endpoints_with_groq():
    """Process all 7 API endpoints using real Groq LLM."""
    print("🚀 Processing All 7 Endpoints with Real Groq LLM")
    print("=" * 60)
    
    # Load environment
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Use the working API key and format from your test
    api_key = "REDACTED_GROQ_KEY"
    model = "llama3-70b-8192"
    
    print(f"✅ Using API Key: {api_key[:20]}...")
    print(f"✅ Using Model: {model}")
    
    # Load the test API specification
    data_path = Path(__file__).parent / "sample_data" / "test_api.yaml"
    with open(data_path, 'r') as f:
        api_spec = yaml.safe_load(f)
    
    # Extract all endpoints
    endpoints = []
    for path, methods in api_spec.get('paths', {}).items():
        for method, operation in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE']:
                endpoints.append({
                    "method": method.upper(),
                    "path": path,
                    "summary": operation.get('summary', ''),
                    "description": operation.get('description', ''),
                    "operationId": operation.get('operationId', ''),
                    "security": operation.get('security', [])
                })
    
    print(f"\n📋 Found {len(endpoints)} endpoints to process:")
    for i, ep in enumerate(endpoints, 1):
        print(f"   {i}. {ep['method']} {ep['path']} - {ep['summary']}")
    
    # Groq API setup (using your working format)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Process each endpoint with Groq LLM
    processed_tools = []
    enriched_intents = []
    
    print(f"\n🧠 Processing endpoints with Groq LLM...")
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n{i}/{len(endpoints)} Processing: {endpoint['method']} {endpoint['path']}")
        
        # Create enriched intent using LLM
        intent_prompt = f"""Convert this API endpoint to an enriched intent. Return a JSON object with this structure:

{{
  "name": "semantic_name_for_intent",
  "details": {{
    "summary": "what the user wants to accomplish",
    "complexity": "simple|moderate|complex",
    "user_context": "browsing_user|authenticated_user|admin_user",
    "domain": "ecommerce",
    "business_context": "brief business description"
  }},
  "confidence": 0.95
}}

API Endpoint:
Method: {endpoint['method']}
Path: {endpoint['path']}
Summary: {endpoint['summary']}
Description: {endpoint['description']}"""
        
        intent_payload = {
            "model": model,
            "messages": [{"role": "user", "content": intent_prompt}],
            "temperature": 0.1
        }
        
        try:
            # Make API call for intent (using your working format)
            response = requests.post(url, headers=headers, json=intent_payload)
            if response.status_code == 200:
                result = response.json()
                intent_response = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    json_start = intent_response.find('{')
                    json_end = intent_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        intent_data = json.loads(intent_response[json_start:json_end])
                        
                        # Add path and requirements
                        intent_data['paths'] = [{"method": endpoint['method'], "endpoint": endpoint['path']}]
                        intent_data['requirements'] = {
                            "authentication": [{"type": "api_key", "required": True}] if endpoint['security'] else [],
                            "permissions": [{"level": "white" if endpoint['method'] == 'GET' else "grey", "action": intent_data['name']}]
                        }
                        
                        enriched_intents.append(intent_data)
                        print(f"   ✅ Intent: {intent_data.get('name', 'unnamed')}")
                    else:
                        print(f"   ⚠️  Could not parse intent JSON")
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Intent JSON error: {e}")
            else:
                print(f"   ❌ Intent API error: {response.status_code}")
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
            # Create MCP tool using LLM
            tool_prompt = f"""Convert this API endpoint to an MCP tool specification. Return a JSON object with this structure:

{{
  "name": "tool_name",
  "description": "detailed description of what this tool does",
  "method": "{endpoint['method']}",
  "path": "{endpoint['path']}",
  "safety_level": "safe|moderate|dangerous",
  "input_schema": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }},
  "metadata": {{
    "complexity": "simple|moderate|complex",
    "confidence": 0.95
  }}
}}

API Endpoint:
Method: {endpoint['method']}
Path: {endpoint['path']}
Summary: {endpoint['summary']}
Description: {endpoint['description']}"""
            
            tool_payload = {
                "model": model,
                "messages": [{"role": "user", "content": tool_prompt}],
                "temperature": 0.1
            }
            
            # Make API call for tool (using your working format)
            response = requests.post(url, headers=headers, json=tool_payload)
            if response.status_code == 200:
                result = response.json()
                tool_response = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    json_start = tool_response.find('{')
                    json_end = tool_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        tool_data = json.loads(tool_response[json_start:json_end])
                        processed_tools.append(tool_data)
                        print(f"   ✅ Tool: {tool_data.get('name', 'unnamed')}")
                    else:
                        print(f"   ⚠️  Could not parse tool JSON")
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Tool JSON error: {e}")
            else:
                print(f"   ❌ Tool API error: {response.status_code}")
            
            # Another small delay
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error processing endpoint: {e}")
    
    # Generate capabilities based on enriched intents
    capabilities = {
        "white": [],  # Safe operations
        "grey": [],   # Moderate operations  
        "black": []   # Dangerous operations
    }
    
    for intent in enriched_intents:
        intent_name = intent.get('name', 'unnamed')
        method = intent.get('paths', [{}])[0].get('method', 'GET')
        
        if method == 'GET':
            capabilities["white"].append(intent_name)
        elif method in ['POST', 'PUT']:
            capabilities["grey"].append(intent_name)
        elif method == 'DELETE':
            capabilities["black"].append(intent_name)
    
    # Save all results
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Save enriched intents
    with open(outputs_dir / "groq_enriched_intents.json", 'w') as f:
        json.dump(enriched_intents, f, indent=2)
    
    # Save capabilities
    with open(outputs_dir / "groq_capabilities.json", 'w') as f:
        json.dump(capabilities, f, indent=2)
    
    # Save MCP tools
    with open(outputs_dir / "groq_mcp_tools.json", 'w') as f:
        json.dump(processed_tools, f, indent=2)
    
    # Results summary
    print(f"\n🎉 Real Groq LLM Processing Complete!")
    print(f"=" * 60)
    print(f"📊 RESULTS COMPARISON:")
    print(f"   📥 Input: {len(endpoints)} API endpoints")
    print(f"   📤 Output: {len(processed_tools)} MCP tools")
    print(f"   📈 Success Rate: {len(processed_tools)}/{len(endpoints)} = {len(processed_tools)/len(endpoints)*100:.1f}%")
    print(f"")
    print(f"💾 Files Generated:")
    print(f"   - groq_enriched_intents.json ({len(enriched_intents)} intents)")
    print(f"   - groq_capabilities.json ({len(capabilities)} permission levels)")
    print(f"   - groq_mcp_tools.json ({len(processed_tools)} tools)")
    print(f"")
    print(f"🆚 COMPARISON WITH MOCK DATA:")
    print(f"   - Mock test: 2/7 endpoints extracted (28.6%)")
    print(f"   - Groq LLM:  {len(processed_tools)}/7 endpoints extracted ({len(processed_tools)/7*100:.1f}%)")
    print(f"")
    
    if len(processed_tools) > 2:
        print(f"✅ SUCCESS! Real LLM extracted {len(processed_tools) - 2} MORE tools than mock data!")
    else:
        print(f"⚠️  Real LLM extracted {len(processed_tools)} tools (same as mock)")
    
    print(f"\n📋 Generated Tools:")
    for tool in processed_tools:
        print(f"   - {tool.get('name', 'unnamed')} ({tool.get('method', 'UNKNOWN')} {tool.get('path', 'unknown')})")
    
    return len(processed_tools)

if __name__ == "__main__":
    process_all_endpoints_with_groq()
