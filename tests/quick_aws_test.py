#!/usr/bin/env python3
"""
Quick test to parse the AWS API and see what we get
"""

import json
import yaml
from pathlib import Path

def quick_aws_test():
    """Quick test of AWS API parsing"""
    
    print("🔍 Quick AWS CloudSearch API Analysis")
    print("=" * 50)
    
    # Load the YAML file
    api_spec_path = "sample_data/test_api.yaml"
    
    with open(api_spec_path, 'r') as f:
        api_spec = yaml.safe_load(f)
    
    # Basic info
    info = api_spec.get('info', {})
    print(f"📋 API Info:")
    print(f"   Title: {info.get('title', 'Unknown')}")
    print(f"   Version: {info.get('version', 'Unknown')}")
    print(f"   Service: {info.get('x-serviceName', 'Unknown')}")
    print()
    
    # Extract endpoints
    endpoints = []
    paths = api_spec.get('paths', {})
    
    print(f"🛣️ Found {len(paths)} path(s):")
    for path, methods in paths.items():
        print(f"   Path: {path}")
        for method, details in methods.items():
            if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                operation_id = details.get('operationId', f"{method}_{path}")
                description = details.get('description', '')[:100] + "..."
                params = len(details.get('parameters', []))
                
                print(f"      {method.upper()}: {operation_id}")
                print(f"         Description: {description}")
                print(f"         Parameters: {params}")
                
                endpoint = {
                    'method': method.upper(),
                    'path': path,
                    'operation_id': operation_id,
                    'description': details.get('description', ''),
                    'parameters': details.get('parameters', []),
                    'request_body': details.get('requestBody'),
                }
                endpoints.append(endpoint)
        print()
    
    print(f"✅ Total Endpoints Found: {len(endpoints)}")
    
    # Show endpoint details
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n📍 Endpoint {i}: {endpoint['operation_id']}")
        print(f"   Method: {endpoint['method']}")
        print(f"   Path: {endpoint['path']}")
        print(f"   Parameters: {len(endpoint['parameters'])}")
        
        # Show some parameters
        if endpoint['parameters']:
            print("   Key Parameters:")
            for param in endpoint['parameters'][:3]:  # Show first 3
                name = param.get('name', 'unknown')
                required = param.get('required', False)
                desc = param.get('description', '')[:50]
                print(f"      - {name} {'(required)' if required else '(optional)'}: {desc}...")
    
    return endpoints

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml
    
    endpoints = quick_aws_test()
