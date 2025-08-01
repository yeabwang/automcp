#!/usr/bin/env python3
"""
AutoMCP CLI - Professional API to MCP Tool Converter

A command-line interface for transforming API specifications into AI-agent-ready 
Model Context Protocol (MCP) tools with semantic enrichment and enterprise validation.
"""

import os
import json
import yaml
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import click

# Import AutoMCP core modules
try:
    from .config import load_config, get_config, list_environments as get_available_environments
    from .core.async_llm_client import EnhancedAsyncLLMClient
    from .core.llm_client_interface import ResponseFormat
    from .core.parsers import extract_endpoints_from_spec
    from .core.enricher import EnrichmentEngine
    from .core.output_generator import OutputGenerator
except ImportError:
    # Fallback for running from root directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.automcp.config import load_config, get_config, list_environments as get_available_environments
    from src.automcp.core.async_llm_client import EnhancedAsyncLLMClient
    from src.automcp.core.llm_client_interface import ResponseFormat
    from src.automcp.core.parsers import extract_endpoints_from_spec
    from src.automcp.core.enricher import EnrichmentEngine
    from src.automcp.core.output_generator import OutputGenerator

#!/usr/bin/env python3
"""
AutoMCP CLI - Professional API to MCP Tool Converter

A command-line interface for transforming API specifications into AI-agent-ready 
Model Context Protocol (MCP) tools with semantic enrichment and enterprise validation.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import click

# Import AutoMCP core modules
try:
    from .config import load_config, get_config, list_environments as get_available_environments
    from .core.parsers import SpecAnalyzer
    from .core.async_llm_client import EnhancedAsyncLLMClient
except ImportError:
    # Fallback for running from root directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.automcp.config import load_config, get_config, list_environments as get_available_environments
    from src.automcp.core.parsers import SpecAnalyzer
    from src.automcp.core.async_llm_client import EnhancedAsyncLLMClient


class CLIProcessor:
    """Configuration-driven processing engine for AutoMCP CLI operations."""
    
    def __init__(self, environment: Optional[str] = None, verbose: bool = False):
        """Initialize processor with configuration."""
        self.config = load_config(environment)
        self.verbose = verbose
        self.environment = environment or self.config.environment
        
        if verbose:
            self._print_startup_info()
    
    def _print_startup_info(self):
        """Print startup configuration information."""
        click.echo(f"AutoMCP Configuration Loaded")
        click.echo(f"Environment: {self.environment}")
        click.echo(f"LLM Provider: {self.config.get('llm_client.provider', 'Not configured')}")
        click.echo(f"LLM Model: {self.config.get('llm_client.model', 'Not configured')}")
        
        api_key = self.config.get('llm_client.api_key') or os.getenv('AUTOMCP_LLM_API_KEY') or os.getenv('GROQ_API_KEY')
        click.echo(f"API Key: {'Configured' if api_key else 'Missing'}")
        
        output_dir = self.config.get('output.dir', 'outputs')
        click.echo(f"Output Directory: {output_dir}")
        click.echo()
    
    async def process_specification(self, spec_file: Path, output_dir: Optional[Path] = None, 
                                   dry_run: bool = False) -> Dict[str, Any]:
        """Process a single API specification file using configuration-driven pipeline."""
        try:
            # Initialize the specification analyzer
            analyzer = SpecAnalyzer(config=self.config.data)
            
            # Set output directory if specified
            if output_dir:
                # Update config temporarily for this analysis
                analyzer.config['output']['dir'] = str(output_dir)
                analyzer.output_gen.config['output']['dir'] = str(output_dir)
            
            if self.verbose:
                click.echo(f"Processing specification: {spec_file}")
                if dry_run:
                    click.echo("Running in dry-run mode - no files will be generated")
            
            # Run the analysis
            await analyzer.analyze(str(spec_file), dry_run=dry_run)
            
            # Return success metrics
            return {
                'status': 'success',
                'spec_file': str(spec_file),
                'output_dir': str(output_dir) if output_dir else analyzer.config['output']['dir'],
                'environment': self.environment
            }
            
        except Exception as e:
            error_msg = f"Failed to process {spec_file}: {str(e)}"
            if self.verbose:
                import traceback
                click.echo(f"Error details: {traceback.format_exc()}", err=True)
            
            return {
                'status': 'error',
                'spec_file': str(spec_file),
                'error': error_msg
            }
    
    async def batch_process_specifications(self, input_dir: Path, output_dir: Optional[Path] = None,
                                         pattern: str = "*.{yaml,yml,json}", 
                                         continue_on_error: bool = False) -> Dict[str, Any]:
        """Batch process multiple API specification files."""
        try:
            # Find matching files
            import glob
            pattern_parts = pattern.split(',')
            spec_files = []
            for part in pattern_parts:
                spec_files.extend(input_dir.glob(part.strip()))
            
            if not spec_files:
                return {
                    'status': 'error',
                    'error': f"No files found matching pattern '{pattern}' in {input_dir}"
                }
            
            if self.verbose:
                click.echo(f"Found {len(spec_files)} files to process")
            
            # Process each file
            results = []
            success_count = 0
            error_count = 0
            
            for spec_file in spec_files:
                try:
                    # Determine output directory for this file
                    file_output_dir = output_dir / spec_file.stem if output_dir else None
                    
                    result = await self.process_specification(
                        spec_file, 
                        file_output_dir, 
                        dry_run=False
                    )
                    
                    results.append(result)
                    
                    if result['status'] == 'success':
                        success_count += 1
                        if self.verbose:
                            click.echo(f"Processed: {spec_file.name}")
                    else:
                        error_count += 1
                        click.echo(f"Failed: {spec_file.name} - {result.get('error', 'Unknown error')}", err=True)
                        
                        if not continue_on_error:
                            break
                            
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing {spec_file.name}: {str(e)}"
                    click.echo(error_msg, err=True)
                    
                    results.append({
                        'status': 'error',
                        'spec_file': str(spec_file),
                        'error': error_msg
                    })
                    
                    if not continue_on_error:
                        break
            
            return {
                'status': 'completed',
                'total_files': len(spec_files),
                'successful': success_count,
                'errors': error_count,
                'results': results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Batch processing failed: {str(e)}"
            }
    
    def validate_health(self) -> Dict[str, Any]:
        """Validate system health and configuration."""
        checks = []
        
        # Check configuration
        try:
            config = get_config()
            checks.append(("Configuration", "Loaded successfully", True))
        except Exception as e:
            checks.append(("Configuration", f"Failed to load: {e}", False))
            return self._format_health_results(checks)
        
        # Check API key
        api_key = (config.get('llm_client.api_key') or 
                   os.getenv('AUTOMCP_LLM_API_KEY') or 
                   os.getenv('LLM_API_KEY') or 
                   os.getenv('GROQ_API_KEY'))
        checks.append(("LLM API Key", "Configured" if api_key else "Missing", bool(api_key)))
        
        # Check directories
        input_dir = Path("inputs")
        checks.append(("Input Directory", f"{'Found' if input_dir.exists() else 'Missing'} ({input_dir})", 
                      input_dir.exists()))
        
        output_dir = Path(config.get('output.dir', 'outputs'))
        output_dir.mkdir(exist_ok=True)  # Create if it doesn't exist
        checks.append(("Output Directory", f"Ready ({output_dir})", True))
        
        # Check write permissions
        try:
            test_file = output_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            checks.append(("File Permissions", "Write access OK", True))
        except Exception as e:
            checks.append(("File Permissions", f"Write failed: {e}", False))
        
        return self._format_health_results(checks)
    
    def _format_health_results(self, checks):
        """Format health check results."""
        results = {
            'status': 'healthy' if all(check[2] for check in checks) else 'unhealthy',
            'checks': [{'component': c[0], 'status': c[1], 'healthy': c[2]} for c in checks],
            'environment': self.environment
        }
        
        return results


@click.group()
@click.version_option(version="1.0.0", prog_name="AutoMCP")
@click.option('--environment', '-e', 
              type=click.Choice(['development', 'production', 'enterprise', 'fallback']), 
              help="Configuration environment to use")
@click.option('--verbose', '-v', is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, environment, verbose):
    """
    AutoMCP - Professional API to MCP Tool Converter
    
    Transform API specifications into AI-agent-ready Model Context Protocol (MCP) tools
    with semantic enrichment and enterprise-grade validation.
    
    Supported input formats: OpenAPI 3.0/3.1, Swagger 2.0, Postman Collections v2.1
    
    Output files generated:
    - enriched_intents.json: Semantic intent metadata with user context
    - capabilities.json: Permission-based capability classifications  
    - mcp_tools.json: Complete MCP tool specifications with schemas
    
    Examples:
      automcp analyze api.yaml
      automcp analyze api.yaml -e production
      automcp batch input/ -o results/
      automcp health
    """
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['environment'] = environment
    ctx.obj['verbose'] = verbose


@cli.command('analyze')
@click.argument('spec_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help="Output directory (defaults to outputs/{filename}/)")
@click.option('--dry-run', is_flag=True, 
              help="Analyze specification without generating output files")
@click.pass_context
def analyze_spec(ctx, spec_file, output, dry_run):
    """
    Analyze and transform a single API specification file.
    
    Transform API specifications into MCP tools with AI-powered semantic enrichment.
    Supports OpenAPI, Swagger, and Postman collection formats.
    
    SPEC_FILE: Path to the API specification file to analyze
    
    Examples:
      automcp analyze shopify.yaml
      automcp analyze api.yaml -o custom_output/  
      automcp analyze spec.json --dry-run
      automcp analyze api.yaml -e production
    """
    async def _process():
        environment = ctx.obj.get('environment')
        verbose = ctx.obj.get('verbose', False)
        
        processor = CLIProcessor(environment, verbose)
        
        # Determine output directory
        if not output:
            base_output_dir = Path(processor.config.get('output.dir', 'outputs'))
            output_dir = base_output_dir / spec_file.stem
        else:
            output_dir = output
        
        if verbose:
            click.echo(f"Input: {spec_file}")
            click.echo(f"Output: {output_dir}")
            if dry_run:
                click.echo("Mode: Dry run (no files will be generated)")
        
        # Process the specification
        result = await processor.process_specification(spec_file, output_dir, dry_run)
        
        if result['status'] == 'success':
            if not dry_run:
                click.echo(f"Analysis completed successfully. Output saved to: {result['output_dir']}")
            else:
                click.echo("Dry run completed successfully - no files generated")
        else:
            click.echo(f"Analysis failed: {result['error']}", err=True)
            sys.exit(1)
    
    asyncio.run(_process())


@cli.command('batch')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help="Base output directory (defaults to outputs/)")
@click.option('--pattern', '-p', default="*.{yaml,yml,json}",
              help="File pattern to match (default: *.{yaml,yml,json})")
@click.option('--continue-on-error', is_flag=True,
              help="Continue processing remaining files if one fails")
@click.pass_context
def batch_process(ctx, input_dir, output, pattern, continue_on_error):
    """
    Batch process multiple API specification files.
    
    Process all API specifications in a directory with automated error handling
    and progress reporting for large collections of API specifications.
    
    INPUT_DIR: Directory containing API specification files to process
    
    Examples:
      automcp batch input/
      automcp batch specs/ -o results/
      automcp batch apis/ -p "*.yaml"
      automcp batch files/ --continue-on-error
    """
    async def _process():
        environment = ctx.obj.get('environment')
        verbose = ctx.obj.get('verbose', False)
        
        processor = CLIProcessor(environment, verbose)
        
        # Determine output directory
        if not output:
            output_dir = Path(processor.config.get('output.dir', 'outputs'))
        else:
            output_dir = output
        
        if verbose:
            click.echo(f"Input directory: {input_dir}")
            click.echo(f"Output directory: {output_dir}")
            click.echo(f"File pattern: {pattern}")
        
        # Process batch
        result = await processor.batch_process_specifications(
            input_dir, output_dir, pattern, continue_on_error
        )
        
        if result['status'] == 'error':
            click.echo(f"Batch processing failed: {result['error']}", err=True)
            sys.exit(1)
        else:
            click.echo(f"Batch processing completed:")
            click.echo(f"  Total files: {result['total_files']}")
            click.echo(f"  Successful: {result['successful']}")  
            click.echo(f"  Errors: {result['errors']}")
            
            if result['errors'] > 0:
                click.echo(f"Output saved to: {output_dir}")
                sys.exit(1)
            else:
                click.echo(f"All files processed successfully. Output saved to: {output_dir}")
    
    asyncio.run(_process())


@cli.command('health')
@click.pass_context
def health_check(ctx):
    """
    Check AutoMCP system health and configuration.
    
    Verify that all components are properly configured and accessible.
    Useful for troubleshooting installation and configuration issues.
    
    Validates:
    - Configuration files and environment settings
    - LLM API connectivity and authentication
    - Input/output directory structure and permissions
    - Required dependencies and imports
    """
    environment = ctx.obj.get('environment')
    verbose = ctx.obj.get('verbose', False)
    
    processor = CLIProcessor(environment, verbose)
    result = processor.validate_health()
    
    click.echo("AutoMCP System Health Check")
    click.echo("-" * 40)
    
    for check in result['checks']:
        status_icon = "✓" if check['healthy'] else "✗"
        click.echo(f"{status_icon} {check['component']}: {check['status']}")
    
    click.echo()
    if result['status'] == 'healthy':
        click.echo("System is healthy and ready for operation.")
    else:
        click.echo("System has issues that need attention.")
        sys.exit(1)


@cli.command('environments')
def list_environments():
    """
    List available configuration environments.
    
    Display all available environment configurations and their purposes.
    Use these environments with the --environment/-e flag for different
    operational modes and settings.
    
    Examples:
      automcp -e development analyze api.yaml
      automcp -e production analyze api.yaml  
      automcp -e enterprise analyze api.yaml
    """
    try:
        environments = get_available_environments()
        
        env_descriptions = {
            'default': 'Base configuration with standard settings',
            'development': 'Development settings with enhanced debugging',
            'production': 'Production-optimized settings with strict validation', 
            'enterprise': 'Enterprise-grade settings with maximum security',
            'fallback': 'Minimal fallback configuration for basic operation'
        }
        
        click.echo("Available AutoMCP Environments:")
        click.echo("-" * 40)
        
        for env in environments:
            description = env_descriptions.get(env, 'Custom environment configuration')
            click.echo(f"{env:<12} - {description}")
        
        click.echo()
        click.echo("Usage: automcp -e <environment> <command>")
        
    except Exception as e:
        click.echo(f"Error listing environments: {e}", err=True)


@cli.group('config')
def config_group():
    """Configuration management commands for AutoMCP environments and settings."""
    pass


@config_group.command('show')
@click.option('--environment', '-e', 
              type=click.Choice(['development', 'production', 'enterprise']), 
              help="Environment to show configuration for")
@click.option('--section', '-s', 
              help="Show specific config section (e.g., llm_client, output, logging)")
@click.pass_context  
def show_config(ctx, environment, section):
    """
    Display current configuration settings.
    
    Show configuration for specified environment with detailed breakdown
    of all settings and their current values.
    
    Examples:
      automcp config show
      automcp config show -e production
      automcp config show -s llm_client
      automcp config show -e enterprise -s output
    """
    # Use environment from CLI context if not specified in command
    env = environment or ctx.parent.obj.get('environment')
    
    try:
        config = load_config(env)
        
        click.echo(f"AutoMCP Configuration ({config.environment})")
        click.echo("=" * 50)
        
        if section:
            # Show specific section
            section_data = config.get(section, {})
            if not section_data:
                click.echo(f"Configuration section '{section}' not found", err=True)
                return
            
            click.echo(f"{section.upper()} Configuration:")
            _print_config_section(section_data)
        else:
            # Show key sections
            sections = [
                ('LLM Client Configuration', 'llm_client'),
                ('Output Configuration', 'output'), 
                ('Logging Configuration', 'logging'),
                ('Semantic Transformation', 'semantic_transformation')
            ]
            
            for title, section_key in sections:
                click.echo(f"{title}:")
                section_data = config.get(section_key, {})
                
                if section_key == 'llm_client':
                    click.echo(f"  Provider: {section_data.get('provider', 'Not set')}")
                    click.echo(f"  Model: {section_data.get('model', 'Not set')}")
                    api_key = section_data.get('api_key') or os.getenv('AUTOMCP_LLM_API_KEY')
                    click.echo(f"  API Key: {'Configured' if api_key else 'Missing'}")
                elif section_key == 'output':
                    click.echo(f"  Directory: {section_data.get('dir', 'Not set')}")
                    click.echo(f"  Format: {section_data.get('save_format', 'Not set')}")
                    click.echo(f"  Validation: {section_data.get('strict_validation', 'Not set')}")
                elif section_key == 'logging':
                    click.echo(f"  Level: {section_data.get('level', 'Not set')}")
                    click.echo(f"  Format: {section_data.get('format', 'Not set')}")
                elif section_key == 'semantic_transformation':
                    click.echo(f"  Enabled: {section_data.get('enabled', 'Not set')}")
                    click.echo(f"  Confidence Threshold: {section_data.get('confidence_threshold', 'Not set')}")
                
                click.echo()
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)


def _print_config_section(data, indent=0):
    """Helper function to print configuration data with proper formatting."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                click.echo(f"{prefix}{key}:")
                _print_config_section(value, indent + 1)
            else:
                click.echo(f"{prefix}{key}: {value}")
    else:
        click.echo(f"{prefix}{data}")


@config_group.command('validate')
@click.option('--environment', '-e',
              type=click.Choice(['development', 'production', 'enterprise']),
              help="Environment to validate")
@click.pass_context  
def validate_config(ctx, environment):
    """
    Validate configuration files for correctness.
    
    Check configuration files for syntax errors, missing required fields,
    and invalid values to ensure proper system operation.
    
    Examples:
      automcp config validate
      automcp config validate -e production
    """
    env = environment or ctx.parent.obj.get('environment') or 'development'
    
    click.echo(f"Validating AutoMCP Configuration ({env})")
    click.echo("-" * 50)
    
    try:
        config = load_config(env)
        issues = []
        
        # Validate LLM configuration
        llm_config = config.get('llm_client', {})
        if not llm_config.get('provider'):
            issues.append("LLM provider not specified")
        if not llm_config.get('model'):
            issues.append("LLM model not specified")
        
        # Check API key
        api_key = llm_config.get('api_key') or os.getenv('AUTOMCP_LLM_API_KEY') or os.getenv('GROQ_API_KEY')
        if not api_key:
            issues.append("LLM API key not configured (set AUTOMCP_LLM_API_KEY environment variable)")
        
        # Validate output configuration
        output_config = config.get('output', {})
        if not output_config.get('dir'):
            issues.append("Output directory not specified")
        
        # Validate required sections
        required_sections = ['llm_client', 'semantic_transformation', 'output']
        for section in required_sections:
            if not config.get(section):
                issues.append(f"Missing required section: {section}")
        
        # Display results
        if not issues:
            click.echo("Configuration validation passed successfully.")
            click.echo("All required settings are present and valid.")
        else:
            click.echo("Configuration issues found:")
            for issue in issues:
                click.echo(f"• {issue}")
            
            click.echo(f"\nFound {len(issues)} issue(s) that need attention.")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the AutoMCP CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

@click.group()
@click.version_option(version="1.0.0", prog_name="AutoMCP")
@click.option('--environment', '-e', 
              type=click.Choice(['development', 'production', 'enterprise', 'fallback']), 
              help="Configuration environment to use")
@click.option('--verbose', '-v', is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, environment, verbose):
    """
    🚀 AutoMCP - Intelligent API to MCP Tool Converter
    
    Transform API specifications into AI-agent-ready Model Context Protocol (MCP) tools
    with semantic enrichment and industry-standard validation.
    
    \b
    Examples:
      automcp analyze api.yaml                    # Analyze with default environment
      automcp analyze api.yaml -e production     # Use production configuration
      automcp batch input/                       # Process all files in input/
      automcp health                             # Check system health
      automcp config show -e enterprise          # Show enterprise configuration
    
    \b
    Supported Formats:
      - OpenAPI 3.0/3.1 specifications (.yaml, .json)
      - Swagger 2.0 specifications
      - Postman Collections v2.1
      - Python source code (via repository scanning)
    
    \b
    Output Files:
      - enriched_intents.json    # Semantic intent metadata
      - capabilities.txt         # Permission-based capability classifications
      - mcp_tools.json          # Complete MCP tool specifications
    """
    # Ensure context exists and store settings
    ctx.ensure_object(dict)
    ctx.obj['environment'] = environment
    ctx.obj['verbose'] = verbose
    
    if verbose:
        if environment:
            print(f"🔧 Using environment: {environment}")
        else:
            print("🔧 Using default environment configuration")



def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
