#!/usr/bin/env powershell
# Script to clean API keys from git history

if (Test-Path "tests/test_aws_groq.py") {
    Write-Host "Cleaning tests/test_aws_groq.py"
    (Get-Content "tests/test_aws_groq.py") -replace "REDACTED_GROQ_KEY", "your-api-key-here" | Set-Content "tests/test_aws_groq.py"
}

if (Test-Path ".env") {
    Write-Host "Cleaning .env"
    (Get-Content ".env") -replace "REDACTED_GROQ_KEY", "your-api-key-here" | Set-Content ".env"
}

# Also check for any other Python files that might contain the key
Get-ChildItem -Recurse -Include "*.py" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match "REDACTED_GROQ_KEY") {
        Write-Host "Cleaning $($_.FullName)"
        $content -replace "REDACTED_GROQ_KEY", "your-api-key-here" | Set-Content $_.FullName
    }
}
