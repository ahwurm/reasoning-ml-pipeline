# Security Guidelines for API Key Management

## Critical Security Notice

**NEVER commit API keys or secrets to version control!**

## API Key Management Best Practices

### 1. Environment Variables

Store your DeepSeek API key in environment variables:

```bash
# Option 1: Export in terminal session
export DEEPSEEK_API_KEY="your-api-key-here"

# Option 2: Add to .env file (recommended)
echo 'DEEPSEEK_API_KEY=your-api-key-here' >> .env
```

### 2. Using .env Files

- Create a `.env` file in the project root
- Add your API key: `DEEPSEEK_API_KEY=your-api-key-here`
- **IMPORTANT**: The `.env` file is already in `.gitignore`
- Never remove `.env` from `.gitignore`

### 3. Command Line Usage

When running scripts, you have three options:

```bash
# Option 1: Use environment variable
export DEEPSEEK_API_KEY="your-api-key"
python incremental_generate.py

# Option 2: Pass as command argument
python incremental_generate.py --api-key "your-api-key"

# Option 3: Use .env file (automatic)
# Just ensure .env file exists with DEEPSEEK_API_KEY
python incremental_generate.py
```

### 4. Security Checklist

Before committing code:

- [ ] No hardcoded API keys in Python files
- [ ] No API keys in configuration files
- [ ] `.env` file is not tracked by git
- [ ] Clear any Python cache: `find . -type d -name "__pycache__" -exec rm -rf {} +`
- [ ] Run `git status` to verify no sensitive files are staged

### 5. If Your API Key is Exposed

1. **Immediately revoke the exposed key** in your DeepSeek dashboard
2. Generate a new API key
3. Update your local `.env` file with the new key
4. Check git history: `git log --all --full-history -- .env`
5. If the key was committed, consider the repository compromised

### 6. Additional Security Measures

- Use different API keys for development and production
- Rotate API keys regularly
- Monitor API usage for suspicious activity
- Consider using a secrets management service for production deployments

## Pre-commit Hook (Optional)

Add this pre-commit hook to prevent accidental commits of secrets:

```bash
#!/bin/sh
# Save as .git/hooks/pre-commit

# Check for common secret patterns
if git diff --cached --name-only | xargs grep -E "(sk-[a-zA-Z0-9]{40,}|api_key\s*=\s*['\"][^'\"]+['\"])" 2>/dev/null; then
    echo "Error: Possible API key detected in staged files!"
    echo "Please remove any secrets before committing."
    exit 1
fi
```

## Contact

If you discover a security vulnerability, please report it immediately to the project maintainers.