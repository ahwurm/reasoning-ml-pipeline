# Git Push Security Check

Perform a comprehensive security audit before pushing code to GitHub. This is critical to prevent exposing API keys, credentials, or other sensitive information.

## Security Checks to Perform:

### 1. API Key Scan
Search for exposed API keys and credentials:
- Look for patterns like `sk-[a-zA-Z0-9]{40,}`, `api_key=`, `API_KEY=`
- Check for hardcoded secrets, tokens, passwords
- Scan both staged and unstaged files
- Check Python cache files (`__pycache__`)

### 2. Environment File Check
- Verify `.env` is NOT staged for commit
- Ensure `.env` is in `.gitignore`
- Check for any other environment files (`.env.*`)

### 3. Git History Review
- Check if sensitive files were ever committed: `git log --all --full-history -- .env`
- Look for previously exposed secrets in history

### 4. File Content Review
Show me:
- All modified files: `git diff --cached`
- All new untracked files: `git status`
- Content of any configuration files being committed

### 5. Python Cache Cleanup
- Remove all `__pycache__` directories
- Clear any `.pyc` files that might contain sensitive data

## Final Steps:

If all security checks pass:
1. Stage the appropriate files
2. Create a commit with a clear message
3. Push to the repository

If any issues are found:
1. Stop the push process
2. Fix the security issues
3. Re-run this security check

## Red Flags - DO NOT PUSH if you find:
- Any hardcoded API keys or secrets
- Environment files staged for commit
- Sensitive information in configuration files
- Previously committed secrets in git history

Remember: Once pushed to GitHub, assume any exposed secrets are compromised and must be revoked immediately.