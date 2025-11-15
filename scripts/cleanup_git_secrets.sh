#!/bin/bash
# Automated script to remove .env and secrets from Git history
# USAGE: bash scripts/cleanup_git_secrets.sh

DIR_REPO="$(pwd)"
SECRETS_FILES=".env .env.* *.key *.secret"

# Remove from history
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch $SECRETS_FILES" --prune-empty --tag-name-filter cat -- --all

echo 'IMPORTANT: Force push required to update remote!'

git push origin --force --all

git push origin --force --tags

echo 'DONE. Remember to rotate all previously committed API keys. Review your provider platforms (Binance, Alpha Vantage, etc.) to rotate any exposed secrets.'
