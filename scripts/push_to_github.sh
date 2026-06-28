#!/usr/bin/env bash
set -euo pipefail
git init -b main
git add .
git commit -m "EchoHawk: acoustic drone detection, classification & DOA"
# Create the empty repo first at https://github.com/new (name: echohawk), then:
git remote add origin https://github.com/shulm/echohawk.git
git push -u origin main     # <-- user authenticates here (browser / PAT / gh auth login)
