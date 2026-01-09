
import re

def parse_reqs(file_path):
    reqs = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Handle package==version, package>=version, etc.
                match = re.match(r'^([^<>=>! ]+)([<>=>! ]+.*)$', line)
                if match:
                    name, version = match.groups()
                    reqs[name.lower()] = line
                else:
                    reqs[line.lower()] = line
    except FileNotFoundError:
        pass
    return reqs

# Load original requirements
orig_reqs = parse_reqs('requirements.txt')

# Load current environment list
current_reqs = parse_reqs('current_env_list.txt')

# Missing ones specifically requested (ensure they are in the final list with these versions)
missing_reqs = {
    'textattack': 'textattack==0.3.10',
    'vllm': 'vllm==0.3.3'
}

# Merge strategy:
# 1. Take everything from current environment (fulfills "include my current config")
merged = current_reqs.copy()

# 2. Add anything from original requirements that's missing in current env
# (ensures project-specific libs like 'beir' are included)
for name, line in orig_reqs.items():
    if name not in merged:
        merged[name] = line

# 3. Explicitly override/ensure the requested missing packages
for name, line in missing_reqs.items():
    merged[name] = line

# Sort and write
sorted_names = sorted(merged.keys())
with open('requirements_merged.txt', 'w') as f:
    for name in sorted_names:
        f.write(merged[name] + '\n')
