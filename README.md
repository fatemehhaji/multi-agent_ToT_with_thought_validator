# Multi-Agent Tree-of-Thought Implementation

Implementation code for the paper "Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent"

## Overview

This codebase implements:
- Tree of Thoughts (ToT) base architecture
- Multiple reasoning agents with ToT 
- Verification agent for reasoning validation
- Consensus-based multi-round voting

## Setup

```bash
# Clone the repository 
git clone https://github.com/fatemehhaji/multi-agent_ToT_with_thought_validator.git
cd multi-agent_ToT_with_thought_validator

# Install dependencies
pip install -r requirements.txt
```

## Files Structure

- tot_validator/gsm8k/run_gsm8k_multiple_verifiers_base.py: Main implementation file
- tot_validator/gsm8k/run_experiment.sh: Shell script to run experiments

## Note
The code requires OpenAI API access and expects GSM8K dataset in the specified data root path.
