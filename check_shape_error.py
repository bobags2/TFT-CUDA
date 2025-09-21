#!/usr/bin/env python3
"""
Test script to identify any 'shape' reference issues in the training code.
"""

import ast
import traceback

def check_python_code_in_bash_script():
    """Extract and validate Python code from the bash script."""
    
    # Read the bash script
    with open('scripts/train_stable.sh', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find Python code sections
    python_sections = []
    lines = content.split('\n')
    
    in_python = False
    current_python_code = []
    start_line = 0
    
    for i, line in enumerate(lines, 1):
        if 'python -c """' in line:
            in_python = True
            start_line = i
            current_python_code = []
        elif in_python and line.strip() == '"""':
            in_python = False
            python_code = '\n'.join(current_python_code)
            python_sections.append((start_line, i, python_code))
        elif in_python:
            current_python_code.append(line)
    
    print(f"Found {len(python_sections)} Python code sections")
    
    # Check each Python section for syntax issues
    for start, end, code in python_sections:
        print(f"\n🔍 Checking Python section lines {start}-{end}")
        
        try:
            # Parse the code to check for syntax errors
            ast.parse(code)
            print(f"   ✅ Syntax OK")
            
            # Check for 'shape' references without object
            lines = code.split('\n')
            for line_num, line in enumerate(lines, 1):
                # Check for 'shape' not preceded by a dot
                if 'shape' in line and '.shape' not in line:
                    actual_line = start + line_num
                    print(f"   ⚠️  Potential issue at line {actual_line}: {line.strip()}")
                    
        except SyntaxError as e:
            print(f"   ❌ Syntax error: {e}")
            print(f"      Line {start + e.lineno}: {e.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    check_python_code_in_bash_script()