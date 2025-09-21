#!/usr/bin/env bash

echo "Debugging device movement edge cases..."

python3 -c '''
# Test what happens with edge cases in device movement
test_cases = [
    {"case": "normal tensor", "value": "torch.randn(3, 3)"},
    {"case": "None value", "value": None},
    {"case": "string value", "value": "test_string"},
    {"case": "empty dict", "value": {}},
]

device = "cpu"  # Use CPU for testing

for test_case in test_cases:
    try:
        value = test_case["value"]
        if test_case["case"] == "normal tensor":
            import torch
            value = torch.randn(3, 3)
        
        print(f"Testing {test_case['case']}: {type(value)}")
        
        # Test the problematic code pattern
        if hasattr(value, "to"):
            result = value.to(device)
            print(f"  ✅ Successfully moved to {device}")
        else:
            print(f"  ℹ️  No 'to' method, keeping as-is")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("Edge case testing complete")
'''