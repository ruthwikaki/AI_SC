#!/usr/bin/env python3
"""Admin CLI for managing LLM configuration"""

import sys
import json
import getpass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from app.llm.config_manager import llm_config

def main():
    print("=== LLM Configuration Manager ===\n")
    
    # Show current config
    print(f"Current Model: {llm_config.get_model_display_name()}")
    print(f"Model Name: {llm_config.get_model_name()}")
    print(f"Temperature: {llm_config.get_temperature()}")
    print(f"Max Tokens: {llm_config.get_max_tokens()}")
    print(f"Timeout: {llm_config.get_timeout()}s\n")
    
    while True:
        print("\nOptions:")
        print("1. Change Model")
        print("2. Update Parameters")
        print("3. Show Full Config")
        print("4. Validate Config")
        print("5. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            # Show available models
            available = llm_config.get_full_config()["available_models"]
            print("\nAvailable Models:")
            for i, (name, info) in enumerate(available.items(), 1):
                print(f"{i}. {info['display_name']} ({name})")
                print(f"   Best for: {info['optimal_use']}")
            
            model_choice = input("\nSelect model number: ").strip()
            try:
                model_index = int(model_choice) - 1
                model_name = list(available.keys())[model_index]
                
                password = getpass.getpass("Admin password: ")
                
                if llm_config.update_model(model_name, password):
                    print(f"✅ Model updated to: {model_name}")
                else:
                    print("❌ Failed to update model")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            print("\nCurrent Parameters:")
            params = llm_config.get_parameters()
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            print("\nEnter parameter to update (or 'done' to finish):")
            updates = {}
            
            while True:
                param = input("Parameter name: ").strip()
                if param.lower() == 'done':
                    break
                    
                if param in params:
                    value = input(f"New value for {param}: ").strip()
                    try:
                        # Try to parse as number
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except:
                        pass  # Keep as string
                        
                    updates[param] = value
            
            if updates:
                password = getpass.getpass("Admin password: ")
                try:
                    if llm_config.update_parameters(updates, password):
                        print("✅ Parameters updated")
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == "3":
            print("\nFull Configuration:")
            print(json.dumps(llm_config.get_full_config(), indent=2))
        
        elif choice == "4":
            if llm_config.validate_config():
                print("✅ Configuration is valid")
            else:
                print("❌ Configuration has errors")
        
        elif choice == "5":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
