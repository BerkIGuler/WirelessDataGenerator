#!/usr/bin/env python3
"""
Script to extract unique city scenario names from processed_channels/pretrain and test directories
"""

import os
import re
from pathlib import Path

def extract_city_names_from_directory(directory_path):
    """Extract unique city names from filenames in a specific directory"""
    
    if not directory_path.exists():
        print(f"Directory {directory_path} does not exist!")
        return []
    
    # Get all .npz files
    npz_files = list(directory_path.glob("*.npz"))
    
    city_names = set()
    
    for file_path in npz_files:
        filename = file_path.name
        
        # Pattern for city files: city_XX_cityname_3p5_bsXXX.npz
        city_match = re.search(r'city_\d+_([^_]+(?:_[^_]+)*)_3p5_bs\d+', filename)
        
        if city_match:
            city_name = city_match.group(1)
            # Replace underscores with spaces for better readability
            city_name = city_name.replace('_', ' ')
            city_names.add(city_name)
        elif filename.startswith('boston5g'):
            city_names.add('Boston')
        elif filename.startswith('asu_campus'):
            city_names.add('ASU Campus')
    
    # Convert to sorted list
    unique_cities = sorted(list(city_names))
    
    return unique_cities

def fix_city_spelling_and_capitalization(city_name):
    """Fix common spelling mistakes and apply proper capitalization"""
    
    # Dictionary of corrections
    corrections = {
        'instanbul': 'Istanbul',
        'bruxelles': 'Brussels',
        'sankt-peterburg': 'Saint Petersburg',
        'la habana': 'La Habana',
        'north jakarta': 'North Jakarta',
        'sumida city': 'Sumida City',
        'taito city': 'Taito City',
        'rio de janeiro': 'Rio de Janeiro',
        'new delhi': 'New Delhi',
        'san francisco': 'San Francisco',
        'san nicolas': 'San Nicolas',
        'hong kong': 'Hong Kong',
        'cape town': 'Cape Town',
        'sandiego': 'San Diego',
        'gurbchen': 'Gurbchen'
    }
    
    # Check if we have a correction for this city
    if city_name.lower() in corrections:
        return corrections[city_name.lower()]
    
    # Apply proper capitalization for other cities
    words = city_name.split()
    capitalized_words = []
    
    for word in words:
        if word.lower() in ['of', 'de', 'la', 'el', 'van', 'von', 'der', 'den']:
            capitalized_words.append(word.lower())
        else:
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

def process_directory(directory_name, directory_path):
    """Process a specific directory and display results"""
    print(f"\n{'-' * 20} {directory_name.upper()} DIRECTORY {'-' * 20}")
    
    unique_cities = extract_city_names_from_directory(directory_path)
    
    if unique_cities:
        # Fix spelling and capitalization
        corrected_cities = [fix_city_spelling_and_capitalization(city) for city in unique_cities]
        
        print(f"Found {len(corrected_cities)} unique cities:")
        print()
        
        # Print numbered list
        for i, city in enumerate(corrected_cities, 1):
            print(f"{i:2d}. {city}")
        
        print()
        print("Total unique cities:", len(corrected_cities))
        
        print()
        print("Comma-separated list:")
        print(", ".join(corrected_cities))
        
        return corrected_cities
    else:
        print("No city names found!")
        return []

def main():
    print("Extracting unique city names from processed_channels directories...")
    
    # Process pretrain directory
    pretrain_dir = Path("processed_channels/pretrain")
    pretrain_cities = process_directory("pretrain", pretrain_dir)
    
    # Process test directory
    test_dir = Path("processed_channels/test")
    test_cities = process_directory("test", test_dir)
    
    # Show summary
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"Pretrain directory: {len(pretrain_cities)} unique cities")
    print(f"Test directory: {len(test_cities)} unique cities")
    print(f"Total across both directories: {len(set(pretrain_cities + test_cities))} unique cities")

if __name__ == "__main__":
    main() 