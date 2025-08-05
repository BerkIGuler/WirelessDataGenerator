"""
DeepMIMO Scenario Query and Listing Script

This script queries the DeepMIMO database for available sub6 GHz scenarios
using the Insite raytracer and provides a detailed listing of all options.

Features:
- Query sub6 GHz outdoor scenarios with Insite raytracer
- Filter and categorize scenarios by type
- Display detailed information about each scenario
- No downloading - just listing available options
"""

import deepmimo as dm
import pprint as pp
from pathlib import Path
import re


def query_available_scenarios():
    """
    Query DeepMIMO for available sub6 GHz Insite scenarios.
    
    Returns:
        list: List of scenario names
    """
    print("=" * 60)
    print("QUERYING DEEPMIMO DATABASE")
    print("=" * 60)
    
    # Query for all sub6 GHz scenarios with Insite raytracer
    query = {
        "bands": ["sub6"],
        "raytracerName": "Insite", 
        "environment": "outdoor",
    }
    
    print("Query parameters:")
    pp.pprint(query)
    print()
    
    print("Searching DeepMIMO database...")
    search_results = dm.search(query)
    
    print(f"Found {len(search_results)} scenarios matching criteria")
    print()
    
    return search_results


def categorize_scenarios(scenarios):
    """
    Categorize scenarios by type and characteristics.
    
    Args:
        scenarios (list): List of scenario names
        
    Returns:
        dict: Categorized scenarios
    """
    categories = {
        'city_scenarios': [],
        'lwm_training': [],
        'other_scenarios': []
    }
    
    for scenario in scenarios:
        if 'city' in scenario.lower():
            if scenario.endswith('lwm'):
                categories['lwm_training'].append(scenario)
            else:
                categories['city_scenarios'].append(scenario)
        else:
            categories['other_scenarios'].append(scenario)
    
    # Sort each category
    for category in categories:
        categories[category].sort()
    
    return categories


def analyze_scenario_patterns(scenarios):
    """
    Analyze patterns in scenario names to extract useful information.
    
    Args:
        scenarios (list): List of scenario names
        
    Returns:
        dict: Analysis results
    """
    analysis = {
        'total_count': len(scenarios),
        'cities': set(),
        'frequencies': set(),
        'unique_prefixes': set(),
        'lwm_count': 0,
        'city_count': 0
    }
    
    for scenario in scenarios:
        # Extract city names
        if 'city' in scenario.lower():
            analysis['city_count'] += 1
            # Extract city name pattern
            match = re.search(r'city_(\d+)_([^_]+)', scenario)
            if match:
                city_num, city_name = match.groups()
                analysis['cities'].add(f"{city_num}_{city_name}")
        
        # Extract frequency information
        if '3p5' in scenario:
            analysis['frequencies'].add('3.5 GHz')
        elif '28' in scenario:
            analysis['frequencies'].add('28 GHz')
        
        # Count LWM scenarios
        if scenario.endswith('lwm'):
            analysis['lwm_count'] += 1
        
        # Extract unique prefixes
        prefix = scenario.split('_')[0] if '_' in scenario else scenario
        analysis['unique_prefixes'].add(prefix)
    
    return analysis


def display_categorized_scenarios(categories):
    """
    Display scenarios organized by category.
    
    Args:
        categories (dict): Categorized scenarios
    """
    print("=" * 60)
    print("SCENARIO CATEGORIES")
    print("=" * 60)
    
    # City scenarios
    if categories['city_scenarios']:
        print(f"\n🏙️  CITY SCENARIOS ({len(categories['city_scenarios'])})")
        print("-" * 50)
        for i, scenario in enumerate(categories['city_scenarios'], 1):
            # Extract city info for better display
            match = re.search(r'city_(\d+)_([^_]+)_(.+)', scenario)
            if match:
                city_num, city_name, freq = match.groups()
                city_display = city_name.replace('_', ' ').title()
                print(f"  {i:2d}. {scenario}")
                print(f"      └─ City #{city_num}: {city_display} @ {freq.replace('p', '.')} GHz")
            else:
                print(f"  {i:2d}. {scenario}")
    
    # LWM training scenarios
    if categories['lwm_training']:
        print(f"\n🤖 LWM TRAINING SCENARIOS ({len(categories['lwm_training'])})")
        print("-" * 50)
        for i, scenario in enumerate(categories['lwm_training'], 1):
            print(f"  {i:2d}. {scenario}")
    
    # Other scenarios
    if categories['other_scenarios']:
        print(f"\n📡 OTHER SCENARIOS ({len(categories['other_scenarios'])})")
        print("-" * 50)
        for i, scenario in enumerate(categories['other_scenarios'], 1):
            print(f"  {i:2d}. {scenario}")


def display_analysis(analysis):
    """
    Display scenario analysis results.
    
    Args:
        analysis (dict): Analysis results
    """
    print("=" * 60)
    print("SCENARIO ANALYSIS")
    print("=" * 60)
    
    print(f"📊 Total scenarios: {analysis['total_count']}")
    print(f"🏙️  City scenarios: {analysis['city_count']}")
    print(f"🤖 LWM training scenarios: {analysis['lwm_count']}")
    print(f"📡 Other scenarios: {analysis['total_count'] - analysis['city_count'] - analysis['lwm_count']}")
    
    print(f"\n📡 Frequencies available: {', '.join(sorted(analysis['frequencies']))}")
    print(f"🌍 Unique cities: {len(analysis['cities'])}")
    
    if analysis['cities']:
        print(f"\n🏙️  City List (showing first 10):")
        cities_list = sorted(list(analysis['cities']))
        for i, city in enumerate(cities_list[:10], 1):
            city_name = city.split('_', 1)[1].replace('_', ' ').title()
            print(f"   {i:2d}. {city_name}")
        
        if len(cities_list) > 10:
            print(f"   ... and {len(cities_list) - 10} more cities")


def check_local_availability(scenarios):
    """
    Check which scenarios are already downloaded locally.
    
    Args:
        scenarios (list): List of scenario names
        
    Returns:
        dict: Local availability information
    """
    scenarios_dir = Path("./deepmimo_scenarios")
    
    if not scenarios_dir.exists():
        return {'available_locally': [], 'not_downloaded': scenarios}
    
    available_locally = []
    not_downloaded = []
    
    for scenario in scenarios:
        scenario_path = scenarios_dir / scenario
        if scenario_path.exists() and scenario_path.is_dir():
            available_locally.append(scenario)
        else:
            not_downloaded.append(scenario)
    
    return {
        'available_locally': available_locally,
        'not_downloaded': not_downloaded
    }


def display_local_availability(local_info):
    """
    Display information about locally available scenarios.
    
    Args:
        local_info (dict): Local availability information
    """
    print("=" * 60)
    print("LOCAL AVAILABILITY")
    print("=" * 60)
    
    total_scenarios = len(local_info['available_locally']) + len(local_info['not_downloaded'])
    local_count = len(local_info['available_locally'])
    
    print(f"💾 Downloaded locally: {local_count}/{total_scenarios} ({100*local_count/total_scenarios:.1f}%)")
    print(f"☁️  Available for download: {len(local_info['not_downloaded'])}")
    
    if local_info['available_locally']:
        print(f"\n✅ Locally Available (showing first 10):")
        for i, scenario in enumerate(local_info['available_locally'][:10], 1):
            print(f"   {i:2d}. {scenario}")
        
        if len(local_info['available_locally']) > 10:
            print(f"   ... and {len(local_info['available_locally']) - 10} more")


def generate_summary_report():
    """Generate a comprehensive summary report of available scenarios."""
    
    print("🌐 DeepMIMO Sub6 GHz Insite Scenario Listing")
    print("=" * 60)
    
    # Query scenarios
    scenarios = query_available_scenarios()
    
    if not scenarios:
        print("❌ No scenarios found matching the criteria.")
        return
    
    # Categorize scenarios
    categories = categorize_scenarios(scenarios)
    
    # Analyze patterns
    analysis = analyze_scenario_patterns(scenarios)
    
    # Check local availability
    local_info = check_local_availability(scenarios)
    
    # Display results
    display_analysis(analysis)
    display_categorized_scenarios(categories)
    display_local_availability(local_info)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"📋 Total available scenarios: {len(scenarios)}")
    print(f"🏙️  City scenarios: {len(categories['city_scenarios'])}")
    print(f"🤖 LWM training scenarios: {len(categories['lwm_training'])}")
    print(f"📡 Other scenarios: {len(categories['other_scenarios'])}")
    print(f"💾 Already downloaded: {len(local_info['available_locally'])}")
    print(f"☁️  Available for download: {len(local_info['not_downloaded'])}")
    
    print(f"\n💡 To download scenarios, use: python3 download_sub6_insite_deepmimo_scenarios.py")
    print(f"💡 To process scenarios, use: python3 batch_process_deepmimo_scenarios.py")


if __name__ == "__main__":
    generate_summary_report() 