import deepmimo as dm
import pprint as pp
from tqdm import tqdm
import os
from pathlib import Path


default_download_dir = Path("./deepmimo_scenarios")

# Query for all sub6 GHz scenarios with Insite raytracer
query = {
    "bands": ["sub6"],
    "raytracerName": "Insite",
    "environment": "outdoor",
}

search_results = dm.search(query)

pp.pprint("Number of results: " + str(len(search_results)))
pp.pprint(search_results)

# Filter scenarios to download:
# 1. All city scenarios (excluding lwm training scenarios)  
# 2. Additional specific scenarios: asu_campus_3p5, boston5g_3p5

filtered_results = []

# Add city scenarios (excluding LWM training)
city_scenarios = [result for result in search_results if "city" in result and not result.endswith("lwm")]
filtered_results.extend(city_scenarios)

# Add specific additional scenarios
additional_scenarios = ["asu_campus_3p5", "boston5g_3p5"]
for target_scenario in additional_scenarios:
    matching_scenarios = [result for result in search_results if target_scenario in result.lower()]
    filtered_results.extend(matching_scenarios)

# Remove duplicates and sort
filtered_results = sorted(list(set(filtered_results)))

print(f"\nFiltered scenarios to download:")
print(f"  - City scenarios: {len(city_scenarios)}")

additional_found = [s for s in filtered_results if not "city" in s]
print(f"  - Additional scenarios: {len(additional_found)}")
if additional_found:
    print(f"    Additional scenarios found: {additional_found}")

pp.pprint("Total scenarios to download: " + str(len(filtered_results)))

# Download scenarios
downloaded_scenarios = []
for scenario in tqdm(filtered_results, desc="Downloading scenarios"):
    try:
        if not os.path.exists(default_download_dir / scenario):
            dm.download(scenario)  # Download to default DeepMIMO location
            downloaded_scenarios.append(scenario)
            print(f"Downloaded: {scenario}")
        else:
            print(f"Already exists: {scenario}")
            downloaded_scenarios.append(scenario)
    except Exception as e:
        print(f"Failed to download {scenario}: {e}")

print(f"\nSuccessfully downloaded/found {len(downloaded_scenarios)} scenarios")
print(f"Total target scenarios: {len(filtered_results)}")
print(f"Download success rate: {100*len(downloaded_scenarios)/len(filtered_results):.1f}%")
