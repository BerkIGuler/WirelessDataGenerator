import deepmimo as dm
import pprint as pp
from tqdm import tqdm
import os
from pathlib import Path
import re


default_download_dir = Path("./deepmimo_scenarios")

# Query for all mmWave scenarios with Insite raytracer
query = {
    "bands": ["mmW"],
    "raytracerName": "Insite", 
    "environment": "outdoor",
}

search_results = dm.search(query)

pp.pprint("Number of results: " + str(len(search_results)))
pp.pprint(search_results)

# Filter scenarios to download:
# Only scenarios that include "city" in the name (excluding lwm training scenarios)
filtered_results = []

# Add city scenarios (excluding LWM training)
city_scenarios = [result for result in search_results if "city" in result and not result.endswith("lwm")]
filtered_results.extend(city_scenarios)

# Remove duplicates and sort
filtered_results = sorted(list(set(filtered_results)))

print(f"\nFiltered scenarios to download:")
print(f"  - City scenarios: {len(city_scenarios)}")

pp.pprint("Total scenarios to download: " + str(len(filtered_results)))

# Download scenarios
downloaded_scenarios = []
for scenario in tqdm(filtered_results, desc="Downloading mmWave city scenarios"):
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

# Print summary of downloaded scenarios
if downloaded_scenarios:
    print(f"\nDownloaded mmWave city scenarios:")
    for i, scenario in enumerate(downloaded_scenarios, 1):
        print(f"  {i:2d}. {scenario}") 