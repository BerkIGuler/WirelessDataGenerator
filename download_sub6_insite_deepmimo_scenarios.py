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

# filter out scenarios used for lwm training?
# also only keep city scenarios

filtered_results = [result for result in search_results if "city" in result and not result.endswith("lwm")]

pp.pprint("Number of filtered results: " + str(len(filtered_results)))
pp.pprint(filtered_results)

# Download scenarios
downloaded_scenarios = []
for scenario in tqdm(filtered_results, desc="Downloading scenarios"):
    try:
        if not os.path.exists(default_download_dir / scenario):
            dm.download(scenario)
            downloaded_scenarios.append(scenario)
            print(f"Downloaded: {scenario}")
        else:
            print(f"Already exists: {scenario}")
            downloaded_scenarios.append(scenario)
    except Exception as e:
        print(f"Failed to download {scenario}: {e}")

print(f"\nSuccessfully downloaded/found {len(downloaded_scenarios)} scenarios")
