###################################
####### IMPORTANT #################
###################################
# You MUST update the env_path below to point to your .env file!
# This file contains your AWS credentials and S3 bucket information.
# Example: env_path = r"C:\Users\YourUsername\Anaconda3\envs\powerbi_env\.env"
env_path = r"C:\Users\USER\anaconda3\envs\powerbi_env\.env"  # FIXME: Update this path for your environment!

# The rest of the code should remain the same

import importlib.util
import sys
import requests

# Define the URL of the script (using latest main branch)
script_url = "https://raw.githubusercontent.com/wildlifeai/Spyfish-Aotearoa-toolkit/main/powerbi_setup/get_kso_data_from_s3.py"

# Download the script content
try:
    response = requests.get(script_url)
    response.raise_for_status()
    script_content = response.text

    # Verify content was downloaded
    if not script_content:
        raise ValueError("Downloaded script is empty")

    spec = importlib.util.spec_from_loader(
        "kso_to_pbi_module", loader=None, origin=script_url
    )
    kso_to_pbi_module = importlib.util.module_from_spec(spec)
    exec(script_content, kso_to_pbi_module.__dict__)
    sys.modules["kso_to_pbi_module"] = kso_to_pbi_module

    # Run with error handling - main() now returns a dictionary of dataframes
    dataframes_dict = kso_to_pbi_module.main(env_path)
    
    # Make all dataframes available to PowerBI by adding them to global scope
    globals().update(dataframes_dict)

except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    raise
