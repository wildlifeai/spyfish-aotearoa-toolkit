# Spyfish Aotearoa PowerBI Dashboard

This documentation guides you to set up the spyfish Aotearoa Dashboard in powerBI.

## Requirements
1. Access to relevant Amazon AWS bucket,
2. [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
3. [PowerBI Desktop](https://www.microsoft.com/en-us/power-platform/products/power-bi/downloads)

## Set up
### Connecting PowerBI to csv files in AWS
#### Install Power BI Desktop
#### Install Anaconda
#### Create an Anaconda environment for the project
Create and set up an environment for Power BI using Anaconda.
Create a new environment:
Open Anaconda Prompt and run:
```Bash
# TODO check with a higher python version
conda create --name powerbi_env python=3.9
```

Activate the new environment:
```Bash
conda activate powerbi_env
```

Install required packages:
```Bash
# TODO is matplotlib even used?
pip install requests pandas boto3 python-dotenv matplotlib
```


#### Safely save the AWS credentials in the Anaconda environment
Copy the `.env_sample_powerbi` file, rename it to `.env` and replace the values there.

Or create an `.env` file with your AWS credentials with the following info:
```
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
S3_BUCKET=<bucket-name>
```
Change the whole string after the equal sign, including the brackets.

Save the file as `.env` in the same location as the anaconda environment (e.g. `C:\Users\<YourUsername>\Anaconda3\envs\powerbi_env\.env`).

#### Get the latest PowerBI dashboard
Download the most up-to-date "DOC Spyfish report vXX.pbit" from this repository and run it in your computer.


#### Link PowerBI to the Python's version of the Anaconda environment
Configure PowerBI to use the right Python environment. In Power BI Desktop:

Go to File > Options and Settings> Options > Python scripting

<img src="img/screenshot_python_script_options_powerbi.png?raw=true" width="500" alt="python_scripting_options"/>

Set the Python home directory to your anaconda environment's path (e.g. `C:\Users\<YourUsername>\Anaconda3\envs\powerbi_env`)

### Read the csv files from AWS as datasets in PowerBI
In PowerBI Desktop, use the "Get Data" option and Select "More".

Select "Python script" and connect to it.

Copy and paste the script in the "powerbi_script.py" file within this directory.

IMPORTANT: Before running the script, change the first line of code to specify the path to your .env file


Click OK.

#### Dynamic Dataset Loading

The PowerBI integration now **dynamically loads all CSV files** from the following S3 bucket prefix:
- `spyfish_metadata/status/` - Contains status tracking files

**Dataset Naming Convention:**
All datasets are automatically named using the pattern: `{parent_directory}_{filename}`

For example:
- `spyfish_metadata/status/processing_status.csv` → `status_processing_status`
- `spyfish_metadata/status/deployment_status.csv` → `status_deployment_status`

You should see all available datasets in the PowerBI Navigator. Like in the screenshot below


<img src="img/navigator_display_datasets_lodaded.png?raw=true" width="500" alt="loaded_datasets"/>

Check the format is correct by previewing the data and "Load" them into PowerBI.

<img src="img/navigator_display_datasets_preview.png?raw=true" width="500" alt="loaded_datasets"/>


## Troubleshoot
### ADO.NET: error
If the data from AWS doesn't load and you get this error "
ADO.NET: Python script error. File "<string>", line 1 404: Not Found ^ SyntaxError: illegal target for annotation
"
Remove the data sources from Powerbi (e.g. the sites_df, species_df..) and run the script again.


## Tracking the latest version
To ensure the pbit file in this repository is the most up to date version. Save the powerbi file as a pbit and rename it to track the version number.
Upload it to this github repo, create a branch and add a commit message summarising the changes.


## Citation

If you use this code, please cite:
Anton V., Bouzaid C., Fonda K., Beran H., Marinovich J., Ladds M. (2025). Spyfish Aotearoa Toolkit. https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/powerbi_setup
