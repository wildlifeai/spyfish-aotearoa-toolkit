# Spyfish Aotearoa Toolkit
This repository contains the scripts and information required to manage the footage and metadata of Spyfish Aotearoa, a collaboration between [Wildlife.ai](https://wildlife.ai/) and [Te Papa Atawhai](https://www.doc.govt.nz/about-us/) to monitor Marine Reserves across Aotearoa using Baited Underwater Videos.

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

## Overview
Spyfish Aotearoa utilises a citizen science and machine learning approach to classify baited underwater video (BUV) footage collected from New Zealand marine reserves. This project has a modular architecture featuring custom-built applications and software.

## sftk

The Spyfish Aotearoa Toolkit (sftk) is a Python package that provides a set of tools to manage the data pipeline and management of the Spyfish Aotearoa project.

### Usage

You can install the package by running this

```
pip install "/path/to/Spyfish-Aotearoa-toolkit"
```
or with the dev tag, which adds a few additional development related libraries (the -e stands for editable, meaning that if you change something in the code it will be reflected in the functioning in the package).
```
pip install -e "/path/to/Spyfish-Aotearoa-toolkit[dev]"
```


You can also use the sftk package by adding the root directory to your `PYTHONPATH` and importing the modules you need.

```python
import sys
sys.path.append('path/to/Spyfish-Aotearoa-toolkit')
from sftk import ...
```

Alternatively you can add to `PYTHONPATH` through your Terminal:

```bash
export PYTHONPATH=$PYTHONPATH:path/to/Spyfish-Aotearoa-toolkit
```

Or add this line to `.bashrc` to make it permanent:

```bash
echo 'export PYTHONPATH=$PYTHONPATH:path/to/Spyfish-Aotearoa-toolkit' >> ~/.bashrc
```



### Environment Variables
Copy `.env_sample` to `.env` and fill in your own values:

For the HPC environment, you need to create the .env File in Your Home Directory
You can create and edit the file directly in your home directory using the terminal.

#### Navigate to your home directory

```bash
cd $HOME
```

#### Create and open the .env file for editing
```bash
nano .env
```
Inside the editor, add your environment variables

## Collaborations

We are working to make our work available to anyone interested.
Take a look into existing [issues][issues-url] (or code), and contribute any way you can.



Feel free to [contact us][contact_info], if you have any questions.



### Pre-Commit checks

We set up some pre-commit checks, so run this before you contribute:
```bash
pip install pre-commit # only the first time
pre-commit install
```

This will fix/change/flag any issues defined in our pre-commit config.
You need to re add any changes performed by the pre-commit.


Make sure your code is up to standard, but if the existing code is showing too many errors, feel free to skip it by adding `--no-verify`:
```bash
git commit --no-verify -m "message"
```

You can run single commands, for example:
```bash
pre-commit run flake8
```


### Spelling checker
Please install a spelling checker in your IDE of choice, for example [this one][spell-checker] VS code/Cursor.



## Metadata management


## Questions

Please feel free to [contact us][contact_info] or raise an issue, if you have any questions.



## Citation

If you use this code, please cite:

Anton V., Fonda K., Atkin E., Beran H., Marinovich J., Ladds M. (2024). Spyfish Aotearoa Toolkit. https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/wildlifeai/Spyfish-Aotearoa-toolkit.svg?style=for-the-badge
[contributors-url]: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildlifeai/Spyfish-Aotearoa-toolkit.svg?style=for-the-badge
[forks-url]: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/network/members
[stars-shield]: https://img.shields.io/github/stars/wildlifeai/Spyfish-Aotearoa-toolkit.svg?style=for-the-badge
[stars-url]: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/stargazers]
[issues-shield]: https://img.shields.io/github/issues/wildlifeai/Spyfish-Aotearoa-toolkit.svg?style=for-the-badge
[issues-url]: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/issues
[license-shield]: https://img.shields.io/github/license/wildlifeai/Spyfish-Aotearoa-toolkit.svg?style=for-the-badge
[license-url]: https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/blob/main/LICENSE.txt
[spell-checker]: https://marketplace.cursorapi.com/items?itemName=streetsidesoftware.code-spell-checker]
[contact_info]: contact@wildlife.ai
