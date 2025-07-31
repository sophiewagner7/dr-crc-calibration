# CRC Screening in the DR
Sophie Wagner, sw3767@cumc.columbia.edu <br>
Columbia University Irving Medical Center

Manuscript:  _Cost-Effectiveness of Colorectal Cancer Screening in the Dominican Republic_ 

This repo contains code to calibrate the natural history progression of colorectal cancer in the Dominican Republic. An outline of the calibration process is as follows:

1. Calibration to U.S. SEER incidence for initial transition probabilities (age-based) and model validation with CISNET CRC models.
2. Calibration to U.S. SEER incidence * DR incidence ratio (age-based) to acheive lower incidence overall and peak incidence at lower age than U.S. Ratios derived from GLOBOCAN estimates.
3. Calibration to U.S. SEER incidence * DR incidence ratio, and DR-HGPS stage distribution (less optimal than U.S. stage distribution, with higher proportion of distant cases).


### Screening model
The screening model is a decision-analytic model that compares the current status quo of no screening in the DR various CRC screening strategies. The model utilizes the estimated transition probabilities from the natural history calibration. Various CRC screening strategies are compared for cost-effectiveness: natural history or no screening, colonoscopy every 10 years, sigmoidoscopy every 5 years, biennial FIT, and biennial FOBT. This portion of the model was carried out in TreeAge Pro Health 2025.

### Directory
`data/` : input data files, including SEER data and calibration targets. Note that SEER data files are not included in the repository, as they require registration for access. Dictionary and metadata files are included. <br>
`notebooks/` : Jupyter notebooks for data cleaning and natural history calibration  <br>
`out/` : output files, including calibration results and plots  <br>
`reference/` : reference code and literature  <br> 
`src/` : source code for calibration, including model definitions and calibration scripts (goodness of fit, model configuration, plotting, and helper functions)  <br>
`.gitignore` : specifies files and directories to be ignored by Git, such as Python cache files and SEER data files  <br>
`README.md` : this file, providing an overview of the project and its structure  <br>
