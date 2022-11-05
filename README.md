# PhaKinPro

PHArmacoKINetic PROperty calculator: Used to predict varies pharmacokinetic properties using QSAR models. If you use please cite \[INSERT CITATION HERE]. There is a [webserver](http://34.170.18.221/) that runs these models, but for large numbers of compounds, running locally using this code is much more effective

# Requierments
Install the requirments from the requirements.txt file. Additionally, if you want to run the webserver, you need to install flask and qunicorn

# Command line use
After downloading, `PhaKinPro/phakinpro.py` can be called from the command line with `python pharkinpro --help`

`--infile` is required and is the fileloc for a csv of SMILES to predict properties for. Requires that csv has header and is comma seperated
`--smiles_col` is the name of the column containing the SMILES strings of interest. Defaults to "SMILES"
`--outfile` is the fileloc of where the output csv file should go. Defaults to `current-working-directory/phakin_output.csv`
`--ad` flag to turn on applicability domain calculation for the models

# Webserver interface
This repository also contains the code to run a local webserver (or host your own). You can start the server by running `qunicorn wsqi:app` (or using the devolpment flask server by setting the `FLASK_APP` variable: `$env:FLASK_APP = "main"` on windows or `export FLASK_ENV=main` on unix). From that access 127.0.0.1:5000 to view the local server
