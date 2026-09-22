$ErrorActionPreference = "Stop"
python make_datasets.py --table table1
python run_table1.py
python summarize.py
python make_figures.py
