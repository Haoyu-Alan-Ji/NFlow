$ErrorActionPreference = "Stop"
python make_datasets.py --table all
python run_table1.py
python run_table2_dss.py
python run_table2_ss.py
python run_table2_r.py --device cpu
python summarize.py
python make_figures.py
