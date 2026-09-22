$ErrorActionPreference = "Stop"
python make_datasets.py --table all --seeds 400
python run_table1.py --conditions trig_interaction --seeds 400 --epochs 20 --warmup 10 --r-train 4 --r-eval 8 --r-final 32
python run_table2_dss.py --seeds 400 --epochs 20 --warmup 10 --r-train 4 --r-eval 8 --r-final 32
Write-Host "DSS smoke finished. Run setup_external_repos.py before SS smoke."
