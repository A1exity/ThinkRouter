$ErrorActionPreference = "Stop"

python -m thinkrouter.experiments.run_official_pipeline --stage prepare-data
python -m thinkrouter.experiments.run_official_pipeline --stage grids
python -m thinkrouter.experiments.run_official_pipeline --stage routers
python -m thinkrouter.experiments.run_official_pipeline --stage report
