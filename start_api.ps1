$env:OPENBLAS_NUM_THREADS=1
$env:OMP_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
$env:NUMEXPR_NUM_THREADS=1
python -m upishield_core.cli.main server run --port 8002