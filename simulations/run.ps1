param(
    [ValidateSet('quick','full','publish')]
    [string]$Profile = 'quick',
    [ValidateSet('cg','all')]
    [string]$Regime = 'all',
    [int]$MaxThreads = 1,
    [int]$Seed = 42
)

$env:OMP_NUM_THREADS = $MaxThreads
$env:MKL_NUM_THREADS = $MaxThreads
$env:OPENBLAS_NUM_THREADS = $MaxThreads
$env:NUMEXPR_NUM_THREADS = $MaxThreads
$env:VECLIB_MAXIMUM_THREADS = $MaxThreads
$env:BLIS_NUM_THREADS = $MaxThreads

$runScript = Join-Path $PSScriptRoot 'src\run_low_temp_suite.py'
$plotScript = Join-Path $PSScriptRoot 'src\plot_low_temp_suite.py'
$validateScript = Join-Path $PSScriptRoot 'src\validate_low_temp_claims.py'
$outDir = Join-Path $PSScriptRoot 'results'

Write-Host "Running low_temp suite (profile=$Profile, regime=$Regime, seed=$Seed, maxThreads=$MaxThreads)";
py -3 $runScript --profile $Profile --regime $Regime --seed $Seed --n-workers $MaxThreads --outdir $outDir
py -3 $plotScript --outdir $outDir
py -3 $validateScript --outdir $outDir
