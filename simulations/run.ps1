param(
    [ValidateSet('safe','full')]
    [string]$Profile = 'safe',
    [int]$MaxThreads = 1
)

$env:OMP_NUM_THREADS = $MaxThreads
$env:MKL_NUM_THREADS = $MaxThreads
$env:OPENBLAS_NUM_THREADS = $MaxThreads
$env:NUMEXPR_NUM_THREADS = $MaxThreads
$env:VECLIB_MAXIMUM_THREADS = $MaxThreads
$env:BLIS_NUM_THREADS = $MaxThreads

$script = Join-Path $PSScriptRoot 'src\main.py'
$outDir = Join-Path $PSScriptRoot 'results'

Write-Host "Running simulations (profile=$Profile, maxThreads=$MaxThreads)";
py -3 $script --profile $Profile --max-threads $MaxThreads --out-dir $outDir
