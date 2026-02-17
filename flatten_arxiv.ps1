
$sourceDir = "manuscript/tex"
$outputDir = "arxiv_submission_flat"
$mainTex = "$sourceDir/main_v2.tex"
$bblFile = "$sourceDir/main_v2.bbl"

if (Test-Path $outputDir) { Remove-Item $outputDir -Recurse -Force }
New-Item -ItemType Directory -Path $outputDir | Out-Null
# New-Item -ItemType Directory -Path "$outputDir/figures" | Out-Null

# Function to inline \input
function Flatten-Tex($filePath) {
    Write-Host "Processing $filePath"
    $content = Get-Content $filePath
    $newContent = @()
    foreach ($line in $content) {
        if ($line -match '\\input\{(.*)\}') {
            $inputPath = "$sourceDir/$($matches[1]).tex"
            if (-not (Test-Path $inputPath)) { $inputPath = "$sourceDir/$($matches[1])" }
            if (Test-Path $inputPath) {
                # $newContent += "% Inlined: $inputPath"
                $newContent += Get-Content $inputPath
            }
            else {
                Write-Host "Warning: Input file not found: $inputPath"
                $newContent += $line
            }
        }
        else {
            $newContent += $line
        }
    }
    return $newContent
}

# 1. Flatten main TeX
$flatContent = Flatten-Tex $mainTex

# 1.5. Copy BBL -> ms.bbl (Moved before processing to allow inlining)
if (Test-Path $bblFile) {
    Copy-Item $bblFile -Destination "$outputDir/ms.bbl"
    Write-Host "Copied bbl file to $outputDir/ms.bbl"
}
else {
    Write-Host "Error: .bbl file not found at $bblFile"
}

# 2. Fix paths and copy figures
$finalContent = @()
foreach ($line in $flatContent) {
    if ($line -match '\\includegraphics(\[(.*)\])?\{(.*)\}') {
        $opts = $matches[2]
        $figPath = $matches[3]
        $figName = Split-Path $figPath -Leaf
        
        # Resolve path relative to manuscript/tex
        $fullFigPath = Join-Path $sourceDir $figPath
        # Resolve to absolute path to handle .. correctly
        $fullFigPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($fullFigPath)
        
        if (Test-Path $fullFigPath) {
            Copy-Item $fullFigPath -Destination "$outputDir/$figName"
            if ($opts) {
                $finalContent += "\includegraphics[$opts]{$figName}"
            }
            else {
                $finalContent += "\includegraphics{$figName}"
            }
        }
        else {
            Write-Host "Warning: Figure not found: $fullFigPath"
            $finalContent += $line
        }
    }
    if ($line -match '\\bibliography\{.*\}') {
        # Inline the .bbl file content
        if (Test-Path "$outputDir/ms.bbl") {
            $finalContent += "% Inlining ms.bbl"
            $finalContent += Get-Content "$outputDir/ms.bbl"
        }
        else {
            $finalContent += "% Warning: ms.bbl not found, bibliography skipped"
        }
    }
    else {
        $finalContent += $line
    }
}

# 3. Write flattened TeX
$finalContent | Set-Content "$outputDir/ms.tex" -Encoding UTF8

# 5. Zip
Compress-Archive -Path "$outputDir/*" -DestinationPath "arxiv_submission_flat.zip" -Force

Write-Host "Done. Created arxiv_submission_flat.zip"
