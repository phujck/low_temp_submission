$sourceDir = "manuscript/tex"
$outputDir = "arxiv_submission_flat"
$mainTex = "$sourceDir/main_v2.tex"
$bblFile = "$sourceDir/main_v2.bbl"

# 1. Clean and Create Output Directory
if (Test-Path $outputDir) { Remove-Item $outputDir -Recurse -Force }
New-Item -ItemType Directory -Path $outputDir | Out-Null
Write-Host "Created output directory: $outputDir"

# 2. Copy BBL file first (crucial for valid inlining)
if (Test-Path $bblFile) {
    Copy-Item $bblFile -Destination "$outputDir/ms.bbl"
    Write-Host "Copied .bbl file."
}
else {
    Write-Host "WARNING: .bbl file not found at $bblFile"
}

# 3. Define Recursive Processing Function
function Process-TexContent {
    param ($filePath)
    
    $processedLines = @()
    if (-not (Test-Path $filePath)) {
        Write-Host "ERROR: File not found for processing: $filePath"
        return $processedLines
    }

    $content = Get-Content $filePath
    
    foreach ($line in $content) {
        # A. Handle \input{...}
        if ($line -match '^\s*\\input\{(.*)\}') {
            $subPathRel = $matches[1]
            $subPath = "$sourceDir/$subPathRel"
            if (-not ($subPath -match '\.tex$')) { $subPath += ".tex" }
            
            # Resolve absolute path for safety
            $subPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($subPath)

            if (Test-Path $subPath) {
                Write-Host "Inlining input: $subPathRel"
                # Recursively process the input file
                $processedLines += Process-TexContent -filePath $subPath
            }
            else {
                Write-Host "WARNING: Input file not found: $subPath"
                $processedLines += $line
            }
        }
        # B. Handle \includegraphics[...]{...}
        elseif ($line -match '\\includegraphics(\[.*\])?\{(.*)\}') {
            $opts = $matches[1] # Includes the []
            $figPathRel = $matches[2]
            $figName = Split-Path $figPathRel -Leaf
            
            # Construct absolute path to the figure source
            $figSrcPath = Join-Path $sourceDir $figPathRel
            $figSrcPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($figSrcPath)

            if (Test-Path $figSrcPath) {
                # Copy figure to root of output dir
                Copy-Item $figSrcPath -Destination "$outputDir/$figName"
                # Write the new line with flattened path
                $processedLines += "\includegraphics$opts{$figName}"
            }
            else {
                Write-Host "WARNING: Figure not found: $figSrcPath"
                $processedLines += $line
            }
        }
        # C. Handle \bibliography{...}
        elseif ($line -match '\\bibliography\{.*\}') {
            if (Test-Path "$outputDir/ms.bbl") {
                Write-Host "Inlining bibliography from ms.bbl"
                $processedLines += "% Inlining ms.bbl"
                $processedLines += Get-Content "$outputDir/ms.bbl"
            }
            else {
                Write-Host "WARNING: Cannot inline bibliography, ms.bbl missing."
                $processedLines += $line
            }
        }
        # D. Regular Line
        else {
            $processedLines += $line
        }
    }
    return $processedLines
}

# 4. Run Processing
Write-Host "Processing main tex file..."
$finalContent = Process-TexContent -filePath $mainTex

# 5. Write Result
$finalContent | Set-Content "$outputDir/ms.tex" -Encoding UTF8
Write-Host "Written flattened content to $outputDir/ms.tex"

# 6. Zip It
Compress-Archive -Path "$outputDir/*" -DestinationPath "arxiv_submission_flat.zip" -Force
Write-Host "Created archive: arxiv_submission_flat.zip"
