
$path = "literature/references_new.bib"
$lines = Get-Content $path
$keys = @{}
$newLines = @()
$skip = $false

foreach ($line in $lines) {
    if ($line -match "^@.*\{([^,]+),") {
        $key = $matches[1].Trim()
        if ($keys.ContainsKey($key)) {
            $skip = $true
        }
        else {
            $keys[$key] = $true
            $skip = $false
        }
    }
    
    if (-not $skip) {
        $newLines += $line
    }
    elseif ($line.Trim() -eq "}") {
        # End of skipped entry
        $skip = $false
    }
}

$finalContent = $newLines -join "`n"
Set-Content $path -Value $finalContent -Encoding UTF8
Write-Host "Removed duplicates."
