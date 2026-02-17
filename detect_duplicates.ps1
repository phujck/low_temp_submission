
$path = "literature/references_new.bib"
$lines = Get-Content $path
$keys = @{}
$duplicates = @()

foreach ($line in $lines) {
    if ($line -match "^@.*\{([^,]+),") {
        $key = $matches[1].Trim()
        if ($keys.ContainsKey($key)) {
            $duplicates += $key
            Write-Host "Duplicate found: $key"
        }
        else {
            $keys[$key] = $true
        }
    }
}

if ($duplicates.Count -eq 0) {
    Write-Host "No duplicates found."
}
else {
    Write-Host "Found $($duplicates.Count) duplicates."
}
