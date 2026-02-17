
$path = "literature/references_new.bib"
$lines = Get-Content $path
# We want to keep lines 1-1358 (indices 0-1357)
# And lines 1625-end (indices 1624-end)
# The bad block is indices 1358 to 1623 (lines 1359-1624).
# Check if file has enough lines
if ($lines.Count -gt 1624) {
    $cleanLines = $lines[0..1357] + $lines[1624..($lines.Count - 1)]
    Set-Content $path -Value $cleanLines -Encoding UTF8
    Write-Host "Fixed bibliography range."
}
else {
    Write-Host "File too short, checking logic."
    # If file is shorter, maybe my previous script deleted some lines but left others?
    # I'll rely on visual inspection if this fails.
}
