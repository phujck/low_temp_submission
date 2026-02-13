param(
    [string]$Command
)

$PythonStr = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonStr = "py"
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonStr = "python3"
}

if ($Command -eq "install") {
    & $PythonStr -m pip install -r requirements.txt
}
elseif ($Command -eq "sim") {
    & $PythonStr simulations/src/run_low_temp_suite.py --regime all --profile full --seed 42
    & $PythonStr simulations/src/plot_low_temp_suite.py
    & $PythonStr simulations/src/validate_low_temp_claims.py
}
elseif ($Command -eq "paper") {
    Push-Location manuscript/tex
    
    # Extract title from main_v2.tex
    $titleLine = Get-Content main_v2.tex | Select-String -Pattern '\\title\{(.*)\}'
    if ($titleLine) {
        $title = $titleLine.Matches.Groups[1].Value
        # Sanitize title for filename
        $cleanTitle = $title -replace '[\\/:*?"<>|]', '' -replace '\s+', ' '
    } else {
        $cleanTitle = "main_v2"
    }

    pdflatex main_v2.tex
    bibtex main_v2
    pdflatex main_v2.tex
    pdflatex main_v2.tex
    if (Test-Path "main_v2.pdf") {
        if (Test-Path "../$cleanTitle.pdf") {
            Remove-Item "../$cleanTitle.pdf" -Force
        }
        Move-Item -Path "main_v2.pdf" -Destination "../$cleanTitle.pdf" -Force
    }
    
    Pop-Location
}
elseif ($Command -eq "clean") {
    Remove-Item -Recurse -Force manuscript/build/*
    Remove-Item -Path "manuscript/tex/*.aux", "manuscript/tex/*.bbl", "manuscript/tex/*.blg", "manuscript/tex/*.log", "manuscript/tex/*.out", "manuscript/tex/*.toc" -ErrorAction SilentlyContinue
}
else {
    Write-Host "Usage: ./manage.ps1 [install|sim|paper|clean]"
}
