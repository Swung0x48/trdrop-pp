# Builds the release APK (signed with the debug key) with Gradle.
#   powershell -File android\build-apk.ps1
# A short temp folder is used: Java can't open its local sockets in a temp path with 8.3 short names
# (e.g. C:\Users\LONGNA~1\AppData\Local\Temp), Gradle then fails with "Unable to establish loopback connection".
$ErrorActionPreference = "Stop"
$android = Split-Path -Parent $MyInvocation.MyCommand.Path
$temp = Join-Path (Split-Path -Parent $android) ".deps\tmp"
New-Item -ItemType Directory -Force $temp | Out-Null
$env:TEMP = $temp
$env:TMP = $temp
$env:JAVA_OPTS = "-Djava.io.tmpdir=$temp -Djava.net.preferIPv4Stack=true"
$env:GRADLE_OPTS = $env:JAVA_OPTS
Push-Location $android
try {
    cmd /c ".\gradlew.bat assembleRelease --console=plain 2>&1" > "$temp\gradle-build.log"
    $code = $LASTEXITCODE
    # Windows now and then still holds classes.dex of the previous build ("used by another process")
    if ($code -ne 0 -and (Select-String -Path "$temp\gradle-build.log" -Pattern "being used by another process" -Quiet)) {
        Start-Sleep -Seconds 3
        cmd /c ".\gradlew.bat assembleRelease --console=plain 2>&1" > "$temp\gradle-build.log"
        $code = $LASTEXITCODE
    }
} finally {
    Pop-Location
}
Get-Content "$temp\gradle-build.log" | Where-Object { $_ -match "error|FAILED|BUILD|e: |What went wrong|Could not|warning:" } | Select-Object -First 40
if ($code -ne 0) { exit $code }
Write-Output "APK: $android\app\build\outputs\apk\release\app-release.apk"
