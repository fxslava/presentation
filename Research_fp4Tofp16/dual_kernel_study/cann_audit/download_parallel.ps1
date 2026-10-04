$ErrorActionPreference = 'Stop'
$directory = '\\wsl.localhost\Ubuntu-22.04\home\ladislav\cann-build'
$packages = @(
  @{ Name='Ascend-cann-toolkit_9.2.0-beta.2_linux-x86_64.run'; Size=1398240001L; Parts=4 },
  @{ Name='Ascend-cann-950-ops_9.2.0-beta.2_linux-x86_64.run'; Size=2850241840L; Parts=8 }
)
$workers = @()
foreach ($package in $packages) {
  $chunkSize = [long][Math]::Ceiling($package.Size / $package.Parts)
  for ($part=0; $part -lt $package.Parts; $part++) {
    $start = [long]($part * $chunkSize)
    $end = [Math]::Min($package.Size-1,$start+$chunkSize-1)
    $path = Join-Path $directory ($package.Name + '.part' + $part)
    $url = 'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.T3/' + $package.Name
    $arguments = @('--fail','--silent','--show-error','--retry','3','--connect-timeout','20','--range',"$start-$end",'-H','"Referer: https://www.hiascend.com/"',$url,'-o',$path)
    $process = Start-Process -FilePath 'C:\Windows\System32\curl.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru
    $workers += @{ Process=$process; Path=$path; Bytes=($end-$start+1) }
  }
}
foreach ($worker in $workers) {
  $worker.Process.WaitForExit()
  if ($worker.Process.ExitCode -ne 0) { throw "Download failed: $($worker.Path)" }
  if ((Get-Item -LiteralPath $worker.Path).Length -ne $worker.Bytes) { throw "Range size mismatch: $($worker.Path)" }
}
foreach ($package in $packages) {
  $destination = Join-Path $directory $package.Name
  $stream = [IO.File]::Create($destination)
  try {
    for ($part=0; $part -lt $package.Parts; $part++) {
      $inputStream = [IO.File]::OpenRead($destination + '.part' + $part)
      try { $inputStream.CopyTo($stream) } finally { $inputStream.Dispose() }
    }
  } finally { $stream.Dispose() }
  Write-Output "$($package.Name): $((Get-Item -LiteralPath $destination).Length) bytes"
}
