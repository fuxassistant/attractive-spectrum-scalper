$ErrorActionPreference = 'Stop'
$sourceRoot = [IO.Path]::GetFullPath('D:\scalper-mini\poi-scalper').TrimEnd('\')
$reviewRoot = 'D:\scalper-mini\attractive-spectrum-scalper\reports\poi_research_review_20261018'
$receiptPath = Join-Path $reviewRoot 'cleanup_execution.jsonl'
if (Test-Path -LiteralPath $receiptPath) { throw 'Execution receipt exists; do not rerun without reconciliation.' }
$plan = Get-Content -LiteralPath (Join-Path $reviewRoot 'proposed_payload_removals.json') -Raw | ConvertFrom-Json
$capsule = Get-Content -LiteralPath (Join-Path $reviewRoot 'capsule_receipt.json') -Raw | ConvertFrom-Json
$capsulePath = Join-Path $reviewRoot 'research_evidence.zip'
if ((Get-FileHash -LiteralPath $capsulePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $capsule.sha256) { throw 'Capsule checksum mismatch' }
$allowed = @(
 'data/archive/price_flow_probe_pre_delivery_bootstrap_20261005/',
 'data/archive/price_flow_probe_pre_gap_flush_20261005/',
 'data/archive/price_flow_probe_pre_stock_mask_20261005/',
 'data/archive/price_flow_probe_pre_occupancy_20261005/',
 'data/archive/price_flow_probe_pre_queue_reactions_20261005/',
 'data/research/data_lab_v1/four_weeks_20261011/',
 'data/research/data_lab_v1/four_weeks_20261011_r2/'
)
$checked = @()
foreach ($entry in $plan.files) {
 $rel = [string]$entry.path
 if (-not ($allowed | Where-Object { $rel.StartsWith($_, [StringComparison]::Ordinal) })) { throw "Outside approved groups: $rel" }
 if ([IO.Path]::GetExtension($rel) -notin @('.parquet','.npy','.npz','.bin','.arrow')) { throw "Protected extension: $rel" }
 $absolute = [IO.Path]::GetFullPath((Join-Path $sourceRoot $rel))
 if (-not $absolute.StartsWith($sourceRoot+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Escaping source root' }
 $item = Get-Item -LiteralPath $absolute -Force
 if ($item.PSIsContainer) { throw 'Directories are not deletion targets' }
 $cursor = $item
 while ($cursor -and $cursor.FullName.Length -ge $sourceRoot.Length) {
  if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point refused: $($cursor.FullName)" }
  $cursor = if ($cursor.PSIsContainer) { $cursor.Parent } else { $cursor.Directory }
 }
 $mtimeNs = ([decimal]$item.LastWriteTimeUtc.Ticks - [decimal]621355968000000000) * 100
 if ($item.Length -ne $entry.bytes -or $mtimeNs -ne [decimal]$entry.mtime_ns) { throw "File changed: $rel" }
 $checked += [pscustomobject]@{ path=$rel; absolute=$absolute; bytes=$item.Length; ticks=$item.LastWriteTimeUtc.Ticks }
}
# Preserve hashes of every non-target file in the affected groups.
$targetSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($x in $checked) { [void]$targetSet.Add($x.absolute) }
$retained = @()
foreach ($prefix in $allowed) {
 foreach ($item in Get-ChildItem -LiteralPath (Join-Path $sourceRoot $prefix) -File -Recurse -Force) {
  if (-not $targetSet.Contains($item.FullName)) {
   $retained += [pscustomobject]@{ path=$item.FullName; bytes=$item.Length; sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
  }
 }
}
$retained | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reviewRoot 'cleanup_retained_files.json') -Encoding utf8
$before = (Get-PSDrive D).Free
$deletedBytes = [long]0
$failures = @()
foreach ($entry in $checked) {
 try {
  $item = Get-Item -LiteralPath $entry.absolute -Force
  if ($item.Length -ne $entry.bytes -or $item.LastWriteTimeUtc.Ticks -ne $entry.ticks) { throw 'Changed after validation' }
  $hash = (Get-FileHash -LiteralPath $entry.absolute -Algorithm SHA256).Hash.ToLowerInvariant()
  $record = @{ path=$entry.path; absolute=$entry.absolute; bytes=$entry.bytes; sha256=$hash; status='INTENT'; utc=[DateTime]::UtcNow.ToString('o') }
  ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath $receiptPath -Encoding utf8
  Remove-Item -LiteralPath $entry.absolute -Force
  if (Test-Path -LiteralPath $entry.absolute) { throw 'Target still exists' }
  $record.status='DELETED'
  ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath $receiptPath -Encoding utf8
  $deletedBytes += $entry.bytes
 } catch {
  $failures += @{ path=$entry.path; error=$_.Exception.Message }
 }
}
$retainedErrors = @()
foreach ($entry in $retained) {
 if (-not (Test-Path -LiteralPath $entry.path) -or (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { $retainedErrors += $entry.path }
}
$summary = @{ status=if ($failures.Count -eq 0 -and $retainedErrors.Count -eq 0) {'COMPLETE'} else {'REVIEW_REQUIRED'}; attempted=$checked.Count; deleted_bytes=$deletedBytes; free_before=$before; free_after=(Get-PSDrive D).Free; failures=$failures; retained_files=$retained.Count; retained_errors=$retainedErrors; scope='Approved stage A exact payload list only; no recursive deletion, no archive deletion' }
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $reviewRoot 'cleanup_result.json') -Encoding utf8
$summary | ConvertTo-Json -Depth 6
