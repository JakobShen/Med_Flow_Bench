import { execFileSync } from 'node:child_process';

export function focusQuPath(processName: string): void {
  const platform = process.platform;
  if (platform === 'darwin') {
    execFileSync('osascript', [
      '-e', 'tell application "System Events"',
      '-e', `  tell process "${processName}" to set frontmost to true`,
      '-e', 'end tell',
    ]);
    return;
  }

  if (platform === 'win32') {
    const command = `$wshell = New-Object -ComObject WScript.Shell; [void]$wshell.AppActivate('${processName}')`;
    execFileSync('powershell', ['-NoProfile', '-Command', command]);
    return;
  }
}
