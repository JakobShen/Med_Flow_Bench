import fs from 'node:fs';
import path from 'node:path';

export function appendStepLog(logFile: string | undefined, data: Record<string, unknown>): void {
  if (!logFile) {
    return;
  }
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  fs.appendFileSync(logFile, `${JSON.stringify({ ts: new Date().toISOString(), ...data })}\n`, 'utf-8');
}
