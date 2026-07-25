#!/usr/bin/env node
import 'dotenv/config';

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

import { agentFromComputer, checkComputerEnvironment } from '@midscene/computer';

import { buildQuPathContext } from './context.js';
import { appendStepLog } from './logger.js';
import { focusQuPath } from './platform.js';
import type { CommandName, Payload } from './types.js';

function parseArgs(argv: string[]): { command: CommandName; payloadFile?: string } {
  const [commandRaw, ...rest] = argv;
  const command = (commandRaw || 'env-check') as CommandName;
  let payloadFile: string | undefined;
  for (let index = 0; index < rest.length; index += 1) {
    if (rest[index] === '--payload-file') {
      payloadFile = rest[index + 1];
    }
  }
  return { command, payloadFile };
}

function readPayload(payloadFile?: string): Payload {
  if (!payloadFile) {
    return {};
  }
  const raw = fs.readFileSync(payloadFile, 'utf-8');
  return JSON.parse(raw) as Payload;
}

function isEmptyModelContentError(error: unknown): boolean {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  return message.includes('empty content from AI model');
}

function countFinishedActionTasks(agent: any): number {
  const executions = agent?.dump?.executions;
  if (!Array.isArray(executions) || executions.length === 0) {
    return 0;
  }

  const latest = executions[executions.length - 1];
  const tasks = latest?.tasks;
  if (!Array.isArray(tasks)) {
    return 0;
  }

  return tasks.filter((task: any) => task?.type === 'Action Space' && task?.status === 'finished').length;
}

async function run(): Promise<void> {
  const { command, payloadFile } = parseArgs(process.argv.slice(2));
  const payload = readPayload(payloadFile);

  if (command === 'env-check') {
    const envCheck = await checkComputerEnvironment();
    const result = {
      ok: true,
      platform: process.platform,
      available: envCheck.available,
      error: envCheck.error,
      displays: envCheck.displays,
    };
    process.stdout.write(`${JSON.stringify(result)}\n`);
    return;
  }

  const processName = payload.qupathProcessName || 'QuPath';
  appendStepLog(payload.stepLogFile, { phase: 'start', command, payload });

  try {
    focusQuPath(processName);
  } catch (error) {
    appendStepLog(payload.stepLogFile, { phase: 'focus_error', command, error: String(error) });
  }

  const reportFileName = payload.reportFileName || path.resolve(`./midscene_${Date.now()}.html`);
  const agentOptions: any = {
    displayId: payload.displayId,
    aiActContext: buildQuPathContext(payload),
    generateReport: true,
    persistExecutionDump: true,
    autoPrintReportMsg: false,
    reportFileName,
    replanningCycleLimit: payload.replanningCycleLimit ?? 10,
    waitAfterAction: payload.waitAfterActionMs ?? 500,
  };
  const agent = await agentFromComputer(agentOptions);

  let result: unknown;
  if (command === 'act') {
    try {
      result = await agent.aiAct(payload.instruction || '');
    } catch (error) {
      const finishedActionTasks = countFinishedActionTasks(agent);
      if (isEmptyModelContentError(error) && finishedActionTasks > 0) {
        result = {
          partial: true,
          finished_action_tasks: finishedActionTasks,
          warning:
            'Midscene executed one or more GUI actions, then the planning model returned empty content while finishing the task. Verify the updated UI state from the fresh screenshot.',
          error: error instanceof Error ? error.stack || error.message : String(error),
        };
      } else {
        throw error;
      }
    }
  } else if (command === 'query') {
    result = await agent.aiQuery(payload.question || '');
  } else if (command === 'assert') {
    await agent.aiAssert(payload.assertion || '');
    result = { asserted: true };
  } else {
    throw new Error(`Unsupported command: ${command}`);
  }

  const response = {
    ok: true,
    command,
    result,
    report_file: reportFileName,
  };
  appendStepLog(payload.stepLogFile, { phase: 'done', command, response });
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

run().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  process.stdout.write(`${JSON.stringify({ ok: false, error: message })}\n`);
  process.exitCode = 1;
});
