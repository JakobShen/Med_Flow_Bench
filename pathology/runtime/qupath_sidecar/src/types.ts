export interface Payload {
  qupathExecutable?: string;
  qupathProcessName?: string;
  displayId?: string;
  displayDpr?: number;
  waitAfterActionMs?: number;
  replanningCycleLimit?: number;
  reportFileName?: string;
  stepLogFile?: string;
  instruction?: string;
  question?: string;
  assertion?: string;
}

export type CommandName = 'env-check' | 'act' | 'query' | 'assert';
