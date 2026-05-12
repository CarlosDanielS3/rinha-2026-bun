const mccRiskPath = process.env.MCC_RISK_PATH ?? "./resources/mcc_risk.json";
const mccRiskFile = Bun.file(mccRiskPath);
const mccRisk: Record<string, number> = await mccRiskFile.json();

const MAX_AMOUNT = 10000;
const MAX_INSTALLMENTS = 12;
const AMOUNT_VS_AVG_RATIO = 10;
const MAX_MINUTES = 1440;
const MAX_KM = 1000;
const MAX_TX_COUNT_24H = 20;
const MAX_MERCHANT_AVG_AMOUNT = 10000;

function clamp(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

export interface TransactionPayload {
  id: string;
  transaction: {
    amount: number;
    installments: number;
    requested_at: string;
  };
  customer: {
    avg_amount: number;
    tx_count_24h: number;
    known_merchants: string[];
  };
  merchant: {
    id: string;
    mcc: string;
    avg_amount: number;
  };
  terminal: {
    is_online: boolean;
    card_present: boolean;
    km_from_home: number;
  };
  last_transaction: {
    timestamp: string;
    km_from_current: number;
  } | null;
}

// Pre-allocated buffer reused across calls to avoid GC pressure
const vectorBuffer = new Float32Array(14);

export function vectorize(payload: TransactionPayload): Float32Array {
  const { transaction, customer, merchant, terminal, last_transaction } =
    payload;

  // dim 0: amount
  vectorBuffer[0] = clamp(transaction.amount / MAX_AMOUNT);

  // dim 1: installments
  vectorBuffer[1] = clamp(transaction.installments / MAX_INSTALLMENTS);

  // dim 2: amount_vs_avg
  vectorBuffer[2] = clamp(
    transaction.amount / customer.avg_amount / AMOUNT_VS_AVG_RATIO,
  );

  // dim 3: hour_of_day (UTC)
  const date = new Date(transaction.requested_at);
  vectorBuffer[3] = date.getUTCHours() / 23;

  // dim 4: day_of_week (mon=0, sun=6)
  // JS: getUTCDay() returns 0=Sun, 1=Mon, ..., 6=Sat
  // Challenge: mon=0, sun=6
  const jsDay = date.getUTCDay();
  const challengeDay = jsDay === 0 ? 6 : jsDay - 1;
  vectorBuffer[4] = challengeDay / 6;

  // dims 5,6: last_transaction dependent
  if (last_transaction === null) {
    vectorBuffer[5] = -1;
    vectorBuffer[6] = -1;
  } else {
    const lastDate = new Date(last_transaction.timestamp);
    const minutesSinceLast = (date.getTime() - lastDate.getTime()) / 60000;
    vectorBuffer[5] = clamp(minutesSinceLast / MAX_MINUTES);
    vectorBuffer[6] = clamp(last_transaction.km_from_current / MAX_KM);
  }

  // dim 7: km_from_home
  vectorBuffer[7] = clamp(terminal.km_from_home / MAX_KM);

  // dim 8: tx_count_24h
  vectorBuffer[8] = clamp(customer.tx_count_24h / MAX_TX_COUNT_24H);

  // dim 9: is_online
  vectorBuffer[9] = terminal.is_online ? 1 : 0;

  // dim 10: card_present
  vectorBuffer[10] = terminal.card_present ? 1 : 0;

  // dim 11: unknown_merchant (1 if NOT in known_merchants)
  vectorBuffer[11] = customer.known_merchants.includes(merchant.id) ? 0 : 1;

  // dim 12: mcc_risk
  vectorBuffer[12] = mccRisk[merchant.mcc] ?? 0.5;

  // dim 13: merchant_avg_amount
  vectorBuffer[13] = clamp(merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT);

  return vectorBuffer;
}
