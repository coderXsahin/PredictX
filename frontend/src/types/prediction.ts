
export type MachineType = "L" | "M" | "H";

export interface MachineReading {
  machine_type: MachineType;
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
}

export interface PredictionResult {
  id: number;
  prediction: 0 | 1;
  status: "HIGH_RISK" | "LOWER_RISK";
  failure_probability: number;
  threshold: number;
  model: string;
  created_at: string;
  message: string;
  disclaimer: string;
}

export interface PredictionHistoryItem {
  id: number;
  machine_type: MachineType;
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
  failure_probability: number;
  threshold: number;
  prediction: 0 | 1;
  status: "HIGH_RISK" | "LOWER_RISK";
  created_at: string;
}

export interface PredictionHistoryResponse {
  count: number;
  items: PredictionHistoryItem[];
}
