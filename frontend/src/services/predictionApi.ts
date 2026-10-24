
import type {
  MachineReading,
  PredictionHistoryResponse,
  PredictionResult,
} from "../types/prediction";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;

    throw new Error(
      typeof detail === "string"
        ? detail
        : `Request failed with status ${response.status}`,
    );
  }

  return response.json() as Promise<T>;
}

export function predictMachine(
  reading: MachineReading,
): Promise<PredictionResult> {
  return request<PredictionResult>("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify(reading),
  });
}

export function fetchPredictionHistory(
  limit = 20,
): Promise<PredictionHistoryResponse> {
  return request<PredictionHistoryResponse>(
    `/api/v1/predictions?limit=${limit}`,
  );
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await request<{ status: string }>("/health");
    return response.status === "UP";
  } catch {
    return false;
  }
}
