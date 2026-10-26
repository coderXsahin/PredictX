
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import {
  fetchPredictionHistory,
  predictMachine,
} from "./services/predictionApi";

vi.mock("./services/predictionApi", () => ({
  checkApiHealth: vi.fn(),
  fetchPredictionHistory: vi.fn(),
  predictMachine: vi.fn(),
}));

const mockedFetchPredictionHistory = vi.mocked(fetchPredictionHistory);
const mockedPredictMachine = vi.mocked(predictMachine);

beforeEach(() => {
  mockedFetchPredictionHistory.mockResolvedValue({
    count: 0,
    items: [],
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PredictX dashboard", () => {
  it("renders the dashboard", () => {
    render(<App />);

    expect(
      screen.getByText("INTELLIGENCE PLATFORM"),
    ).toBeTruthy();

    expect(
      screen.getByRole("link", { name: /overview/i }),
    ).toBeTruthy();
  });

  it("loads prediction history", async () => {
    render(<App />);

    await waitFor(() => {
      expect(mockedFetchPredictionHistory).toHaveBeenCalled();
    });
  });

  it("submits a machine reading and displays the prediction", async () => {
    const user = userEvent.setup();

    mockedPredictMachine.mockResolvedValue({
      id: 101,
      prediction: 0,
      status: "LOWER_RISK",
      failure_probability: 0.017,
      threshold: 0.4,
      model: "Random Forest",
      created_at: "2026-10-25T10:00:00Z",
      message: "Machine is operating at a lower predicted risk.",
      disclaimer: "This prediction is an estimate, not a guarantee.",
    });

    render(<App />);

    const submitButton = await screen.findByRole("button", {
      name: /predict|analy[sz]e|check risk/i,
    });

    await user.click(submitButton);

    await waitFor(() => {
      expect(mockedPredictMachine).toHaveBeenCalledTimes(1);
    });

    expect(
      await screen.findByText(/lower risk/i),
    ).toBeTruthy();
  });
});
