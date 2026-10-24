
import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Cpu,
  Gauge,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Thermometer,
  Wrench,
  Zap,
} from "lucide-react";
import {
  fetchPredictionHistory,
  predictMachine,
} from "./services/predictionApi";
import type {
  MachineReading,
  PredictionHistoryItem,
  PredictionResult,
} from "./types/prediction";
import "./App.css";

const initialReading: MachineReading = {
  machine_type: "M",
  air_temperature: 300,
  process_temperature: 310,
  rotational_speed: 1500,
  torque: 45,
  tool_wear: 180,
};

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

function App() {
  const [reading, setReading] = useState<MachineReading>(initialReading);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(true);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);

    try {
      const data = await fetchPredictionHistory(20);
      setHistory(data.items);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
      setError("Cannot load prediction history. Check that the API is running.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  function updateReading<K extends keyof MachineReading>(
    key: K,
    value: MachineReading[K],
  ) {
    setReading((previous) => ({ ...previous, [key]: value }));
  }

  async function handlePredict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const prediction = await predictMachine(reading);
      setResult(prediction);
      setApiOnline(true);
      await loadHistory();
    } catch (err) {
      setApiOnline(false);
      setError(
        err instanceof Error
          ? err.message
          : "Prediction failed. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  const highRiskCount = history.filter(
    (item) => item.status === "HIGH_RISK",
  ).length;

  const averageProbability =
    history.length === 0
      ? 0
      : history.reduce(
          (total, item) => total + item.failure_probability,
          0,
        ) / history.length;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Activity size={22} /></div>
          <div>
            <strong>PredictX</strong>
            <span>INTELLIGENCE PLATFORM</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <a className="nav-item active" href="#overview">
          <Gauge size={18} /> Overview
        </a>
        <a className="nav-item" href="#prediction">
          <Cpu size={18} /> Machine prediction
        </a>
        <a className="nav-item" href="#history">
          <Clock3 size={18} /> Prediction history
        </a>

        <div className="sidebar-bottom">
          <div className="system-card">
            <span className={`status-dot ${apiOnline ? "" : "offline"}`} />
            <div>
              <strong>{apiOnline ? "API connected" : "API unavailable"}</strong>
              <span>PredictX backend</span>
            </div>
          </div>
          <span className="sidebar-version">PREDICTX · V1.1</span>
        </div>
      </aside>

      <section className="main-panel" id="overview">
        <header className="topbar">
          <div>
            <div className="eyebrow">INDUSTRIAL ANALYTICS / OVERVIEW</div>
            <h1>Machine health overview</h1>
            <p>Predict potential equipment failures before they happen.</p>
          </div>
          <button
            className="secondary-button"
            onClick={() => void loadHistory()}
            disabled={historyLoading}
          >
            <RefreshCw size={16} /> Refresh data
          </button>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={18} /> {error}
            <button onClick={() => setError("")} aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}

        <section className="stats-grid" aria-label="Prediction statistics">
          <article className="stat-card">
            <div className="stat-top">
              <span>Total predictions loaded</span>
              <Activity size={18} />
            </div>
            <strong>{history.length}</strong>
            <small>Most recent 20 records</small>
          </article>

          <article className="stat-card">
            <div className="stat-top">
              <span>High-risk alerts</span>
              <AlertTriangle size={18} />
            </div>
            <strong>{highRiskCount}</strong>
            <small>Within loaded history</small>
          </article>

          <article className="stat-card">
            <div className="stat-top">
              <span>Average predicted risk</span>
              <Gauge size={18} />
            </div>
            <strong>{formatPercent(averageProbability)}</strong>
            <small>Mean of loaded probabilities</small>
          </article>

          <article className="stat-card">
            <div className="stat-top">
              <span>Configured threshold</span>
              <ShieldCheck size={18} />
            </div>
            <strong>40%</strong>
            <small>Current alert decision threshold</small>
          </article>
        </section>

        <section className="content-grid">
          <article className="panel prediction-panel" id="prediction">
            <div className="panel-heading">
              <div className="panel-icon"><Cpu size={19} /></div>
              <div>
                <h2>Run a prediction</h2>
                <p>Enter the current machine sensor readings.</p>
              </div>
            </div>

            <form onSubmit={handlePredict}>
              <label className="field">
                <span>Machine type</span>
                <select
                  value={reading.machine_type}
                  onChange={(event) =>
                    updateReading(
                      "machine_type",
                      event.target.value as MachineReading["machine_type"],
                    )
                  }
                >
                  <option value="L">L — Low</option>
                  <option value="M">M — Medium</option>
                  <option value="H">H — High</option>
                </select>
              </label>

              <div className="form-grid">
                <label className="field">
                  <span><Thermometer size={15} /> Air temperature (K)</span>
                  <input
                    type="number"
                    min="250"
                    max="350"
                    step="0.1"
                    required
                    value={reading.air_temperature}
                    onChange={(event) =>
                      updateReading("air_temperature", Number(event.target.value))
                    }
                  />
                </label>

                <label className="field">
                  <span><Thermometer size={15} /> Process temperature (K)</span>
                  <input
                    type="number"
                    min="250"
                    max="400"
                    step="0.1"
                    required
                    value={reading.process_temperature}
                    onChange={(event) =>
                      updateReading(
                        "process_temperature",
                        Number(event.target.value),
                      )
                    }
                  />
                </label>

                <label className="field">
                  <span><Zap size={15} /> Rotational speed (rpm)</span>
                  <input
                    type="number"
                    min="1"
                    max="10000"
                    required
                    value={reading.rotational_speed}
                    onChange={(event) =>
                      updateReading(
                        "rotational_speed",
                        Number(event.target.value),
                      )
                    }
                  />
                </label>

                <label className="field">
                  <span><Gauge size={15} /> Torque (Nm)</span>
                  <input
                    type="number"
                    min="0"
                    max="1000"
                    step="0.1"
                    required
                    value={reading.torque}
                    onChange={(event) =>
                      updateReading("torque", Number(event.target.value))
                    }
                  />
                </label>

                <label className="field full-width">
                  <span><Wrench size={15} /> Tool wear (minutes)</span>
                  <input
                    type="number"
                    min="0"
                    max="10000"
                    required
                    value={reading.tool_wear}
                    onChange={(event) =>
                      updateReading("tool_wear", Number(event.target.value))
                    }
                  />
                </label>
              </div>

              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? (
                  <><LoaderCircle className="spin" size={17} /> Analyzing machine...</>
                ) : (
                  <><Activity size={17} /> Analyze machine health</>
                )}
              </button>
              <p className="form-note">
                Predictions are experimental and do not guarantee machine safety.
              </p>
            </form>
          </article>

          <article className="panel result-panel">
            <div className="panel-heading">
              <div className="panel-icon"><ShieldCheck size={19} /></div>
              <div>
                <h2>Prediction result</h2>
                <p>Output from the trained Random Forest model.</p>
              </div>
            </div>

            {!result ? (
              <div className="empty-result">
                <div className="empty-icon"><Activity size={30} /></div>
                <h3>Awaiting analysis</h3>
                <p>Submit sensor readings to see the model's risk estimate.</p>
              </div>
            ) : (
              <div className="result-content">
                <div className={`risk-banner ${result.status === "HIGH_RISK" ? "risk-high" : "risk-low"}`}>
                  {result.status === "HIGH_RISK" ? (
                    <AlertTriangle size={21} />
                  ) : (
                    <CheckCircle2 size={21} />
                  )}
                  <div>
                    <span>MODEL RISK CLASSIFICATION</span>
                    <strong>{result.status.replace("_", " ")}</strong>
                  </div>
                </div>

                <div className="probability-block">
                  <div className="probability-label">
                    <span>Failure probability</span>
                    <strong>{formatPercent(result.failure_probability)}</strong>
                  </div>
                  <div
                    className="probability-track"
                    role="progressbar"
                    aria-label="Failure probability"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={result.failure_probability * 100}
                  >
                    <div
                      className={`probability-fill ${result.status === "HIGH_RISK" ? "fill-high" : ""}`}
                      style={{
                        width: `${result.failure_probability * 100}%`,
                      }}
                    />
                  </div>
                  <div className="scale-labels">
                    <span>0% risk</span><span>100% risk</span>
                  </div>
                </div>

                <div className="result-detail">
                  <span>Decision threshold</span>
                  <strong>{formatPercent(result.threshold)}</strong>
                </div>
                <div className="result-detail">
                  <span>Prediction ID</span>
                  <strong>#{result.id}</strong>
                </div>
                <div className="result-detail">
                  <span>Created at</span>
                  <strong>{formatDate(result.created_at)}</strong>
                </div>

                <div className="recommendation">
                  {result.message}
                </div>
                <p className="disclaimer">{result.disclaimer}</p>
              </div>
            )}
          </article>
        </section>

        <section className="panel history-panel" id="history">
          <div className="history-heading">
            <div className="panel-heading">
              <div className="panel-icon"><Clock3 size={19} /></div>
              <div>
                <h2>Recent prediction history</h2>
                <p>Persisted prediction records from PostgreSQL.</p>
              </div>
            </div>
            <button
              className="text-button"
              onClick={() => void loadHistory()}
              disabled={historyLoading}
            >
              View latest <RefreshCw size={14} />
            </button>
          </div>

          {historyLoading ? (
            <div className="table-message">
              <LoaderCircle className="spin" size={20} /> Loading history...
            </div>
          ) : history.length === 0 ? (
            <div className="table-message">
              No predictions yet. Run your first machine analysis above.
            </div>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Machine</th>
                    <th>Torque</th>
                    <th>Speed</th>
                    <th>Tool wear</th>
                    <th>Failure probability</th>
                    <th>Status</th>
                    <th>Created at</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id}>
                      <td>#{item.id}</td>
                      <td>{item.machine_type}</td>
                      <td>{item.torque.toFixed(1)} Nm</td>
                      <td>{item.rotational_speed} rpm</td>
                      <td>{item.tool_wear} min</td>
                      <td>{formatPercent(item.failure_probability)}</td>
                      <td>
                        <span className={`status-pill ${item.status === "HIGH_RISK" ? "pill-high" : "pill-low"}`}>
                          {item.status === "HIGH_RISK" ? (
                            <AlertTriangle size={12} />
                          ) : (
                            <CheckCircle2 size={12} />
                          )}
                          {item.status.replace("_", " ")}
                        </span>
                      </td>
                      <td>{formatDate(item.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="table-footer">
            <span>Showing up to 20 recent records</span>
            <span><span className="status-dot" /> Connected to prediction history API</span>
          </div>
        </section>

        <footer className="footer">
          <span>PredictX · Predictive Maintenance Intelligence</span>
          <span>Experimental model · Not a safety certification</span>
        </footer>
      </section>
    </main>
  );
}

export default App;
