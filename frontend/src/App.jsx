import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selected) => {
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API_URL}/api/v1/ai/disease/predict`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell min-vh-100 bg-light">
      <nav className="navbar navbar-dark bg-success shadow-sm">
        <div className="container py-2">
          <span className="navbar-brand fw-bold">🌾 Krishi Yukti</span>
          <span className="navbar-text text-white">AI-powered agriculture ecosystem</span>
        </div>
      </nav>

      <main className="container py-4">
        <div className="mb-4">
          <h1 className="display-6 fw-bold">AI Crop Doctor</h1>
          <p className="text-secondary">Upload a crop or leaf image for an AI-assisted disease assessment.</p>
        </div>

        <div className="row g-4">
          <div className="col-lg-7">
            <div className="card border-0 shadow-sm">
              <div className="card-body p-4">
                <h2 className="h5 fw-bold">1. Upload crop image</h2>
                <input
                  className="form-control my-3"
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFile(e.target.files?.[0])}
                />
                {preview && (
                  <img src={preview} alt="Selected crop" className="img-fluid rounded mb-3" style={{ maxHeight: 360, objectFit: "contain", width: "100%" }} />
                )}
                <button className="btn btn-success" disabled={!file || loading} onClick={analyze}>
                  {loading ? "Analyzing…" : "Analyze Crop"}
                </button>
                {error && <div className="alert alert-danger mt-3 mb-0">{error}</div>}
              </div>
            </div>
          </div>

          <div className="col-lg-5">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-4">
                <h2 className="h5 fw-bold">2. AI assessment</h2>
                {!result && <p className="text-secondary">Your prediction will appear here after analysis.</p>}
                {result && (
                  <>
                    <div className="alert alert-success">
                      <div className="small text-uppercase">Predicted condition</div>
                      <div className="fs-4 fw-bold">{result.disease.replaceAll("___", " — ")}</div>
                    </div>
                    <div className="mb-3">
                      <div className="d-flex justify-content-between"><span>Confidence</span><strong>{(result.confidence * 100).toFixed(1)}%</strong></div>
                      <div className="progress mt-2" role="progressbar">
                        <div className="progress-bar bg-success" style={{ width: `${result.confidence * 100}%` }} />
                      </div>
                    </div>
                    <div className="alert alert-warning small mb-0">{result.message}</div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
