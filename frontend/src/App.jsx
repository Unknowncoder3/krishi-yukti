function App() {
  return (
    <div className="app-shell">
      <nav className="navbar navbar-expand-lg navbar-dark bg-success shadow-sm">
        <div className="container">
          <a className="navbar-brand fw-bold" href="#top">🌾 Krishi Yukti</a>
          <span className="navbar-text text-white">AI-powered agriculture ecosystem</span>
        </div>
      </nav>

      <main id="top" className="container py-4">
        <div className="mb-4">
          <h1 className="display-6 fw-bold">Farmer Dashboard</h1>
          <p className="text-secondary mb-0">
            Understand crop health, monitor farm conditions, and make better decisions from one place.
          </p>
        </div>

        <div className="row g-3 mb-4">
          {[
            ["🌦️", "Weather", "Coming soon"],
            ["🌱", "Crop Health", "AI module"],
            ["📈", "Yield Forecast", "AI module"],
            ["🛒", "Marketplace", "Coming soon"],
          ].map(([icon, title, value]) => (
            <div className="col-sm-6 col-lg-3" key={title}>
              <div className="card h-100 border-0 shadow-sm dashboard-card">
                <div className="card-body">
                  <div className="fs-3">{icon}</div>
                  <div className="small text-secondary mt-2">{title}</div>
                  <div className="fw-semibold mt-1">{value}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="row g-4">
          <div className="col-lg-7">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-4">
                <h2 className="h5 fw-bold">AI Crop Doctor</h2>
                <p className="text-secondary">
                  Upload a crop or leaf image to receive an AI-assisted disease assessment. The production model will return the predicted class, confidence, and safe next-step guidance.
                </p>
                <button className="btn btn-success" type="button" disabled>
                  Disease Detection — coming next
                </button>
              </div>
            </div>
          </div>

          <div className="col-lg-5">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-4">
                <h2 className="h5 fw-bold">Build Status</h2>
                <div className="list-group list-group-flush">
                  <div className="list-group-item px-0 d-flex justify-content-between">
                    <span>Repository</span><span className="badge text-bg-success">Ready</span>
                  </div>
                  <div className="list-group-item px-0 d-flex justify-content-between">
                    <span>FastAPI</span><span className="badge text-bg-success">Ready</span>
                  </div>
                  <div className="list-group-item px-0 d-flex justify-content-between">
                    <span>React + Bootstrap</span><span className="badge text-bg-success">Ready</span>
                  </div>
                  <div className="list-group-item px-0 d-flex justify-content-between">
                    <span>ML pipeline</span><span className="badge text-bg-warning">Next</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
