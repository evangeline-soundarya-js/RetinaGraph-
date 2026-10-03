import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Activity, AlertCircle, FileText, Share2, Info, CheckCircle, Database } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">
          <Activity size={32} color="#4fd1c5" />
          <h1>RetinaGraph AI</h1>
        </div>
        <div className="badge prototype-badge">
          Research Prototype
        </div>
      </header>

      <main className="main-content">
        <div className="disclaimer">
          <AlertCircle size={20} />
          <p><strong>Research Purpose Only:</strong> This application is a prototype architecture for graph-based fundus analysis. It is NOT clinically validated and must not be used for medical diagnosis. The model weights are currently untrained.</p>
        </div>

        <div className="dashboard">
          <div className="panel upload-panel">
            <h2>1. Input Image</h2>
            
            <div 
              className={`dropzone ${preview ? 'has-image' : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              {preview ? (
                <img src={preview} alt="Fundus preview" className="preview-image" />
              ) : (
                <div className="dropzone-content">
                  <Upload size={48} color="#718096" />
                  <p>Click to upload fundus image</p>
                  <span>Supports JPG, JPEG, PNG</span>
                </div>
              )}
            </div>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept=".jpg,.jpeg,.png" 
              style={{ display: 'none' }} 
            />

            <button 
              className="analyze-btn" 
              onClick={handleAnalyze} 
              disabled={!file || loading}
            >
              {loading ? <span className="loader"></span> : 'Run Analysis Pipeline'}
            </button>
          </div>

          <div className="panel results-panel">
            <h2>2. Analysis Results</h2>
            
            {!result && !loading && !error && (
              <div className="empty-state">
                <FileText size={48} color="#718096" />
                <p>Upload an image and run the analysis to see results.</p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="pulse-graph">
                  <Share2 size={64} className="spinning" color="#4fd1c5" />
                </div>
                <p>Constructing Graph & Running GAT Inference...</p>
              </div>
            )}

            {error && (
              <div className="error-state">
                <AlertCircle size={32} color="#e53e3e" />
                <p>{error}</p>
              </div>
            )}

            {result && result.status === "rejected" && (
              <div className="error-state">
                <AlertCircle size={48} color="#e53e3e" />
                <h3 style={{ marginTop: '1rem', color: '#fc8181' }}>Image not suitable for retinal analysis.</h3>
                <p style={{ marginTop: '0.5rem' }}>{result.reason}</p>
              </div>
            )}

            {result && result.status !== "rejected" && (
              <div className="results-content">
                <div className="result-card primary">
                  <div className="card-header">
                    <CheckCircle size={24} color="#4fd1c5" />
                    <h3>Prediction</h3>
                  </div>
                  <div className="prediction-value">{result.prediction}</div>
                  <div className="model-status">
                    <Info size={16} /> Status: {result.model_status}
                  </div>
                </div>

                <div className="stats-grid">
                  <div className="result-card">
                    <div className="card-header">
                      <Share2 size={20} />
                      <h3>Graph Stats</h3>
                    </div>
                    <div className="stat-row">
                      <span>Nodes (Keypoints):</span>
                      <strong>{result.graph?.nodes}</strong>
                    </div>
                    <div className="stat-row">
                      <span>Edges (Connections):</span>
                      <strong>{result.graph?.edges}</strong>
                    </div>
                  </div>

                  <div className="result-card">
                    <div className="card-header">
                      <Database size={20} />
                      <h3>Evidence</h3>
                    </div>
                    <ul className="evidence-list">
                      {result.evidence?.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="result-card explanation">
                  <div className="card-header">
                    <Info size={20} />
                    <h3>Explanation Layer</h3>
                  </div>
                  <p className="explanation-text">{result.explanation?.message}</p>
                  <div className="meta-info">
                    Method: {result.explanation?.important_regions_method}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
