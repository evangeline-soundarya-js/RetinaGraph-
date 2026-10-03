import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, Activity, AlertCircle, FileText, Share2, Info, CheckCircle, Database, MessageSquare, Download } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const [feedback, setFeedback] = useState({
    assessment: '',
    agreement: 'pending',
    notes: ''
  });

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
      setFeedback({ assessment: '', agreement: 'pending', notes: '' });
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

  const handleExport = () => {
    if (!result) return;
    const summary = {
      image_name: file?.name,
      timestamp: new Date().toISOString(),
      model_result: result,
      reviewer_feedback: feedback
    };
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `retinagraph_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">
          <Activity size={32} color="#4fd1c5" />
          <h1>RetinaGraph AI</h1>
          <span className="subtitle">Explainable Retinal Graph Analysis</span>
        </div>
        <div className="badge prototype-badge">
          Research Prototype
        </div>
      </header>

      <div className="disclaimer">
        <AlertCircle size={20} />
        <p><strong>Research Purpose Only:</strong> This application provides structural graph analysis for research review. It is NOT clinically validated and must not be used for medical diagnosis.</p>
      </div>

      <main className="main-layout">
        <div className="left-column">
          <div className="panel upload-panel">
            <h2>Image Input</h2>
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

            {file && (
              <div className="controls">
                <button 
                  className="analyze-btn primary-btn" 
                  onClick={handleAnalyze} 
                  disabled={loading}
                >
                  {loading ? <span className="loader"></span> : 'Run Analysis'}
                </button>
                <button 
                  className="reset-btn"
                  onClick={() => { setFile(null); setPreview(null); setResult(null); setError(null); }}
                  disabled={loading}
                >
                  Clear
                </button>
              </div>
            )}
            
            {loading && (
              <div className="analysis-progress">
                <p>1. Image Validation...</p>
                <p>2. CLAHE Preprocessing...</p>
                <p>3. Topology Feature Extraction...</p>
                <p>4. KNN Graph Construction...</p>
                <p>5. GAT Inference...</p>
              </div>
            )}
            
            {error && (
              <div className="error-state">
                <AlertCircle size={24} color="#e53e3e" />
                <p>{error}</p>
              </div>
            )}
            
            {result && result.status === "rejected" && (
              <div className="error-state">
                <AlertCircle size={24} color="#e53e3e" />
                <p><strong>Validation Failed:</strong> {result.reason}</p>
              </div>
            )}
          </div>
        </div>

        <div className="right-column">
          {result && result.status !== "rejected" ? (
            <>
              <div className="panel prediction-panel">
                <h2>Model Prediction</h2>
                <div className="prediction-content">
                  <div className="prediction-main">
                    <CheckCircle size={32} color="#4fd1c5" />
                    <span className="pred-text">{result.prediction || "N/A"}</span>
                  </div>
                  <div className="prediction-meta">
                    <p><strong>Confidence:</strong> {result.confidence ? (result.confidence * 100).toFixed(1) + "%" : "Unavailable"}</p>
                    <p><strong>Status:</strong> {result.model_status}</p>
                  </div>
                </div>
              </div>

              <div className="panel evidence-panel">
                <h2>Evidence Visualization</h2>
                <div className="evidence-viewer">
                  <div className="image-overlay-container" style={{ position: 'relative', width: '100%', maxWidth: '512px', margin: '0 auto' }}>
                    <img src={preview} alt="Fundus" style={{ width: '100%', display: 'block', borderRadius: '4px' }} />
                    <svg viewBox="0 0 512 512" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                      {/* Edges */}
                      {result.explanation?.top_edges?.map((edge, i) => (
                        <line 
                          key={`edge-${i}`}
                          x1={edge.source_coords[0]} 
                          y1={edge.source_coords[1]} 
                          x2={edge.target_coords[0]} 
                          y2={edge.target_coords[1]} 
                          stroke="rgba(236, 201, 75, 0.8)" 
                          strokeWidth="3" 
                        />
                      ))}
                      {/* Nodes */}
                      {result.explanation?.top_nodes?.map((node, i) => (
                        <circle 
                          key={`node-${i}`}
                          cx={node.coordinates[0]} 
                          cy={node.coordinates[1]} 
                          r="6" 
                          fill="rgba(245, 101, 101, 0.9)" 
                          stroke="white"
                          strokeWidth="1.5"
                        />
                      ))}
                    </svg>
                  </div>
                  <p className="caption">Top {result.explanation?.top_nodes?.length || 0} highest-attention topological junctions.</p>
                </div>
                
                <div className="graph-stats">
                  <div className="stat-box">
                    <Database size={16} /> Nodes: {result.graph?.nodes}
                  </div>
                  <div className="stat-box">
                    <Share2 size={16} /> Edges: {result.graph?.edges}
                  </div>
                </div>
              </div>

              <div className="panel feedback-panel">
                <h2>Reviewer Feedback</h2>
                <div className="feedback-form">
                  <div className="form-group">
                    <label>Agreement with Model:</label>
                    <select 
                      value={feedback.agreement} 
                      onChange={(e) => setFeedback({...feedback, agreement: e.target.value})}
                    >
                      <option value="pending">Pending Review</option>
                      <option value="agree">Agree</option>
                      <option value="disagree">Disagree</option>
                      <option value="unsure">Unsure / Need More Data</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Reviewer Notes / Structural Assessment:</label>
                    <textarea 
                      rows="4" 
                      placeholder="Enter research notes on highlighted structures..."
                      value={feedback.notes}
                      onChange={(e) => setFeedback({...feedback, notes: e.target.value})}
                    ></textarea>
                  </div>
                </div>
              </div>

              <div className="action-panel">
                <button className="export-btn" onClick={handleExport}>
                  <Download size={18} /> Export JSON Report
                </button>
              </div>
            </>
          ) : (
            <div className="panel empty-panel">
              <FileText size={48} color="#718096" />
              <p>Analysis results will appear here.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
