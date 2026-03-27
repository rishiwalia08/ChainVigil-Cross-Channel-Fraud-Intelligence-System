import { useState, useCallback, useRef, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function App() {
  const [activeTab, setActiveTab] = useState('pipeline')
  const [loading, setLoading] = useState(false)
  const [logs, setLogs] = useState([])
  const [pipelineState, setPipelineState] = useState({
    generated: false,
    ingested: false,
    trained: false,
    analyzed: false,
  })
  const [stats, setStats] = useState(null)
  const [accounts, setAccounts] = useState([])
  const [clusters, setClusters] = useState([])
  const [riskDist, setRiskDist] = useState(null)
  const [selectedAccount, setSelectedAccount] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [report, setReport] = useState(null)
  const [trainingResults, setTrainingResults] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [graphLoading, setGraphLoading] = useState(false)
  const [hoveredNode, setHoveredNode] = useState(null)
  const graphRef = useRef(null)
  const logRef = useRef(null)

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  const addLog = useCallback((msg, type = 'info') => {
    const ts = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, { msg: `[${ts}] ${msg}`, type }])
  }, [])

  const apiCall = useCallback(async (endpoint, method = 'POST', body = null) => {
    try {
      const opts = { method, headers: { 'Content-Type': 'application/json' } }
      if (body) opts.body = JSON.stringify(body)
      const res = await fetch(`${API_BASE}${endpoint}`, opts)
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'API Error')
      }
      return await res.json()
    } catch (e) {
      addLog(`Error: ${e.message}`, 'error')
      throw e
    }
  }, [addLog])

  // ─── Pipeline Actions ────────────────────────────────────

  const runGenerate = async () => {
    setLoading(true)
    addLog('Starting synthetic data generation...', 'info')
    try {
      const res = await apiCall('/api/generate')
      addLog(`✅ ${res.message}`, 'success')
      setPipelineState(s => ({ ...s, generated: true }))
    } catch { /* logged */ } finally { setLoading(false) }
  }

  const runIngest = async () => {
    setLoading(true)
    addLog('Building Unified Entity Graph...', 'info')
    try {
      const res = await apiCall('/api/ingest')
      addLog(`✅ ${res.message}`, 'success')
      setStats(res.details)
      setPipelineState(s => ({ ...s, ingested: true }))
    } catch { /* logged */ } finally { setLoading(false) }
  }

  const runTrain = async () => {
    setLoading(true)
    addLog('Training GNN model (this may take a minute)...', 'warning')
    try {
      const res = await apiCall('/api/train')
      addLog(`✅ ${res.message}`, 'success')
      setTrainingResults(res.details)
      setPipelineState(s => ({ ...s, trained: true }))
    } catch { /* logged */ } finally { setLoading(false) }
  }

  const runAnalyze = async () => {
    setLoading(true)
    addLog('Running risk analysis & cluster detection...', 'info')
    try {
      const res = await apiCall('/api/analyze')
      addLog(`✅ ${res.message}`, 'success')
      setRiskDist(res.details?.risk_distribution)
      setPipelineState(s => ({ ...s, analyzed: true }))
      // Fetch accounts and clusters
      fetchAccounts()
      fetchClusters()
    } catch { /* logged */ } finally { setLoading(false) }
  }

  const runFullPipeline = async () => {
    setLoading(true)
    addLog('🚀 Running full pipeline...', 'info')
    try {
      const res = await apiCall('/api/pipeline/run')
      addLog(`✅ ${res.message}`, 'success')
      setTrainingResults(res.details?.training)
      setRiskDist(res.details?.risk?.distribution)
      setPipelineState({ generated: true, ingested: true, trained: true, analyzed: true })
      fetchAccounts()
      fetchClusters()
    } catch { /* logged */ } finally { setLoading(false) }
  }

  const fetchAccounts = async () => {
    try {
      const res = await apiCall('/api/accounts?limit=100', 'GET')
      setAccounts(res.accounts || [])
    } catch { /* */ }
  }

  const fetchClusters = async () => {
    try {
      const res = await apiCall('/api/clusters', 'GET')
      setClusters(res.clusters || [])
    } catch { /* */ }
  }

  const fetchExplanation = async (accountId) => {
    setSelectedAccount(accountId)
    addLog(`Generating XAI explanation for ${accountId}...`, 'info')
    try {
      const res = await apiCall(`/api/explain/${accountId}`, 'GET')
      setExplanation(res)
      addLog(`✅ Explanation generated for ${accountId}`, 'success')
    } catch { /* */ }
  }

  const fetchReport = async () => {
    addLog('Generating full audit report...', 'info')
    try {
      const res = await apiCall('/api/report', 'GET')
      setReport(res)
      addLog('✅ Audit report generated', 'success')
    } catch { /* */ }
  }

  const fetchGraphData = async () => {
    setGraphLoading(true)
    addLog('Loading graph visualization data...', 'info')
    try {
      const res = await apiCall('/api/graph/visual?max_nodes=500', 'GET')
      setGraphData(res)
      addLog(`✅ Graph loaded: ${res.showing_nodes} nodes, ${res.showing_links} links`, 'success')
    } catch { /* */ } finally { setGraphLoading(false) }
  }

  // ─── Helpers ─────────────────────────────────────────────

  const getRiskColor = (score) => {
    if (score >= 0.85) return 'var(--color-danger)'
    if (score >= 0.6) return 'var(--color-warning)'
    if (score >= 0.4) return 'var(--accent-blue)'
    return 'var(--accent-green)'
  }

  const getRiskLevel = (score) => {
    if (score >= 0.85) return 'high'
    if (score >= 0.6) return 'medium'
    return 'low'
  }

  const getActionClass = (action) => {
    return (action || '').toLowerCase()
  }

  // ─── Render ──────────────────────────────────────────────

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">⛓</div>
          <div>
            <div className="header-title">ChainVigil</div>
            <div className="header-subtitle">Cross-Channel Mule Detection</div>
          </div>
        </div>
        <div className="header-status">
          <div className="status-badge">
            <span className={`status-dot ${pipelineState.analyzed ? 'active' : 'inactive'}`} />
            {pipelineState.analyzed ? 'Analysis Active' : 'Idle'}
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav-tabs">
        {[
          { id: 'pipeline', icon: '🔄', label: 'Pipeline' },
          { id: 'graph', icon: '🔗', label: 'Graph' },
          { id: 'accounts', icon: '👤', label: 'Accounts' },
          { id: 'clusters', icon: '🕸️', label: 'Clusters' },
          { id: 'xai', icon: '🧠', label: 'XAI Auditor' },
          { id: 'report', icon: '📄', label: 'Reports' },
        ].map(tab => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="nav-tab-icon">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </nav>

      {/* ═══ Pipeline Tab ═══ */}
      {activeTab === 'pipeline' && (
        <div>
          <div className="pipeline-section">
            <div className="pipeline-steps">
              {[
                { n: 1, label: 'Generate Data', desc: 'Synthetic transactions', done: pipelineState.generated },
                { n: 2, label: 'Build Graph', desc: 'Unified Entity Graph', done: pipelineState.ingested },
                { n: 3, label: 'Train GNN', desc: 'GraphSAGE + GAT', done: pipelineState.trained },
                { n: 4, label: 'Analyze Risk', desc: 'Cluster detection', done: pipelineState.analyzed },
              ].map(step => (
                <div key={step.n} className={`pipeline-step ${step.done ? 'completed' : ''}`}>
                  <div className="pipeline-step-number">{step.done ? '✓' : step.n}</div>
                  <div>
                    <div className="pipeline-step-label">{step.label}</div>
                    <div className="pipeline-step-desc">{step.desc}</div>
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <button id="btn-run-pipeline" className="btn btn-primary" onClick={runFullPipeline} disabled={loading}>
                {loading ? <span className="spinner" /> : '🚀'} Run Full Pipeline
              </button>
              <button id="btn-generate" className="btn btn-secondary" onClick={runGenerate} disabled={loading}>
                Generate Data
              </button>
              <button id="btn-ingest" className="btn btn-secondary" onClick={runIngest}
                disabled={loading || !pipelineState.generated}>
                Build Graph
              </button>
              <button id="btn-train" className="btn btn-secondary" onClick={runTrain}
                disabled={loading || !pipelineState.ingested}>
                Train Model
              </button>
              <button id="btn-analyze" className="btn btn-secondary" onClick={runAnalyze}
                disabled={loading || !pipelineState.trained}>
                Analyze Risk
              </button>
            </div>
          </div>

          {/* Stats */}
          {(stats || riskDist) && (
            <div className="stats-grid">
              {stats && (
                <>
                  <div className="stat-card cyan">
                    <div className="stat-label">Nodes</div>
                    <div className="stat-value">{(stats.nx_nodes || 0).toLocaleString()}</div>
                  </div>
                  <div className="stat-card purple">
                    <div className="stat-label">Edges</div>
                    <div className="stat-value">{(stats.nx_edges || 0).toLocaleString()}</div>
                  </div>
                </>
              )}
              {riskDist && (
                <>
                  <div className="stat-card rose">
                    <div className="stat-label">Escalate</div>
                    <div className="stat-value">{riskDist.escalate || 0}</div>
                  </div>
                  <div className="stat-card amber">
                    <div className="stat-label">Freeze</div>
                    <div className="stat-value">{riskDist.freeze || 0}</div>
                  </div>
                  <div className="stat-card emerald">
                    <div className="stat-label">Monitor</div>
                    <div className="stat-value">{riskDist.monitor || 0}</div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Training Results */}
          {trainingResults && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <span className="card-title">📊 Training Results</span>
              </div>
              <div className="stats-grid">
                <div className="stat-card cyan">
                  <div className="stat-label">AUC-ROC</div>
                  <div className="stat-value">{(trainingResults.best_val_auc || 0).toFixed(4)}</div>
                </div>
                <div className="stat-card emerald">
                  <div className="stat-label">Test F1</div>
                  <div className="stat-value">
                    {(trainingResults.test_metrics?.f1 || 0).toFixed(4)}
                  </div>
                </div>
                <div className="stat-card purple">
                  <div className="stat-label">Precision</div>
                  <div className="stat-value">
                    {(trainingResults.test_metrics?.precision || 0).toFixed(4)}
                  </div>
                </div>
                <div className="stat-card amber">
                  <div className="stat-label">Recall</div>
                  <div className="stat-value">
                    {(trainingResults.test_metrics?.recall || 0).toFixed(4)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Risk Distribution */}
          {riskDist && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <span className="card-title">⚖️ Risk Distribution</span>
              </div>
              <div className="distribution-chart">
                {[
                  { label: 'Escalate', count: riskDist.escalate || 0, color: 'var(--color-danger)' },
                  { label: 'Freeze', count: riskDist.freeze || 0, color: 'var(--color-warning)' },
                  { label: 'Monitor', count: riskDist.monitor || 0, color: 'var(--accent-blue)' },
                  { label: 'Clear', count: riskDist.clear || 0, color: 'var(--accent-green)' },
                ].map(bar => {
                  const maxCount = Math.max(riskDist.escalate || 0, riskDist.freeze || 0,
                    riskDist.monitor || 0, riskDist.clear || 0, 1)
                  const height = (bar.count / maxCount) * 150
                  return (
                    <div key={bar.label} className="distribution-bar">
                      <div className="distribution-bar-count" style={{ color: bar.color }}>
                        {bar.count}
                      </div>
                      <div className="distribution-bar-fill" style={{
                        height: `${height}px`, background: bar.color, opacity: 0.7
                      }} />
                      <div className="distribution-bar-label">{bar.label}</div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Log Output */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">📟 Pipeline Log</span>
              <button className="btn btn-sm btn-secondary" onClick={() => setLogs([])}>Clear</button>
            </div>
            <div className="log-output" ref={logRef}>
              {logs.length === 0 && (
                <div className="log-line" style={{ opacity: 0.5 }}>
                  Waiting for pipeline execution...
                </div>
              )}
              {logs.map((log, i) => (
                <div key={i} className={`log-line ${log.type}`}>{log.msg}</div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ═══ Graph Tab ═══ */}
      {activeTab === 'graph' && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h2 className="section-title">🔗 Unified Entity Graph</h2>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {graphData && (
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Showing {graphData.showing_nodes}/{graphData.total_nodes_in_graph} nodes •{' '}
                  {graphData.showing_links}/{graphData.total_edges_in_graph} edges
                </span>
              )}
              <button className="btn btn-primary btn-sm" onClick={fetchGraphData}
                disabled={graphLoading || !pipelineState.ingested}>
                {graphLoading ? <span className="spinner" /> : '🔄'} Load Graph
              </button>
            </div>
          </div>

          {/* Legend */}
          <div style={{
            display: 'flex', gap: 20, marginBottom: 16, padding: '10px 16px',
            background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)', flexWrap: 'wrap', fontSize: 12
          }}>
            <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>NODES:</span>
            {[
              { label: 'Account', color: '#3b82f6' },
              { label: 'Mule Account', color: '#f43f5e' },
              { label: 'Device', color: '#10b981' },
              { label: 'IP Address', color: '#a78bfa' },
              { label: 'ATM', color: '#f59e0b' },
            ].map(item => (
              <span key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: item.color, display: 'inline-block'
                }} />
                {item.label}
              </span>
            ))}
            <span style={{ color: 'var(--text-muted)', fontWeight: 600, marginLeft: 12 }}>EDGES:</span>
            {[
              { label: 'Transfer', color: 'rgba(59, 130, 246, 0.4)' },
              { label: 'Suspicious', color: 'rgba(244, 63, 94, 0.7)' },
              { label: 'Device/IP', color: 'rgba(148, 163, 184, 0.2)' },
            ].map(item => (
              <span key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{
                  width: 16, height: 3, borderRadius: 2,
                  background: item.color, display: 'inline-block'
                }} />
                {item.label}
              </span>
            ))}
          </div>

          {!graphData ? (
            <div className="empty-state" style={{ background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-subtle)' }}>
              <div className="empty-state-icon">🔗</div>
              <div className="empty-state-title">Graph Not Loaded</div>
              <div className="empty-state-desc">
                {pipelineState.ingested
                  ? 'Click "Load Graph" to visualize the Unified Entity Graph.'
                  : 'Run the pipeline first (at least Generate + Build Graph).'}
              </div>
            </div>
          ) : (
            <div style={{
              border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius)',
              overflow: 'hidden', background: '#f0f4f8', position: 'relative'
            }}>
              {hoveredNode && (
                <div style={{
                  position: 'absolute', top: 12, left: 12, zIndex: 10,
                  background: 'rgba(255, 255, 255, 0.95)', border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)', padding: '12px 16px',
                  fontSize: 12, color: 'var(--text-primary)', minWidth: 200,
                  backdropFilter: 'blur(8px)', boxShadow: 'var(--shadow-card-hover)'
                }}>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, marginBottom: 6, color: 'var(--accent-blue-dark)' }}>
                    {hoveredNode.id}
                  </div>
                  <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>Type: {hoveredNode.entity_type}</div>
                  {hoveredNode.entity_type === 'Account' && (
                    <>
                      <div>Mule: {hoveredNode.is_mule ? '🚨 Yes' : '✅ No'}</div>
                      {hoveredNode.risk_score > 0 && (
                        <div style={{ color: hoveredNode.risk_score >= 0.85 ? 'var(--color-danger)' : 'var(--text-secondary)' }}>
                          Risk: {(hoveredNode.risk_score * 100).toFixed(1)}%
                        </div>
                      )}
                      {hoveredNode.cluster_id && (
                        <div style={{ color: 'var(--color-danger)' }}>Cluster: {hoveredNode.cluster_id}</div>
                      )}
                    </>
                  )}
                </div>
              )}
              <ForceGraph2D
                ref={graphRef}
                graphData={{ nodes: graphData.nodes, links: graphData.links }}
                width={1380}
                height={650}
                backgroundColor="#f0f4f8"
                nodeRelSize={4}
                nodeVal={node => {
                  if (node.entity_type === 'Account') return node.is_mule ? 6 : 3
                  return 2
                }}
                nodeColor={node => {
                  if (node.entity_type === 'Account') {
                    if (node.is_mule) return '#f43f5e'
                    if (node.risk_score >= 0.6) return '#f59e0b'
                    return '#3b82f6'
                  }
                  if (node.entity_type === 'Device') return '#10b981'
                  if (node.entity_type === 'IPAddress') return '#a78bfa'
                  if (node.entity_type === 'ATMTerminal') return '#f59e0b'
                  return '#64748b'
                }}
                nodeCanvasObjectMode={() => 'after'}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  if (node.entity_type !== 'Account' || globalScale < 1.5) return
                  const label = node.id.replace('ACC-', '')
                  const fontSize = 10 / globalScale
                  ctx.font = `${fontSize}px JetBrains Mono, monospace`
                  ctx.textAlign = 'center'
                  ctx.textBaseline = 'middle'
                  ctx.fillStyle = 'rgba(26, 35, 50, 0.7)'
                  ctx.fillText(label, node.x, node.y + 8 / globalScale)
                }}
                linkColor={link => {
                  if (link.is_suspicious) return 'rgba(244, 63, 94, 0.6)'
                  if (link.edge_type === 'TRANSFERRED_TO') return 'rgba(59, 130, 246, 0.3)'
                  return 'rgba(148, 163, 184, 0.1)'
                }}
                linkWidth={link => {
                  if (link.is_suspicious) return 1.5
                  if (link.edge_type === 'TRANSFERRED_TO') return 0.8
                  return 0.3
                }}
                linkDirectionalArrowLength={link => link.edge_type === 'TRANSFERRED_TO' ? 3 : 0}
                linkDirectionalArrowRelPos={1}
                onNodeHover={node => setHoveredNode(node || null)}
                onNodeClick={node => {
                  if (node.entity_type === 'Account') {
                    fetchExplanation(node.id)
                    setActiveTab('xai')
                  }
                }}
                cooldownTicks={100}
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
              />
            </div>
          )}
        </div>
      )}

      {/* ═══ Accounts Tab ═══ */}
      {activeTab === 'accounts' && (
        <div>
          <h2 className="section-title">👤 Account Risk Scores</h2>
          {accounts.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🔍</div>
              <div className="empty-state-title">No Data Available</div>
              <div className="empty-state-desc">Run the pipeline first to generate risk scores.</div>
            </div>
          ) : (
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Account ID</th>
                    <th>Risk Score</th>
                    <th>Action</th>
                    <th>Flagged</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {accounts.slice(0, 50).map(acc => (
                    <tr key={acc.account_id}>
                      <td style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                        {acc.account_id}
                      </td>
                      <td>
                        <div className="risk-bar-container">
                          <span className="risk-score" style={{ color: getRiskColor(acc.mule_probability) }}>
                            {(acc.mule_probability * 100).toFixed(1)}%
                          </span>
                          <div className="risk-bar">
                            <div className={`risk-bar-fill ${getRiskLevel(acc.mule_probability)}`}
                              style={{ width: `${acc.mule_probability * 100}%` }} />
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`risk-badge ${getActionClass(acc.recommended_action)}`}>
                          {acc.recommended_action}
                        </span>
                      </td>
                      <td>{acc.is_flagged ? '🚨' : '✅'}</td>
                      <td>
                        <button className="btn btn-sm btn-secondary"
                          onClick={() => { fetchExplanation(acc.account_id); setActiveTab('xai') }}>
                          🧠 Explain
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ═══ Clusters Tab ═══ */}
      {activeTab === 'clusters' && (
        <div>
          <h2 className="section-title">🕸️ Detected Mule Ring Clusters</h2>
          {clusters.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🕸️</div>
              <div className="empty-state-title">No Clusters Detected</div>
              <div className="empty-state-desc">Run analysis to detect mule ring clusters.</div>
            </div>
          ) : (
            <div className="clusters-grid">
              {clusters.map(cluster => (
                <div key={cluster.cluster_id} className="cluster-card">
                  <div className="cluster-header">
                    <span className="cluster-id">{cluster.cluster_id}</span>
                    <span className="risk-badge escalate">
                      {(cluster.avg_risk_score * 100).toFixed(1)}% Risk
                    </span>
                  </div>
                  <div className="cluster-metrics">
                    <div className="cluster-metric">
                      <span className="cluster-metric-label">Members</span>
                      <span className="cluster-metric-value" style={{ color: 'var(--accent-blue)' }}>
                        {cluster.size}
                      </span>
                    </div>
                    <div className="cluster-metric">
                      <span className="cluster-metric-label">Density</span>
                      <span className="cluster-metric-value" style={{ color: 'var(--accent-purple)' }}>
                        {(cluster.density || 0).toFixed(3)}
                      </span>
                    </div>
                    <div className="cluster-metric">
                      <span className="cluster-metric-label">Volume</span>
                      <span className="cluster-metric-value" style={{ color: 'var(--color-warning)' }}>
                        ₹{((cluster.total_volume || 0) / 1000).toFixed(1)}K
                      </span>
                    </div>
                    <div className="cluster-metric">
                      <span className="cluster-metric-label">Avg Velocity</span>
                      <span className="cluster-metric-value" style={{ color: 'var(--color-danger)' }}>
                        {((cluster.avg_velocity_seconds || 0) / 60).toFixed(1)}m
                      </span>
                    </div>
                  </div>
                  <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-muted)' }}>
                    <strong>Hub:</strong>{' '}
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: 'var(--color-danger)' }}>
                      {cluster.hub_account}
                    </span>
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
                    <strong>Channels:</strong> {(cluster.channels_used || []).join(', ')}
                  </div>
                  <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(cluster.members || []).slice(0, 8).map(m => (
                      <span key={m} style={{
                        padding: '2px 8px', background: '#fef2f2',
                        border: '1px solid #fecaca', borderRadius: 6,
                        fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                        color: '#dc2626', cursor: 'pointer'
                      }}
                        onClick={() => { fetchExplanation(m); setActiveTab('xai') }}>
                        {m}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ═══ XAI Tab ═══ */}
      {activeTab === 'xai' && (
        <div>
          <h2 className="section-title">🧠 Explainable AI Auditor</h2>
          {!explanation ? (
            <div className="empty-state">
              <div className="empty-state-icon">🧠</div>
              <div className="empty-state-title">No Explanation Selected</div>
              <div className="empty-state-desc">
                Click "Explain" on an account from the Accounts tab, or click an account in a cluster.
              </div>
            </div>
          ) : (
            <div>
              <div className="explanation-panel">
                <div className="explanation-header">
                  <span className="explanation-account" style={{ color: 'var(--accent-blue-dark)' }}>
                    {explanation.account_id}
                  </span>
                  <span className="risk-score" style={{
                    color: getRiskColor(explanation.confidence_score),
                    fontSize: 24
                  }}>
                    {(explanation.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>

                <div style={{
                  padding: 16, background: 'var(--bg-tinted)',
                  borderRadius: 'var(--radius-sm)', marginBottom: 20,
                  fontSize: 14, lineHeight: 1.8, color: 'var(--text-secondary)'
                }}>
                  {explanation.xai_reasoning}
                </div>

                <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: 'var(--text-muted)' }}>
                  FEATURE ATTRIBUTIONS
                </h3>
                <div className="feature-bars">
                  {(explanation.feature_attributions || []).map(feat => (
                    <div key={feat.name} className="feature-bar-row">
                      <span className="feature-bar-label">{feat.name.replace(/_/g, ' ')}</span>
                      <div className="feature-bar-track">
                        <div className="feature-bar-fill"
                          style={{ width: `${feat.importance * 100}%` }} />
                      </div>
                      <span className="feature-bar-value">{(feat.importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ═══ Reports Tab ═══ */}
      {activeTab === 'report' && (
        <div>
          <h2 className="section-title">📄 Audit Reports</h2>
          <div style={{ marginBottom: 20 }}>
            <button id="btn-generate-report" className="btn btn-primary" onClick={fetchReport}
              disabled={!pipelineState.analyzed}>
              📄 Generate Full Audit Report
            </button>
          </div>

          {!report ? (
            <div className="empty-state">
              <div className="empty-state-icon">📄</div>
              <div className="empty-state-title">No Report Generated</div>
              <div className="empty-state-desc">
                Run the pipeline and click "Generate Full Audit Report".
              </div>
            </div>
          ) : (
            <div>
              <div className="stats-grid">
                <div className="stat-card cyan">
                  <div className="stat-label">Accounts Analyzed</div>
                  <div className="stat-value">{report.summary?.total_accounts_analyzed || 0}</div>
                </div>
                <div className="stat-card rose">
                  <div className="stat-label">Flagged</div>
                  <div className="stat-value">{report.summary?.flagged_accounts || 0}</div>
                </div>
                <div className="stat-card purple">
                  <div className="stat-label">Clusters</div>
                  <div className="stat-value">{report.summary?.clusters_detected || 0}</div>
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <span className="card-title">📋 Report JSON</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {report.report_id}
                  </span>
                </div>
                <div className="log-output" style={{ maxHeight: 500, fontSize: 11 }}>
                  <pre style={{ color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(report, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <footer style={{
        padding: '24px 0', marginTop: 40,
        borderTop: '1px solid var(--border-subtle)',
        textAlign: 'center', fontSize: 12, color: 'var(--text-muted)'
      }}>
        ChainVigil v1.0.0 — Cross-Channel Mule Detection using Graph Intelligence & GNN
      </footer>
    </div>
  )
}

export default App
