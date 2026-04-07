import { useState, useCallback, useRef, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import LoadingScreen from './LoadingScreen'
import DecryptedText from './DecryptedText'
import GridScan from './GridScan'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE
  || (typeof window !== 'undefined' && window.location.port === '5173'
    ? 'http://127.0.0.1:8000'
    : '')

// ─── Intelligence Tab Component ──────────────────────────────────────────────
// Lives outside App so it can safely use hooks.
function IntelligenceTab({
  clusters, fetchClusters,
  intelAccount, setIntelAccount,
  intelText, setIntelText,
  intelLoading, fetchIntelligence, intelResult,
  nlpText, setNlpText, fetchNLP, nlpResult,
  metricsData, fetchMetrics,
}) {
  // Fetch clusters when this tab first mounts (covers the case where
  // the pipeline ran but clusters state is empty because the user refreshed)
  useEffect(() => {
    if (clusters.length === 0) fetchClusters()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const quickIds = clusters.flatMap(c => c.members || []).slice(0, 6)

  const getRiskColor = (tier) => ({
    CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e'
  })[tier] || '#6b7280'

  return (
    <div>
      {/* ── Input row ── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <div style={{ flex: 1, minWidth: 180 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Account ID</div>
          <input
            id="intel-account-input"
            type="text"
            placeholder="e.g. ACC-00042"
            value={intelAccount}
            onChange={e => setIntelAccount(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && fetchIntelligence()}
            style={{
              width: '100%', padding: '10px 14px',
              background: 'var(--bg-tinted)', border: '1px solid var(--border-subtle)',
              borderRadius: 8, color: 'var(--text-primary)', fontSize: 14,
              fontFamily: "'JetBrains Mono', monospace", boxSizing: 'border-box'
            }}
          />
        </div>
        <div style={{ flex: 2, minWidth: 240 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Transaction Note (optional — NLP)</div>
          <input
            id="intel-text-input"
            type="text"
            placeholder="e.g. urgent advance fee split payment"
            value={intelText}
            onChange={e => setIntelText(e.target.value)}
            style={{
              width: '100%', padding: '10px 14px',
              background: 'var(--bg-tinted)', border: '1px solid var(--border-subtle)',
              borderRadius: 8, color: 'var(--text-primary)', fontSize: 14,
              boxSizing: 'border-box'
            }}
          />
        </div>
        <button
          id="btn-run-intelligence"
          className="btn btn-primary"
          onClick={fetchIntelligence}
          disabled={intelLoading || !intelAccount.trim()}
        >
          {intelLoading ? <span className="spinner" /> : '🤖'} Analyze
        </button>
      </div>

      {/* ── Quick-fill badges from cluster members ── */}
      {quickIds.length > 0 && (
        <div style={{ marginBottom: 20, padding: '10px 14px', background: 'var(--bg-tinted)', borderRadius: 8, fontSize: 12, color: 'var(--text-muted)' }}>
          💡 Click an account to auto-fill:{' '}
          {quickIds.map(id => (
            <button
              key={id}
              onClick={() => setIntelAccount(id)}
              style={{
                marginLeft: 6, padding: '2px 10px',
                background: intelAccount === id ? '#dc2626' : '#fef2f2',
                border: '1px solid #fecaca', borderRadius: 6,
                fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                color: intelAccount === id ? '#fff' : '#dc2626',
                cursor: 'pointer', transition: 'all .15s'
              }}
            >{id}</button>
          ))}
        </div>
      )}
      {quickIds.length === 0 && (
        <div style={{ marginBottom: 16, padding: '10px 14px', background: 'var(--bg-tinted)', borderRadius: 8, fontSize: 12, color: 'var(--text-muted)' }}>
          ⚠️ No cluster data yet — run the pipeline first, then come back here.
        </div>
      )}

      {/* ── Results ── */}
      {intelResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Score banner */}
          <div style={{
            padding: '20px 24px', borderRadius: 10,
            background: `${getRiskColor(intelResult.root_cause?.risk_tier)}18`,
            border: `1px solid ${getRiskColor(intelResult.root_cause?.risk_tier)}`,
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12
          }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>Account</div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-primary)' }}>
                {intelResult.account_id}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                GNN Score: <strong>{((intelResult.gnn_score || 0) * 100).toFixed(1)}%</strong>
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>Final Risk Score</div>
              <div style={{ fontSize: 36, fontWeight: 800, color: getRiskColor(intelResult.root_cause?.risk_tier) }}>
                {(((intelResult.root_cause?.final_risk_score) || 0) * 100).toFixed(1)}%
              </div>
              <span style={{
                padding: '4px 14px', borderRadius: 20, fontSize: 12, fontWeight: 700,
                background: getRiskColor(intelResult.root_cause?.risk_tier), color: 'white'
              }}>{intelResult.root_cause?.risk_tier || '—'}</span>
            </div>
          </div>

          {/* 3-column module scores */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
            {/* Temporal */}
            <div className="stat-card" style={{ borderTop: '3px solid #6366f1' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>⏱ TEMPORAL</div>
              <div className="stat-value" style={{ color: '#6366f1', fontSize: 26 }}>
                {(((intelResult.temporal?.temporal_risk_score) || 0) * 100).toFixed(0)}%
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: intelResult.temporal?.burst_detected ? '#ef4444' : '#22c55e' }}>
                {intelResult.temporal?.burst_detected ? '🔴 Burst Detected' : '🟢 No Burst'}
              </div>
              <div style={{ fontSize: 12, marginTop: 4, color: intelResult.temporal?.rapid_relay_detected ? '#ef4444' : '#22c55e' }}>
                {intelResult.temporal?.rapid_relay_detected ? '🔴 Rapid Relay' : '🟢 No Relay'}
              </div>
              {(intelResult.temporal?.temporal_signals || []).map((s, i) => (
                <div key={i} style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>↳ {s}</div>
              ))}
            </div>

            {/* Behavioral */}
            <div className="stat-card" style={{ borderTop: '3px solid #10b981' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>🧬 BEHAVIORAL</div>
              <div className="stat-value" style={{ color: '#10b981', fontSize: 26 }}>
                {(((intelResult.behavioral?.behavioral_risk_score) || 0) * 100).toFixed(0)}%
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: intelResult.behavioral?.dormancy_reactivation ? '#ef4444' : '#22c55e' }}>
                {intelResult.behavioral?.dormancy_reactivation ? '🔴 Dormancy Signal' : '🟢 No Dormancy'}
              </div>
              <div style={{ fontSize: 12, marginTop: 4, color: intelResult.behavioral?.odd_hour_activity ? '#f97316' : '#22c55e' }}>
                {intelResult.behavioral?.odd_hour_activity ? '🟠 Odd Hours' : '🟢 Normal Hours'}
              </div>
              <div style={{ fontSize: 12, marginTop: 4, color: intelResult.behavioral?.device_ip_switching ? '#f97316' : '#22c55e' }}>
                {intelResult.behavioral?.device_ip_switching ? '🟠 Device Switching' : '🟢 Stable Device'}
              </div>
            </div>

            {/* NLP */}
            <div className="stat-card" style={{ borderTop: '3px solid #f59e0b' }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>🔤 NLP</div>
              {intelResult.nlp ? (
                <>
                  <div className="stat-value" style={{ color: intelResult.nlp.is_suspicious ? '#ef4444' : '#22c55e', fontSize: 26 }}>
                    {(((intelResult.nlp?.nlp_risk_score) || 0) * 100).toFixed(0)}%
                  </div>
                  <div style={{ marginTop: 8, fontSize: 12, color: intelResult.nlp.is_suspicious ? '#ef4444' : '#22c55e' }}>
                    {intelResult.nlp.is_suspicious ? '🔴 Suspicious Text' : '🟢 Clean Text'}
                  </div>
                  <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(intelResult.nlp.matched_patterns || []).map((p, i) => (
                      <span key={i} style={{ padding: '2px 8px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 12, fontSize: 10, color: '#dc2626' }}>{p}</span>
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>Add a transaction note above to enable NLP detection</div>
              )}
            </div>
          </div>

          {/* Root Cause */}
          <div style={{ background: 'var(--bg-tinted)', borderRadius: 10, padding: '16px 20px' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>📋 Root Cause Analysis</div>
            {(intelResult.root_cause?.explanation || []).length > 0 ? (
              <ul style={{ margin: 0, padding: '0 0 0 18px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                {intelResult.root_cause.explanation.map((bullet, i) => (
                  <li key={i} style={{ fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.6 }}>{bullet}</li>
                ))}
              </ul>
            ) : (
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No significant signals detected — account appears clean.</div>
            )}
          </div>

          {/* Decision */}
          {intelResult.decision && (
            <div style={{
              borderRadius: 10, padding: '16px 20px',
              background: `${getRiskColor(intelResult.root_cause?.risk_tier)}0d`,
              border: `1px solid ${getRiskColor(intelResult.root_cause?.risk_tier)}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>⚡ Automated Decision</div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <span style={{
                    padding: '6px 20px', borderRadius: 20, fontSize: 14, fontWeight: 800, letterSpacing: 1,
                    background: getRiskColor(intelResult.root_cause?.risk_tier), color: 'white'
                  }}>{intelResult.decision.action}</span>
                  {intelResult.decision.sla_hours > 0 && (
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>SLA: {intelResult.decision.sla_hours}h</span>
                  )}
                </div>
              </div>
              {(intelResult.decision.playbook || []).length > 0 && (
                <>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Playbook</div>
                  <ol style={{ margin: 0, padding: '0 0 0 18px', display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {intelResult.decision.playbook.map((step, i) => (
                      <li key={i} style={{ fontSize: 13, color: 'var(--text-primary)' }}>{step}</li>
                    ))}
                  </ol>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Standalone NLP Scanner ── */}
      <div style={{ marginTop: 32, paddingTop: 24, borderTop: '1px solid var(--border-subtle)' }}>
        <h3 style={{ fontSize: 16, marginBottom: 8, color: 'var(--text-primary)' }}>🔤 NLP Fraud Text Scanner</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 12 }}>Scan any transaction note independently for fraud language.</p>
        <div style={{ display: 'flex', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
          <input
            id="nlp-standalone-input"
            type="text"
            value={nlpText}
            onChange={e => setNlpText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && fetchNLP()}
            placeholder="Type any transaction description..."
            style={{
              flex: 1, minWidth: 280, padding: '10px 14px',
              background: 'var(--bg-tinted)', border: '1px solid var(--border-subtle)',
              borderRadius: 8, color: 'var(--text-primary)', fontSize: 14
            }}
          />
          <button id="btn-scan-nlp" className="btn btn-primary" onClick={fetchNLP}>🔍 Scan</button>
        </div>
        {/* Preset examples */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
          {[
            'urgent advance fee split payment avoid tax',
            'monthly rent for flat 4B',
            'mule shell company nominee transfer',
            'easy money work from home commission',
          ].map(s => (
            <button key={s} onClick={() => setNlpText(s)} style={{
              padding: '3px 12px', background: 'var(--bg-tinted)',
              border: '1px solid var(--border-subtle)', borderRadius: 20,
              fontSize: 11, color: 'var(--text-secondary)', cursor: 'pointer'
            }}>{s.slice(0, 32)}…</button>
          ))}
        </div>
        {nlpResult && (
          <div style={{
            padding: '16px 20px', borderRadius: 10,
            background: nlpResult.is_suspicious ? 'rgba(239,68,68,0.08)' : 'rgba(34,197,94,0.08)',
            border: `1px solid ${nlpResult.is_suspicious ? '#ef4444' : '#22c55e'}`
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: nlpResult.is_suspicious ? '#ef4444' : '#22c55e' }}>
                {nlpResult.is_suspicious ? '🔴 SUSPICIOUS TEXT DETECTED' : '🟢 CLEAN — No fraud patterns found'}
              </div>
              <div style={{ fontSize: 28, fontWeight: 800, color: nlpResult.is_suspicious ? '#ef4444' : '#22c55e' }}>
                {(((nlpResult.nlp_risk_score) || 0) * 100).toFixed(0)}%
              </div>
            </div>
            {(nlpResult.matched_patterns || []).length > 0 && (
              <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {nlpResult.matched_patterns.map((p, i) => (
                  <span key={i} style={{ padding: '3px 12px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 20, fontSize: 12, color: '#dc2626', fontWeight: 600 }}>{p}</span>
                ))}
              </div>
            )}
            {(nlpResult.matched_terms || []).length > 0 && (
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-muted)' }}>
                Matched: <strong style={{ color: 'var(--text-primary)' }}>{nlpResult.matched_terms.join(', ')}</strong>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Observability Metrics ── */}
      <div style={{ marginTop: 32, paddingTop: 24, borderTop: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: 16, margin: 0, color: 'var(--text-primary)' }}>📊 System Observability Metrics</h3>
          <button id="btn-refresh-metrics" className="btn btn-secondary" onClick={fetchMetrics}>🔄 Refresh</button>
        </div>
        {!metricsData ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>Click Refresh to load live metrics.</div>
        ) : (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
              Uptime: {(metricsData.uptime_seconds || 0).toFixed(0)}s &nbsp;·&nbsp; {(metricsData.generated_at || '').slice(0, 19)}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 12, marginBottom: 16 }}>
              {[
                { label: 'Fraud Rate', value: (((metricsData.gauges?.fraud_rate) || 0) * 100).toFixed(1) + '%', color: '#ef4444' },
                { label: 'Model AUC', value: ((metricsData.gauges?.model_auc) || 0).toFixed(4), color: '#6366f1' },
                { label: 'Flagged Accounts', value: metricsData.gauges?.flagged_accounts || 0, color: '#f97316' },
                { label: 'Clusters', value: metricsData.gauges?.clusters_detected || 0, color: '#a78bfa' },
                { label: 'Total Predictions', value: metricsData.counters?.total_predictions || 0, color: '#10b981' },
                { label: 'Intelligence Calls', value: metricsData.counters?.intelligence_queries || 0, color: '#3b82f6' },
                { label: 'NLP Scans', value: metricsData.counters?.nlp_analyses || 0, color: '#f59e0b' },
              ].map(m => (
                <div key={m.label} className="stat-card">
                  <div className="stat-value" style={{ color: m.color, fontSize: 20 }}>{m.value}</div>
                  <div className="stat-label">{m.label}</div>
                </div>
              ))}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
              {Object.entries(metricsData.latency_ms || {}).map(([scope, stats]) => (
                <div key={scope} style={{ background: 'var(--bg-tinted)', borderRadius: 8, padding: '12px 16px' }}>
                  <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: 'var(--text-muted)', marginBottom: 6 }}>{scope} latency ms</div>
                  <div style={{ fontSize: 13 }}>p50: <strong>{stats.p50}</strong></div>
                  <div style={{ fontSize: 13 }}>p95: <strong style={{ color: '#f97316' }}>{stats.p95}</strong></div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>samples: {stats.count}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

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
  const [graphRef] = [useRef(null)]
  const graphRefActual = useRef(null)
  const logRef = useRef(null)

  const [sanctions, setSanctions] = useState(null)
  const [sarReport, setSarReport] = useState(null)
  const [liveEvents, setLiveEvents] = useState([])
  const [liveConnected, setLiveConnected] = useState(false)
  const eventSourceRef = useRef(null)

  // ── Intelligence state (Steps 2–6) ──────────────────────────
  const [intelResult, setIntelResult] = useState(null)
  const [intelAccount, setIntelAccount] = useState('')
  const [intelText, setIntelText] = useState('')
  const [intelLoading, setIntelLoading] = useState(false)
  const [metricsData, setMetricsData] = useState(null)
  const [nlpText, setNlpText] = useState('urgent advance fee split payment avoid tax')
  const [nlpResult, setNlpResult] = useState(null)


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
      const apiUrl = `${API_BASE}${endpoint}`
      const res = await fetch(apiUrl, opts)
      if (!res.ok) {
        let detail = 'API Error'
        try {
          const err = await res.json()
          detail = err.detail || err.message || JSON.stringify(err)
        } catch {
          try {
            const txt = await res.text()
            if (txt) detail = txt.slice(0, 180)
          } catch {
            // keep default detail
          }
        }
        throw new Error(`${res.status} ${detail}`)
      }
      return await res.json()
    } catch (e) {
      const msg = e?.message || 'Unknown error'
      addLog(`Error: ${msg}`, 'error')
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



  const fetchSanctions = async () => {
    addLog('Running behaviour-based sanctions screening...', 'info')
    try {
      const res = await apiCall('/api/sanctions/summary', 'GET')
      setSanctions(res)
      addLog(`✅ Sanctions screening complete: ${res.total_alerts} alerts`, 'success')
    } catch { /* */ }
  }

  const fetchSAR = async () => {
    addLog('Generating FIU-IND SAR report...', 'info')
    try {
      const res = await apiCall('/api/report/sar', 'GET')
      setSarReport(res)
      addLog('✅ SAR report generated', 'success')
    } catch { /* */ }
  }

  const fetchIntelligence = async () => {
    if (!intelAccount.trim()) return
    setIntelLoading(true)
    addLog(`🧠 Running intelligence analysis for ${intelAccount}...`, 'info')
    try {
      const textParam = intelText.trim() ? `?text=${encodeURIComponent(intelText.trim())}` : ''
      const res = await apiCall(`/api/intelligence/analyze/${intelAccount.trim()}${textParam}`, 'GET')
      setIntelResult(res)
      addLog(`✅ Intelligence analysis complete for ${intelAccount}`, 'success')
    } catch { /* */ } finally { setIntelLoading(false) }
  }

  const fetchMetrics = async () => {
    try {
      const res = await apiCall('/metrics', 'GET')
      setMetricsData(res)
    } catch { /* */ }
  }

  const fetchNLP = async () => {
    try {
      const res = await apiCall('/api/intelligence/nlp', 'POST', { text: nlpText, tx_id: 'DEMO-NLP' })
      setNlpResult(res)
    } catch { /* */ }
  }

  const startLiveFeed = () => {
    if (eventSourceRef.current) return
    const es = new EventSource(`${API_BASE}/api/stream/live`)
    es.onopen = () => setLiveConnected(true)
    es.onmessage = (e) => {
      try {
        const tx = JSON.parse(e.data)
        setLiveEvents(prev => [tx, ...prev].slice(0, 80))
      } catch { }
    }
    es.onerror = () => { setLiveConnected(false) }
    eventSourceRef.current = es
  }

  const stopLiveFeed = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setLiveConnected(false)
    }
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
    <div className="flex min-h-screen text-on-surface font-body bg-surface selection:bg-primary-fixed selection:text-on-primary-fixed">
      <LoadingScreen />

      {/* Sidebar Navigation */}
      <aside className="w-64 bg-surface-container-low border-r border-outline-variant flex flex-col fixed h-full z-10 transition-transform">
        {/* Brand */}
        <div className="p-6 border-b border-outline-variant flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary-fixed flex items-center justify-center shadow-md">
            <span className="material-symbols-outlined text-on-primary">link</span>
          </div>
          <div>
            <h2 className="font-headline font-bold text-xl tracking-tight text-on-surface">ChainVigil</h2>
            <p className="text-[10px] uppercase font-bold tracking-widest text-primary mt-0.5">Terminal</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-6 px-4 space-y-2">
          {[
            { id: 'pipeline', icon: 'account_tree', label: 'Pipeline Engine' },
            { id: 'graph', icon: 'hub', label: 'Entity Graph' },
            { id: 'accounts', icon: 'person_search', label: 'Account Risk' },
            { id: 'clusters', icon: 'group_work', label: 'Mule Clusters' },
            { id: 'livefeed', icon: 'dynamic_feed', label: 'Live Feed' },
            { id: 'xai', icon: 'psychology', label: 'XAI Auditor' },
            { id: 'sar', icon: 'assessment', label: 'SAR Automation' },
            { id: 'intelligence', icon: 'insights', label: 'Intelligence' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeTab === tab.id
                ? 'bg-primary/10 text-primary font-bold border border-primary/20'
                : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface font-medium'
                }`}
            >
              <span className="material-symbols-outlined">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        {/* System Status Footer */}
        <div className="p-4 m-4 rounded-xl bg-surface-container border border-outline-variant/50">
          <div className="flex items-center gap-2 mb-2">
            <div className="relative flex h-3 w-3">
              {pipelineState.analyzed ? (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-fixed opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-primary"></span>
                </>
              ) : (
                <span className="relative inline-flex rounded-full h-3 w-3 bg-outline"></span>
              )}
            </div>
            <span className="text-xs font-bold text-on-surface uppercase tracking-wider">
              {pipelineState.analyzed ? 'System Active' : 'System Idle'}
            </span>
          </div>
          <p className="text-[10px] text-on-surface-variant leading-tight">Connected to Deep Graph Backend. Ready for analysis.</p>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 ml-64 p-8 grid-bg relative">
        {activeTab === 'pipeline' && (
          <div style={{ position: 'fixed', left: '16rem', top: 0, bottom: 0, right: 0, pointerEvents: 'none', zIndex: 0, opacity: 0.6 }}>
            <GridScan
              sensitivity={0.55}
              lineThickness={0.5}
              linesColor="#000000"
              gridScale={0.1}
              scanColor="#000000"
              scanOpacity={0.15}
              enablePost={false}
              bloomIntensity={0}
              chromaticAberration={0}
              noiseIntensity={0.02}
            />
          </div>
        )}

        <div className="relative z-10">
          {activeTab === 'pipeline' && (
            <div className="max-w-7xl mx-auto">
              {/* Top System Bar */}
              <header className="flex justify-between items-center mb-8 glass-card rounded-2xl p-4 border border-outline-variant shadow-sm backdrop-blur-md">
                <div>
                  <h1 className="font-headline text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-fixed">
                    <DecryptedText
                      text="Pipeline Engine"
                      speed={150}
                      maxIterations={15}
                      animateOn="view"
                      revealDirection="start"
                      sequential={true}
                    />
                  </h1>
                  <p className="text-on-surface-variant font-medium mt-1">
                    <DecryptedText
                      text="Cross-Channel Data Synthesis & GNN Training"
                      speed={120}
                      maxIterations={10}
                      animateOn="view"
                      revealDirection="start"
                      sequential={true}
                    />
                  </p>
                </div>
                <div className="flex items-center gap-4 bg-surface-container-low px-4 py-2 rounded-xl border border-outline-variant/30">
                  <div className="text-right">
                    <div className="text-xs text-on-surface-variant font-bold uppercase tracking-wider">Backend Status</div>
                    <div className="text-sm font-mono font-medium text-primary">Connected</div>
                  </div>
                  <span className="material-symbols-outlined text-primary text-3xl">dns</span>
                </div>
              </header>

              {/* Global Action Bar */}
              <div className="flex gap-4 mb-8">
                <button
                  onClick={runFullPipeline}
                  disabled={loading}
                  className="bg-primary text-on-primary hover:bg-primary-fixed hover:text-on-primary-fixed font-bold py-4 px-8 rounded-2xl shadow-lg transition-all flex items-center gap-3 text-lg border border-transparent disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  {loading && <span className="material-symbols-outlined animate-spin">refresh</span>}
                  {loading ? 'Running...' : 'Run Full Pipeline'}
                </button>
              </div>

              {/* Pipeline Stages (Bento Grid) */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {/* Stage 1: Generate Data */}
                <div className={`glass-card rounded-3xl p-6 border ${pipelineState.generated ? 'border-primary shadow-primary/20' : 'border-outline-variant'} shadow-sm relative overflow-hidden group hover:border-primary transition-colors flex flex-col justify-between h-56`}>
                  <div className={`absolute -right-10 -top-10 w-32 h-32 rounded-full blur-2xl transition-all ${pipelineState.generated ? 'bg-primary/20' : 'bg-primary/5 group-hover:bg-primary/10'}`}></div>
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <span className={`material-symbols-outlined text-3xl ${pipelineState.generated ? 'text-primary' : 'text-on-surface-variant'}`}>data_object</span>
                      <span className="bg-surface-variant text-on-surface-variant text-xs font-bold px-2 py-1 rounded-full uppercase tracking-wider">
                        {pipelineState.generated ? 'Done' : 'Stage 1'}
                      </span>
                    </div>
                    <h3 className="font-headline text-xl font-bold mb-1">Data Synthesis</h3>
                    <p className="text-sm text-on-surface-variant">Synthetic transactions</p>
                  </div>
                  <button
                    onClick={runGenerate}
                    disabled={loading}
                    className="mt-4 w-full bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors border border-outline-variant disabled:opacity-50 cursor-pointer"
                  >
                    {pipelineState.generated ? <span className="material-symbols-outlined text-lg text-primary">check_circle</span> : <span className="material-symbols-outlined text-lg">play_arrow</span>}
                    Generate Data
                  </button>
                </div>

                {/* Stage 2: Build Graph */}
                <div className={`glass-card rounded-3xl p-6 border ${pipelineState.ingested ? 'border-primary shadow-primary/20' : 'border-outline-variant'} shadow-sm relative overflow-hidden group hover:border-primary transition-colors flex flex-col justify-between h-56`}>
                  <div className={`absolute -right-10 -top-10 w-32 h-32 rounded-full blur-2xl transition-all ${pipelineState.ingested ? 'bg-primary/20' : 'bg-primary/5 group-hover:bg-primary/10'}`}></div>
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <span className={`material-symbols-outlined text-3xl ${pipelineState.ingested ? 'text-primary' : 'text-on-surface-variant'}`}>account_tree</span>
                      <span className="bg-surface-variant text-on-surface-variant text-xs font-bold px-2 py-1 rounded-full uppercase tracking-wider">
                        {pipelineState.ingested ? 'Done' : 'Stage 2'}
                      </span>
                    </div>
                    <h3 className="font-headline text-xl font-bold mb-1">Build Graph</h3>
                    <p className="text-sm text-on-surface-variant">Unified entity graph</p>
                  </div>
                  <button
                    onClick={runIngest}
                    disabled={loading || !pipelineState.generated}
                    className="mt-4 w-full bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors border border-outline-variant disabled:opacity-50 cursor-pointer"
                  >
                    {pipelineState.ingested ? <span className="material-symbols-outlined text-lg text-primary">check_circle</span> : <span className="material-symbols-outlined text-lg">play_arrow</span>}
                    Build Graph
                  </button>
                </div>

                {/* Stage 3: Train GNN */}
                <div className={`glass-card rounded-3xl p-6 border ${pipelineState.trained ? 'border-primary shadow-primary/20' : 'border-outline-variant'} shadow-sm relative overflow-hidden group hover:border-primary transition-colors flex flex-col justify-between h-56`}>
                  <div className={`absolute -right-10 -top-10 w-32 h-32 rounded-full blur-2xl transition-all ${pipelineState.trained ? 'bg-primary/20' : 'bg-primary/5 group-hover:bg-primary/10'}`}></div>
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <span className={`material-symbols-outlined text-3xl ${pipelineState.trained ? 'text-primary' : 'text-on-surface-variant'}`}>model_training</span>
                      <span className="bg-surface-variant text-on-surface-variant text-xs font-bold px-2 py-1 rounded-full uppercase tracking-wider">
                        {pipelineState.trained ? 'Done' : 'Stage 3'}
                      </span>
                    </div>
                    <h3 className="font-headline text-xl font-bold mb-1">Train Model</h3>
                    <p className="text-sm text-on-surface-variant">Train GraphSAGE + GAT embeddings.</p>
                  </div>
                  <button
                    onClick={runTrain}
                    disabled={loading || !pipelineState.ingested}
                    className="mt-4 w-full bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors border border-outline-variant disabled:opacity-50 cursor-pointer"
                  >
                    {pipelineState.trained ? <span className="material-symbols-outlined text-lg text-primary">check_circle</span> : <span className="material-symbols-outlined text-lg">play_arrow</span>}
                    Train Model
                  </button>
                </div>

                {/* Stage 4: Analyze Risk */}
                <div className={`glass-card rounded-3xl p-6 border ${pipelineState.analyzed ? 'border-primary shadow-primary/20' : 'border-outline-variant'} shadow-sm relative overflow-hidden group hover:border-primary transition-colors flex flex-col justify-between h-56`}>
                  <div className={`absolute -right-10 -top-10 w-32 h-32 rounded-full blur-2xl transition-all ${pipelineState.analyzed ? 'bg-primary/20' : 'bg-primary/5 group-hover:bg-primary/10'}`}></div>
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <span className={`material-symbols-outlined text-3xl ${pipelineState.analyzed ? 'text-primary' : 'text-on-surface-variant'}`}>radar</span>
                      <span className="bg-surface-variant text-on-surface-variant text-xs font-bold px-2 py-1 rounded-full uppercase tracking-wider">
                        {pipelineState.analyzed ? 'Done' : 'Stage 4'}
                      </span>
                    </div>
                    <h3 className="font-headline text-xl font-bold mb-1">Analyze Risk</h3>
                    <p className="text-sm text-on-surface-variant">Detect clusters and score entities.</p>
                  </div>
                  <button
                    onClick={runAnalyze}
                    disabled={loading || !pipelineState.trained}
                    className="mt-4 w-full bg-surface-container-high hover:bg-surface-container-highest text-on-surface font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors border border-outline-variant disabled:opacity-50 cursor-pointer"
                  >
                    {pipelineState.analyzed ? <span className="material-symbols-outlined text-lg text-primary">check_circle</span> : <span className="material-symbols-outlined text-lg">play_arrow</span>}
                    Analyze Risk
                  </button>
                </div>
              </div>

              {/* Analysis Results */}
              {(stats || trainingResults || riskDist) && (
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">

                  {/* Risk Distribution */}
                  <div className="glass-card rounded-2xl p-6 border border-outline-variant shadow-sm flex flex-col justify-between lg:col-span-3">
                    <h3 className="font-headline font-semibold text-lg mb-4 text-on-surface flex items-center gap-2">
                      <span className="material-symbols-outlined text-secondary">balance</span>
                      Risk Categories
                    </h3>
                    <div className="flex justify-between items-end h-full mt-4 gap-4">
                      {(() => {
                        if (!riskDist) {
                          return (
                            <>
                              <div className="w-full flex flex-col items-center gap-2">
                                <div className="w-full bg-error/20 rounded-t-lg h-24 relative border-b-2 border-error"></div>
                                <div className="text-xs font-mono text-on-surface-variant uppercase">ESCALATE</div>
                              </div>
                              <div className="w-full flex flex-col items-center gap-2">
                                <div className="w-full bg-tertiary-container/50 rounded-t-lg h-16 relative border-b-2 border-tertiary"></div>
                                <div className="text-xs font-mono text-on-surface-variant uppercase">FREEZE</div>
                              </div>
                              <div className="w-full flex flex-col items-center gap-2">
                                <div className="w-full bg-primary-container/50 rounded-t-lg h-10 relative border-b-2 border-primary"></div>
                                <div className="text-xs font-mono text-on-surface-variant uppercase">MONITOR</div>
                              </div>
                              <div className="w-full flex flex-col items-center gap-2">
                                <div className="w-full bg-surface-container-highest rounded-t-lg h-10 relative border-b-2 border-outline"></div>
                                <div className="text-xs font-mono text-on-surface-variant uppercase">CLEAR</div>
                              </div>
                            </>
                          );
                        }
                        const maxCount = Math.max(riskDist.escalate || 0, riskDist.freeze || 0, riskDist.monitor || 0, riskDist.clear || 0, 1);
                        return (
                          <>
                            <div className="w-full flex flex-col items-center gap-2">
                              <div className="text-lg font-mono text-error font-bold">{riskDist.escalate || 0}</div>
                              <div className="w-full bg-error/20 rounded-t-lg relative border-b-2 border-error transition-all" style={{ height: `${Math.max((riskDist.escalate / maxCount) * 120, 10)}px` }}></div>
                              <div className="text-xs font-mono text-on-surface-variant uppercase">ESCALATE</div>
                            </div>
                            <div className="w-full flex flex-col items-center gap-2">
                              <div className="text-lg font-mono text-tertiary font-bold">{riskDist.freeze || 0}</div>
                              <div className="w-full bg-tertiary-container/50 rounded-t-lg relative border-b-2 border-tertiary transition-all" style={{ height: `${Math.max((riskDist.freeze / maxCount) * 120, 10)}px` }}></div>
                              <div className="text-xs font-mono text-on-surface-variant uppercase">FREEZE</div>
                            </div>
                            <div className="w-full flex flex-col items-center gap-2">
                              <div className="text-lg font-mono text-primary font-bold">{riskDist.monitor || 0}</div>
                              <div className="w-full bg-primary-container/50 rounded-t-lg relative border-b-2 border-primary transition-all" style={{ height: `${Math.max((riskDist.monitor / maxCount) * 120, 10)}px` }}></div>
                              <div className="text-xs font-mono text-on-surface-variant uppercase">MONITOR</div>
                            </div>
                            <div className="w-full flex flex-col items-center gap-2">
                              <div className="text-lg font-mono text-secondary font-bold">{riskDist.clear || 0}</div>
                              <div className="w-full bg-surface-container-highest rounded-t-lg relative border-b-2 border-outline transition-all" style={{ height: `${Math.max((riskDist.clear / maxCount) * 120, 10)}px` }}></div>
                              <div className="text-xs font-mono text-on-surface-variant uppercase">CLEAR</div>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  </div>

                  {/* Training Metrics */}
                  <div className="glass-card rounded-2xl p-6 border border-outline-variant shadow-sm flex flex-col justify-between lg:col-span-1">
                    <h3 className="font-headline font-semibold text-lg mb-4 text-on-surface flex items-center gap-2">
                      <span className="material-symbols-outlined text-tertiary">monitoring</span>
                      Performance
                    </h3>
                    <div className="flex flex-col gap-3">
                      <div className="bg-surface-container-low p-3 rounded-xl border border-outline-variant/50 flex justify-between items-center">
                        <div className="text-xs text-on-surface-variant font-mono">AUC-ROC</div>
                        <div className="text-lg font-bold text-tertiary">
                          {trainingResults ? (trainingResults.best_val_auc || 0).toFixed(4) : '--'}
                        </div>
                      </div>
                      <div className="bg-surface-container-low p-3 rounded-xl border border-outline-variant/50 flex justify-between items-center">
                        <div className="text-xs text-on-surface-variant font-mono">TEST F1</div>
                        <div className="text-lg font-bold text-tertiary">
                          {trainingResults ? (trainingResults.test_metrics?.f1 || 0).toFixed(4) : '--'}
                        </div>
                      </div>
                      {trainingResults && (
                        <>
                          <div className="bg-surface-container-low p-3 rounded-xl border border-outline-variant/50 flex justify-between items-center">
                            <div className="text-xs text-on-surface-variant font-mono">PRECISION</div>
                            <div className="text-lg font-bold text-tertiary">{(trainingResults.test_metrics?.precision || 0).toFixed(4)}</div>
                          </div>
                          <div className="bg-surface-container-low p-3 rounded-xl border border-outline-variant/50 flex justify-between items-center">
                            <div className="text-xs text-on-surface-variant font-mono">RECALL</div>
                            <div className="text-lg font-bold text-tertiary">{(trainingResults.test_metrics?.recall || 0).toFixed(4)}</div>
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Graph Metrics */}
                  <div className="glass-card rounded-2xl p-6 border border-outline-variant shadow-sm flex flex-col justify-between lg:col-span-1">
                    <h3 className="font-headline font-semibold text-lg mb-4 text-on-surface flex items-center gap-2">
                      <span className="material-symbols-outlined text-primary">hub</span>
                      Graph Topology
                    </h3>
                    <div className="flex flex-col gap-4">
                      <div className="bg-surface-container-low p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-center">
                        <div className="text-sm text-on-surface-variant font-mono mb-1">NODES</div>
                        <div className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-fixed">
                          {stats ? (stats.nx_nodes || 0).toLocaleString() : '--'}
                        </div>
                      </div>
                      <div className="bg-surface-container-low p-4 rounded-xl border border-outline-variant/50 flex flex-col justify-center">
                        <div className="text-sm text-on-surface-variant font-mono mb-1">EDGES</div>
                        <div className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary-fixed">
                          {stats ? (stats.nx_edges || 0).toLocaleString() : '--'}
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              )}

              {/* Terminal Log */}
              <div className="rounded-2xl overflow-hidden shadow-xl border border-outline-variant">
                <div className="bg-inverse-surface text-inverse-on-surface px-4 py-3 flex justify-between items-center border-b border-surface-variant/20">
                  <div className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-sm text-primary-fixed">terminal</span>
                    <span className="font-mono text-sm tracking-wider">system_log.stdout</span>
                  </div>
                  <div className="flex gap-4 items-center">
                    <button onClick={() => setLogs([])} className="text-xs text-inverse-on-surface/60 hover:text-white transition-colors">Clear</button>
                    <div className="flex gap-2">
                      <div className="w-3 h-3 rounded-full bg-surface-variant/30"></div>
                      <div className="w-3 h-3 rounded-full bg-surface-variant/30"></div>
                      <div className="w-3 h-3 rounded-full bg-surface-variant/30"></div>
                    </div>
                  </div>
                </div>
                <div className="bg-[#1e1e1e] p-6 h-64 overflow-y-auto font-mono text-sm text-gray-300 leading-relaxed custom-scrollbar" ref={logRef}>
                  {logs.length === 0 && (
                    <div className="flex text-primary-fixed mb-2">
                      <span className="mr-2">❯</span>
                      <span>System initialized. Waiting for pipeline execution...</span>
                    </div>
                  )}
                  {logs.map((log, i) => (
                    <div key={i} className={`flex mb-1 ${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-green-400' : 'text-gray-400'}`}>
                      <span className="mr-2 text-primary-fixed">❯</span>
                      <span>{log.msg}</span>
                    </div>
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

                    {explanation.plain_english_summary && (
                      <div style={{
                        padding: 16,
                        background: '#fff7ed',
                        border: '1px solid #fed7aa',
                        borderRadius: 'var(--radius-sm)',
                        marginBottom: 20,
                      }}>
                        <div style={{
                          fontSize: 11,
                          color: '#9a3412',
                          marginBottom: 8,
                          textTransform: 'uppercase',
                          letterSpacing: 1,
                          fontWeight: 700,
                        }}>
                          Plain-English Summary (LLM)
                        </div>
                        <div style={{ fontSize: 13, color: '#7c2d12', lineHeight: 1.7 }}>
                          {explanation.plain_english_summary}
                        </div>
                        <div style={{ marginTop: 8, fontSize: 11, color: '#9a3412' }}>
                          Model: {explanation.llm_meta?.model || 'unknown'} · Source: {explanation.llm_meta?.source || 'unknown'}
                        </div>
                      </div>
                    )}

                    {(explanation.key_driver_meanings || []).length > 0 && (
                      <div style={{
                        padding: 16,
                        background: '#f8fafc',
                        border: '1px solid #e2e8f0',
                        borderRadius: 'var(--radius-sm)',
                        marginBottom: 20,
                      }}>
                        <div style={{
                          fontSize: 11,
                          color: '#334155',
                          marginBottom: 10,
                          textTransform: 'uppercase',
                          letterSpacing: 1,
                          fontWeight: 700,
                        }}>
                          Key Drivers (What They Mean)
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                          {explanation.key_driver_meanings.map((d, idx) => (
                            <div key={`${d.feature}-${idx}`}>
                              <div style={{ fontSize: 13, color: '#0f172a', fontWeight: 700 }}>
                                {d.feature.replace(/_/g, ' ')} ({((d.importance || 0) * 100).toFixed(1)}%)
                              </div>
                              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
                                {d.meaning}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {(explanation.suggested_actions || []).length > 0 && (
                      <div style={{
                        padding: 16,
                        background: '#f0fdf4',
                        border: '1px solid #86efac',
                        borderRadius: 'var(--radius-sm)',
                        marginBottom: 20,
                      }}>
                        <div style={{
                          fontSize: 11,
                          color: '#166534',
                          marginBottom: 8,
                          textTransform: 'uppercase',
                          letterSpacing: 1,
                          fontWeight: 700,
                        }}>
                          Suggested Actions
                        </div>
                        <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 6 }}>
                          {explanation.suggested_actions.map((action, idx) => (
                            <li key={idx} style={{ fontSize: 13, color: '#166534', lineHeight: 1.6 }}>
                              {action}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

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

          {/* ── Live Feed Tab ── */}
          {activeTab === 'livefeed' && (
            <div className="tab-content">
              <div className="section-card">
                <div className="section-header">
                  <h2 className="section-title">📡 Real-Time Transaction Feed</h2>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: liveConnected ? '#22c55e' : '#6b7280', display: 'inline-block' }} />
                      {liveConnected ? 'LIVE' : 'Disconnected'}
                    </span>
                    {!liveConnected
                      ? <button className="btn btn-primary" onClick={startLiveFeed}>▶ Start Feed</button>
                      : <button className="btn btn-secondary" onClick={stopLiveFeed}>⏹ Stop</button>
                    }
                  </div>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 16 }}>
                  Live UPI/ATM/NEFT/RTGS transactions scored in real-time by the rule engine. Color-coded by risk level.
                </p>
                {liveEvents.length === 0 ? (
                  <div className="empty-state">
                    <div style={{ fontSize: 48 }}>📡</div>
                    <h3>Feed Not Started</h3>
                    <p>Click "Start Feed" to begin streaming live transaction events.</p>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 600, overflowY: 'auto' }}>
                    {liveEvents.map((evt, i) => (
                      <div key={i} style={{
                        display: 'grid', gridTemplateColumns: '60px 1fr 1fr 1fr 90px', gap: 10,
                        alignItems: 'center', padding: '10px 14px', borderRadius: 8,
                        background: evt.severity === 'CRITICAL' ? '#ef444410' : evt.severity === 'HIGH' ? '#f9731610' : evt.severity === 'MEDIUM' ? '#eab30810' : '#22c55e08',
                        border: `1px solid ${evt.severity === 'CRITICAL' ? '#ef444430' : evt.severity === 'HIGH' ? '#f9731630' : evt.severity === 'MEDIUM' ? '#eab30830' : '#22c55e20'}`,
                        fontSize: 12
                      }}>
                        <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)', fontSize: 10 }}>
                          {evt.timestamp?.slice(11, 19)}
                        </span>
                        <span>
                          <span style={{ fontWeight: 600 }}>{evt.sender_id}</span>
                          <span style={{ color: 'var(--text-muted)' }}> → {evt.receiver_id}</span>
                        </span>
                        <span style={{ fontWeight: 700, color: evt.risk_score >= 75 ? '#ef4444' : evt.risk_score >= 45 ? '#f97316' : '#22c55e' }}>
                          ₹{Number(evt.amount).toLocaleString('en-IN')}
                        </span>
                        <span style={{ color: 'var(--text-secondary)' }}>{evt.channel}</span>
                        <span style={{
                          padding: '2px 8px', borderRadius: 12, textAlign: 'center', fontWeight: 700, fontSize: 11,
                          background: evt.severity === 'CRITICAL' ? '#ef4444' : evt.severity === 'HIGH' ? '#f97316' : evt.severity === 'MEDIUM' ? '#eab308' : '#22c55e',
                          color: 'white'
                        }}>
                          {evt.severity}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── SAR Report Tab ── */}
          {activeTab === 'sar' && (
            <div className="tab-content">
              <div className="section-card">
                <div className="section-header">
                  <h2 className="section-title">📋 FIU-IND Suspicious Activity Report</h2>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button className="btn btn-primary" onClick={fetchSAR}>
                      {sarReport ? '🔄 Regenerate SAR' : '📋 Generate SAR'}
                    </button>
                  </div>
                </div>
                {!sarReport ? (
                  <div className="empty-state">
                    <div style={{ fontSize: 48 }}>📋</div>
                    <h3>SAR Not Generated</h3>
                    <p>Run the pipeline first, then click "Generate SAR" to produce a FIU-IND compliant Suspicious Activity Report.</p>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                    {/* Report Header */}
                    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
                        <div>
                          <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 1 }}>Suspicious Activity Report</div>
                          <div style={{ fontSize: 18, fontWeight: 700, color: 'white', marginTop: 4 }}>
                            {sarReport.report_header?.sar_reference_number}
                          </div>
                          <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
                            {sarReport.report_header?.generated_at?.slice(0, 10)} · {sarReport.report_header?.reporting_entity}
                          </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span style={{
                            padding: '6px 16px', borderRadius: 20, fontWeight: 700, fontSize: 13,
                            background: sarReport.executive_summary?.priority_level === 'CRITICAL' ? '#ef4444' : '#f97316',
                            color: 'white'
                          }}>
                            {sarReport.executive_summary?.priority_level}
                          </span>
                          <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>
                            Confidence: {sarReport.executive_summary?.overall_risk_confidence_score}% ({sarReport.executive_summary?.overall_risk_confidence_label})
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Executive Summary */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
                      {[
                        { label: 'Accounts Analyzed', value: sarReport.executive_summary?.total_accounts_analyzed },
                        { label: 'Mule Accounts', value: sarReport.executive_summary?.mule_accounts_flagged, color: '#ef4444' },
                        { label: 'Ring Clusters', value: sarReport.executive_summary?.mule_ring_clusters_detected, color: '#f97316' },
                        { label: 'Crime Patterns', value: sarReport.executive_summary?.financial_crime_patterns_detected, color: '#eab308' },
                        { label: 'Sanctions Alerts', value: sarReport.executive_summary?.sanctions_alerts, color: '#8b5cf6' },
                      ].map(s => (
                        <div key={s.label} className="stat-card">
                          <div className="stat-value" style={{ color: s.color || 'var(--text-primary)' }}>{s.value ?? '—'}</div>
                          <div className="stat-label">{s.label}</div>
                        </div>
                      ))}
                    </div>

                    {/* Recommended Actions */}
                    <div style={{ background: 'var(--surface-elevated)', borderRadius: 10, padding: 16 }}>
                      <h3 style={{ margin: '0 0 12px', fontSize: 14, color: 'var(--text-secondary)' }}>📌 Recommended Regulatory Actions</h3>
                      <ul style={{ margin: 0, padding: '0 0 0 18px', display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {sarReport.executive_summary?.recommended_regulatory_actions?.map((a, i) => (
                          <li key={i} style={{ fontSize: 13, color: 'var(--text-primary)' }}>{a}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Suspicious Subjects */}
                    {sarReport.suspicious_subjects?.length > 0 && (
                      <div>
                        <h3 style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 10 }}>🚨 Suspicious Subjects ({sarReport.suspicious_subjects.length})</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {sarReport.suspicious_subjects.slice(0, 10).map((subj, i) => (
                            <div key={i} className="account-card" style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, alignItems: 'center' }}>
                              <div>
                                <div style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--text-muted)' }}>{subj.subject_reference}</div>
                                <div style={{ fontSize: 13, marginTop: 4 }}>
                                  Activity: <strong>{subj.suspicious_activity_types?.join(', ')}</strong>
                                  {subj.sanctions_alert && <span style={{ marginLeft: 8, color: '#ef4444', fontWeight: 700 }}>⚠️ SANCTIONS HIT</span>}
                                </div>
                                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                                  Filing: {subj.regulatory_filing}
                                </div>

                                {subj.xai_auditor && (
                                  <div style={{ marginTop: 12, background: 'var(--bg-tinted)', borderRadius: 8, padding: 10, border: '1px solid var(--border-subtle)' }}>
                                    <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 6 }}>
                                      🧠 XAI Auditor (SAR)
                                    </div>
                                    {subj.xai_auditor.plain_english_summary && (
                                      <div style={{
                                        background: '#fff7ed',
                                        border: '1px solid #fed7aa',
                                        borderRadius: 8,
                                        padding: 10,
                                        marginBottom: 8,
                                      }}>
                                        <div style={{ fontSize: 10, fontWeight: 700, color: '#9a3412', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                                          Plain-English Summary (LLM)
                                        </div>
                                        <div style={{ fontSize: 12, color: '#7c2d12', lineHeight: 1.6 }}>
                                          {subj.xai_auditor.plain_english_summary}
                                        </div>
                                        <div style={{ marginTop: 6, fontSize: 10, color: '#9a3412' }}>
                                          Model: {subj.xai_auditor.llm_meta?.model || 'unknown'} · Source: {subj.xai_auditor.llm_meta?.source || 'unknown'}
                                        </div>
                                      </div>
                                    )}
                                    {subj.xai_auditor.xai_reasoning && (
                                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 8 }}>
                                        {subj.xai_auditor.xai_reasoning}
                                      </div>
                                    )}
                                    {(subj.xai_auditor.key_driver_meanings || []).length > 0 && (
                                      <div style={{
                                        background: '#f8fafc',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: 8,
                                        padding: 10,
                                        marginBottom: 8,
                                      }}>
                                        <div style={{ fontSize: 10, fontWeight: 700, color: '#334155', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                                          Key Drivers (What They Mean)
                                        </div>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                          {subj.xai_auditor.key_driver_meanings.map((d, idx) => (
                                            <div key={`${d.feature}-${idx}`}>
                                              <div style={{ fontSize: 12, color: '#0f172a', fontWeight: 700 }}>
                                                {(d.feature || '').replace(/_/g, ' ')} ({((d.importance || 0) * 100).toFixed(1)}%)
                                              </div>
                                              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                                                {d.meaning}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                    {(subj.xai_auditor.suggested_actions || []).length > 0 && (
                                      <div style={{
                                        background: '#f0fdf4',
                                        border: '1px solid #86efac',
                                        borderRadius: 8,
                                        padding: 10,
                                        marginBottom: 8,
                                      }}>
                                        <div style={{ fontSize: 10, fontWeight: 700, color: '#166534', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                                          Suggested Actions
                                        </div>
                                        <ul style={{ margin: 0, paddingLeft: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
                                          {subj.xai_auditor.suggested_actions.map((a, idx) => (
                                            <li key={idx} style={{ fontSize: 12, color: '#166534', lineHeight: 1.5 }}>{a}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {(subj.xai_auditor.feature_attributions || []).length > 0 && (
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                        {subj.xai_auditor.feature_attributions.map((feat, idx) => (
                                          <div key={`${feat.name}-${idx}`} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 52px', gap: 8, alignItems: 'center' }}>
                                            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{(feat.name || '').replace(/_/g, ' ')}</span>
                                            <div style={{ width: '100%', height: 6, borderRadius: 999, background: 'var(--border-subtle)' }}>
                                              <div
                                                style={{
                                                  width: `${Math.max(0, Math.min(100, (feat.importance || 0) * 100))}%`,
                                                  height: '100%',
                                                  borderRadius: 999,
                                                  background: 'linear-gradient(90deg, #06b6d4, #6366f1)'
                                                }}
                                              />
                                            </div>
                                            <span style={{ fontSize: 11, color: 'var(--text-secondary)', textAlign: 'right' }}>
                                              {((feat.importance || 0) * 100).toFixed(1)}%
                                            </span>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                              <div style={{ textAlign: 'right' }}>
                                <div style={{ fontWeight: 700, color: subj.risk_score >= 80 ? '#ef4444' : '#f97316', fontSize: 20 }}>
                                  {subj.risk_score}%
                                </div>
                                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{subj.recommended_action}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Regulatory Framework */}
                    <div style={{ background: 'var(--surface-elevated)', borderRadius: 10, padding: 16 }}>
                      <h3 style={{ margin: '0 0 10px', fontSize: 13, color: 'var(--text-muted)' }}>📜 Regulatory Framework</h3>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {sarReport.report_header?.regulatory_framework?.map((r, i) => (
                          <span key={i} style={{ background: '#1e293b', border: '1px solid #334155', padding: '3px 10px', borderRadius: 12, fontSize: 11, color: '#94a3b8' }}>
                            {r}
                          </span>
                        ))}
                      </div>
                      <p style={{ margin: '14px 0 0', fontSize: 11, color: '#64748b', fontStyle: 'italic' }}>
                        {sarReport.report_header?.classification}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ═══ Intelligence Tab ═══ */}
          {activeTab === 'intelligence' && (
            <IntelligenceTab
              clusters={clusters}
              fetchClusters={fetchClusters}
              intelAccount={intelAccount}
              setIntelAccount={setIntelAccount}
              intelText={intelText}
              setIntelText={setIntelText}
              intelLoading={intelLoading}
              fetchIntelligence={fetchIntelligence}
              intelResult={intelResult}
              nlpText={nlpText}
              setNlpText={setNlpText}
              fetchNLP={fetchNLP}
              nlpResult={nlpResult}
              metricsData={metricsData}
              fetchMetrics={fetchMetrics}
            />
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
      </main>
    </div>
  )
}

export default App
