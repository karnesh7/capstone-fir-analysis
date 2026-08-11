import React, { useState } from 'react';
import { Shield, Scale, FileText, AlertTriangle, Zap, ArrowRight, X, Info } from 'lucide-react';
import './Stage1Card.css';

export default function Stage1Card({ data }) {
  const [selectedStatute, setSelectedStatute] = useState(null);

  if (!data) return null;

  const { fir_summary, intent, severity, legal_basis, statutes, mapped_sections,
          chunks_retrieved, chunks_after_filtering, confidence, telemetry } = data;

  const severityColor = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#22c55e',
  }[severity?.toLowerCase()] || '#94a3b8';

  return (
    <div className="stage-card stage1">
      <div className="stage-card-header">
        <div className="stage-badge stage1-badge">
          <Shield size={14} /> Stage 1
        </div>
        <span className="stage-title">FIR Analysis & Section Mapping</span>
        {telemetry && (
          <div className="stage-telemetry-badge" title="RAG Pipeline Execution Telemetry">
            <Zap size={12} style={{ color: '#eab308' }} />
            <span><strong>{telemetry.latency_s}s</strong> • {telemetry.model}</span>
          </div>
        )}
      </div>

      {/* Telemetry pill bar */}
      {telemetry && (
        <div className="telemetry-bar">
          <span className="telemetry-pill">⚡ <strong>{telemetry.latency_s}s</strong> latency</span>
          <span className="telemetry-pill">🧠 {telemetry.model}</span>
          <span className="telemetry-pill">🎯 {telemetry.vectors_scanned || 865} statutes scanned ({telemetry.similarity_metric || 'Cosine'})</span>
        </div>
      )}

      {/* FIR Summary */}
      <div className="stage-section">
        <h4><FileText size={14} /> Case Summary</h4>
        <div className="info-grid">
          <span className="info-label">FIR ID</span>
          <span>{fir_summary?.fir_id}</span>
          <span className="info-label">Date</span>
          <span>{fir_summary?.date}</span>
          <span className="info-label">Complainant</span>
          <span>{fir_summary?.complainant || 'N/A'}</span>
          <span className="info-label">Accused</span>
          <span>{fir_summary?.accused?.join(', ') || 'N/A'}</span>
          <span className="info-label">Victim</span>
          <span>{fir_summary?.victim || 'N/A'}</span>
          <span className="info-label">Location</span>
          <span>{fir_summary?.location || 'N/A'}</span>
        </div>
      </div>

      {/* Intent */}
      <div className="stage-section">
        <h4><AlertTriangle size={14} /> Intent Identification</h4>
        <div className="intent-row">
          <span className="intent-primary">{intent?.primary}</span>
          <span className="confidence-badge">
            {Math.round((intent?.confidence || 0) * 100)}% confident
          </span>
        </div>
        {intent?.secondary?.length > 0 && (
          <div className="intent-secondary">
            Secondary: {intent.secondary.join(', ')}
          </div>
        )}
        <div className="severity-row">
          Severity:
          <span className="severity-badge" style={{ background: severityColor }}>
            {severity?.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Statutes */}
      <div className="stage-section">
        <h4><Scale size={14} /> Applicable Sections ({statutes?.length || 0}) <span className="statute-click-hint">(click section to compare IPC ↔ BNS)</span></h4>
        <div className="statutes-list">
          {statutes?.map((s, i) => (
            <div
              key={i}
              className="statute-item clickable-statute"
              onClick={() => setSelectedStatute(s)}
              title="Click to view full IPC ↔ BNS cross-comparison"
            >
              <div className="statute-primary">
                <strong>{s.primary.law} {s.primary.section}</strong>
                {s.primary.title && <span className="statute-title"> — {s.primary.title}</span>}
                <span className="compare-tag"><Info size={11} /> Compare</span>
              </div>
              {s.primary.reasoning && (
                <p className="statute-reasoning">{s.primary.reasoning}</p>
              )}
              {s.corresponding_sections?.length > 0 && (
                <div className="statute-corresponding">
                  {s.corresponding_sections.map((c, j) => (
                    <span key={j} className="corr-badge">
                      → {c.law} {c.section} (New Law)
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Legal Basis */}
      {legal_basis && (
        <div className="stage-section">
          <h4>Legal Basis</h4>
          <p className="legal-basis-text">{legal_basis}</p>
        </div>
      )}

      {/* Footer stats */}
      <div className="stage-footer">
        <span>Chunks: {chunks_retrieved} retrieved → {chunks_after_filtering} filtered</span>
        <span>Confidence: {typeof confidence === 'number' ? Math.round(confidence * 100) + '%' : confidence}</span>
      </div>

      {/* IPC <-> BNS Comparison Modal */}
      {selectedStatute && (
        <div className="comparison-modal-backdrop" onClick={() => setSelectedStatute(null)}>
          <div className="comparison-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="comparison-modal-header">
              <div className="comparison-header-title">
                <Scale size={18} className="modal-icon" />
                <h3>Statute Cross-Comparison: IPC ↔ BNS (2023)</h3>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedStatute(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="comparison-columns">
              <div className="comparison-box old-law">
                <div className="comparison-box-tag">Legacy Law</div>
                <h4>{selectedStatute.primary.law} {selectedStatute.primary.section}</h4>
                {selectedStatute.primary.title && <div className="law-title">{selectedStatute.primary.title}</div>}
                {selectedStatute.primary.extract ? (
                  <p className="law-extract">{selectedStatute.primary.extract}</p>
                ) : (
                  <p className="law-extract">{selectedStatute.primary.reasoning || 'Standard penal provision under legacy code.'}</p>
                )}
              </div>

              <div className="comparison-divider">
                <ArrowRight size={20} />
              </div>

              <div className="comparison-box new-law">
                <div className="comparison-box-tag new-tag">Bharatiya Nyaya Sanhita (2023)</div>
                {selectedStatute.corresponding_sections?.length > 0 ? (
                  selectedStatute.corresponding_sections.map((c, idx) => (
                    <div key={idx} className="corr-detail-item">
                      <h4>{c.law} {c.section}</h4>
                      <p className="law-extract">
                        {c.extract || 'Corresponding modernized provision in BNS 2023 replacing the legacy section with updated procedural definitions and penal structure.'}
                      </p>
                    </div>
                  ))
                ) : (
                  <div className="no-corr">
                    <h4>Direct Application</h4>
                    <p className="law-extract">This statute applies directly under the primary identified legal provision.</p>
                  </div>
                )}
              </div>
            </div>

            {selectedStatute.primary.reasoning && (
              <div className="comparison-reasoning-box">
                <strong>FIR Application Rationale:</strong>
                <p>{selectedStatute.primary.reasoning}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
