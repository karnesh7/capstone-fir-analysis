import React, { useState } from 'react';
import { Scale, ChevronDown, ChevronUp, Search, Zap } from 'lucide-react';
import './Stage2Card.css';
import VerdictCard from './VerdictCard';
import SectionInfluence from './SectionInfluence';
import KanoonCaseList from './KanoonCaseList';

export default function Stage2Card({ data }) {
  const [expanded, setExpanded] = useState(false);

  if (!data) return null;

  const { status, sections_searched, cases, verdict_prediction,
          section_influence, api_calls_used, error, telemetry } = data;

  /* ---------- error state ---------- */
  if (status === 'error') {
    return (
      <div className="stage-card stage2">
        <div className="stage-card-header">
          <div className="stage-badge stage2-badge"><Scale size={14} /> Stage 2</div>
          <span className="stage-title">Case Law &amp; Verdict Prediction</span>
        </div>
        <div className="no-match-msg">
          <p>Could not search Indian Kanoon: {error}</p>
        </div>
      </div>
    );
  }

  const statusLabel = {
    success: 'Cases Found',
    partial: 'Partial Results',
    no_results: 'No Cases Found',
  }[status] || status;

  const statusColor = {
    success: '#22c55e',
    partial: '#f59e0b',
    no_results: '#94a3b8',
  }[status] || '#94a3b8';

  const displayCases = expanded ? cases : cases?.slice(0, 3);

  return (
    <div className="stage-card stage2">
      {/* ---- Header ---- */}
      <div className="stage-card-header">
        <div className="stage-badge stage2-badge"><Scale size={14} /> Stage 2</div>
        <span className="stage-title">Indian Kanoon Case Law &amp; Verdict Prediction</span>
        <span className="match-badge" style={{ background: statusColor }}>
          {statusLabel}
        </span>
      </div>

      {telemetry && (
        <div className="telemetry-bar" style={{ marginTop: '10px' }}>
          <span className="telemetry-pill">⚡ <strong>{telemetry.latency_s}s</strong> Kanoon search + verdict</span>
          <span className="telemetry-pill">📡 {telemetry.cases_found} court precedents retrieved</span>
          <span className="telemetry-pill">🤖 {telemetry.engine || 'Indian Kanoon API + Groq'}</span>
        </div>
      )}

      {sections_searched?.length > 0 && (
        <div className="kanoon-sections-searched">
          <Search size={12} />
          <span>Searched: {sections_searched.join(', ')}</span>
        </div>
      )}

      {/* ---- No results ---- */}
      {status === 'no_results' ? (
        <div className="no-match-msg">
          <p>No case law found on Indian Kanoon for the identified IPC sections.</p>
          <p className="hint">You can still ask legal questions in Stage 3.</p>
        </div>
      ) : (
        <>
          <VerdictCard verdict={verdict_prediction} />
          <SectionInfluence sections={section_influence} />
          <KanoonCaseList cases={displayCases} />

          {cases?.length > 3 && (
            <button className="kanoon-toggle" onClick={() => setExpanded(prev => !prev)}>
              {expanded
                ? <><ChevronUp size={14} /> Show fewer</>
                : <><ChevronDown size={14} /> Show all {cases.length} cases</>
              }
            </button>
          )}
        </>
      )}

      {api_calls_used > 0 && (
        <div className="kanoon-footer">
          <span className="kanoon-api-note">
            {cases?.length || 0} case(s) summarized · {api_calls_used} API call(s) · Source: Indian Kanoon
          </span>
        </div>
      )}
    </div>
  );
}
