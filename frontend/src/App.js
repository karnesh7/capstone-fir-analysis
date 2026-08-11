import React, { useState, useCallback, useEffect } from 'react';
import { Download, Edit2, Check, X, FileText } from 'lucide-react';
import { GoogleOAuthProvider } from '@react-oauth/google';
import './App.css';
import { useLexIR } from './hooks/useLexIR';
import Sidebar from './components/Sidebar';
import FIRForm from './components/FIRForm';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import LoginPage from './components/LoginPage';

const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID;

function EditableTitle({ title, onSave, visible }) {
  const [isEditing, setIsEditing] = useState(false);
  const [val, setVal] = useState(title);

  useEffect(() => { setVal(title); }, [title]);

  if (!visible) return null;

  if (isEditing) {
    return (
      <div className="top-title-edit" onMouseDown={e => e.stopPropagation()}>
        <input 
          autoFocus 
          value={val} 
          onChange={e => setVal(e.target.value)} 
          onKeyDown={e => {
            if (e.key === 'Enter') { onSave(val); setIsEditing(false); }
            if (e.key === 'Escape') { setVal(title); setIsEditing(false); }
          }}
        />
        <button 
          onMouseDown={e => {
            e.preventDefault(); // Prevent blur
            onSave(val); 
            setIsEditing(false); 
          }} 
          className="title-btn save"
        >
          <Check size={16} />
        </button>
        <button 
          onMouseDown={e => {
            e.preventDefault(); // Prevent blur
            setIsEditing(false); 
            setVal(title); 
          }} 
          className="title-btn cancel"
        >
          <X size={16} />
        </button>
      </div>
    );
  }

  return (
    <div className="top-title-display" onClick={() => setIsEditing(true)}>
      <span>{title}</span>
      <Edit2 size={14} className="edit-icon" />
    </div>
  );
}

function App() {
  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem('lexir_user');
    return savedUser ? JSON.parse(savedUser) : null;
  });

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('lexir_user');
  };

  const {
    connected, currentStage, loading, error,
    fir, stage1, stage2, chatMessages,
    sessions, loadedSession, pipelineProgress,
    sessionId, isFIRExpanded, setIsFIRExpanded,
    startAnalysis, askQuestion, showCases, resetChat,
    loadSession, deleteSession, formResetKey,
    chatTitle, renameSession,
  } = useLexIR(`ws://localhost:8000/ws?user_email=${user?.email || ''}`);

  const hasAnalysis = !!(stage1 || stage2);
  const qaReady = currentStage >= 3;

  const handleLoginSuccess = (userData) => {
    setUser(userData);
    localStorage.setItem('lexir_user', JSON.stringify(userData));
    
    // Notify backend about the user and get a secure session
    fetch('http://localhost:8000/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData),
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'ok') {
        console.log('Backend session established');
        // Force refresh sessions list after backend sync
        window.location.reload(); 
      }
    })
    .catch(err => console.error('Error syncing user with backend:', err));
  };

  /* ---- PDF download ---- */
  const [pdfLoading, setPdfLoading] = useState(false);

  const handleDownloadPDF = useCallback(async () => {
    setPdfLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/fir/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fir: fir || {},
          analysis: stage1?._raw_analysis || null,
        }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `FIR_${fir?.fir_id || 'report'}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF download failed:', err);
      alert('Failed to generate PDF. Is the server running?');
    } finally {
      setPdfLoading(false);
    }
  }, [fir, stage1]);

  /* ---- Export Full Legal Brief ---- */
  const handleExportBrief = useCallback(() => {
    if (!stage1) return;
    const firData = fir || stage1.fir_summary || {};
    const firId = firData.fir_id || 'FIR-REPORT';

    let md = `# LEXIR LEGAL INTELLIGENCE & CASE BRIEFING REPORT\n`;
    md += `**Generated:** ${new Date().toLocaleString()}\n`;
    md += `**Case Title:** ${chatTitle || 'Legal Assessment'}\n`;
    md += `**User:** ${user?.name || 'Legal Analyst'} (${user?.email || 'N/A'})\n\n`;
    md += `---\n\n`;

    md += `## 1. FIRST INFORMATION REPORT (FIR) DETAILS\n\n`;
    md += `- **FIR ID:** ${firData.fir_id || 'N/A'}\n`;
    md += `- **Date:** ${firData.date || 'N/A'}\n`;
    md += `- **Complainant:** ${firData.complainant_name || firData.complainant || 'N/A'}\n`;
    md += `- **Accused:** ${Array.isArray(firData.accused_names) ? firData.accused_names.join(', ') : (firData.accused_names || firData.accused || 'N/A')}\n`;
    md += `- **Victim:** ${firData.victim_name || firData.victim || 'N/A'}\n`;
    md += `- **Location / Police Station:** ${firData.location || 'N/A'} (PS: ${firData.police_station || 'N/A'})\n\n`;

    md += `### Incident Summary\n`;
    md += `${firData.incident_description || firData.incident || 'N/A'}\n\n`;

    if (firData.victim_impact) {
      md += `**Victim Impact:** ${firData.victim_impact}\n\n`;
    }
    if (firData.evidence) {
      md += `**Evidence Collected:** ${firData.evidence}\n\n`;
    }

    md += `---\n\n`;
    md += `## 2. STAGE 1: STATUTE RAG & LEGAL SECTION MAPPING\n\n`;
    md += `- **Primary Intent:** ${stage1.intent?.primary || 'N/A'} (Confidence: ${Math.round((stage1.intent?.confidence || 0) * 100)}%)\n`;
    md += `- **Assessed Severity:** ${(stage1.severity || 'unknown').toUpperCase()}\n`;
    if (stage1.telemetry) {
      md += `- **RAG Telemetry:** ${stage1.telemetry.latency_s}s latency • Model: ${stage1.telemetry.model} • ${stage1.telemetry.vectors_scanned} statutes scanned\n`;
    }
    md += `\n### Applicable Legal Provisions:\n`;
    if (stage1.statutes && stage1.statutes.length > 0) {
      stage1.statutes.forEach((s, idx) => {
        md += `${idx + 1}. **${s.primary.law} ${s.primary.section}**${s.primary.title ? ` — ${s.primary.title}` : ''}\n`;
        if (s.primary.reasoning) md += `   - *Application:* ${s.primary.reasoning}\n`;
        if (s.corresponding_sections && s.corresponding_sections.length > 0) {
          s.corresponding_sections.forEach(c => {
            md += `   - *Modernized BNS Counterpart:* **${c.law} ${c.section}**\n`;
          });
        }
      });
    } else {
      md += `*No criminal sections identified (Civil dispute / non-penal complaint).*\n`;
    }

    if (stage1.legal_basis) {
      md += `\n**Legal Basis & Statutory Reasoning:**\n${stage1.legal_basis}\n\n`;
    }

    if (stage2) {
      md += `---\n\n`;
      md += `## 3. STAGE 2: COURT PRECEDENTS & VERDICT PREDICTION\n\n`;
      if (stage2.verdict_prediction) {
        md += `- **Predicted Verdict:** ${stage2.verdict_prediction.verdict || 'N/A'}\n`;
        md += `- **Conviction / Probability:** ${stage2.verdict_prediction.confidence ? `${Math.round(stage2.verdict_prediction.confidence * 100)}%` : 'N/A'}\n`;
        md += `- **Estimated Punishment Range:** ${stage2.verdict_prediction.punishment_range || 'N/A'}\n`;
        if (stage2.verdict_prediction.summary) {
          md += `- **Verdict Analysis:** ${stage2.verdict_prediction.summary}\n`;
        }
      }

      if (stage2.cases && stage2.cases.length > 0) {
        md += `\n### Indian Kanoon Court Precedents:\n`;
        stage2.cases.forEach((c, idx) => {
          md += `#### ${idx + 1}. ${c.title || 'Case Precedent'}\n`;
          md += `- **Court:** ${c.court || 'High Court / Supreme Court'} (${c.year || 'N/A'})\n`;
          if (c.summary) md += `- **Summary:** ${c.summary}\n`;
          if (c.verdict) md += `- **Outcome:** ${c.verdict}\n`;
        });
      }
    }

    const qaMessages = chatMessages.filter(m => m.type === 'user_question' || m.type === 'qa_answer');
    if (qaMessages.length > 0) {
      md += `\n---\n\n`;
      md += `## 4. STAGE 3: LEGAL CONSULTATION Q&A TRANSCRIPT\n\n`;
      qaMessages.forEach(m => {
        if (m.type === 'user_question') {
          md += `**Q: ${m.text}**\n\n`;
        } else if (m.type === 'qa_answer') {
          md += `*A:* ${m.text}\n\n`;
        }
      });
    }

    md += `---\n\n*Generated by LexIR — Legal Intelligence & Retrieval System*\n`;

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LexIR_Legal_Brief_${firId}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }, [fir, stage1, stage2, chatMessages, chatTitle, user]);

  if (!user) {
    if (GOOGLE_CLIENT_ID) {
      return (
        <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
          <LoginPage onLoginSuccess={handleLoginSuccess} hasGoogleAuth={true} />
        </GoogleOAuthProvider>
      );
    }
    return <LoginPage onLoginSuccess={handleLoginSuccess} hasGoogleAuth={false} />;
  }

  return (
    <div className="app-layout">
      <Sidebar
        connected={connected}
        currentStage={currentStage}
        hasAnalysis={hasAnalysis}
        sessions={sessions}
        activeSessionId={sessionId}
        onShowCases={showCases}
        onReset={resetChat}
        onLoadSession={loadSession}
        onDeleteSession={deleteSession}
        onNewSession={resetChat}
        user={user}
        onLogout={handleLogout}
      />

      <main className="main-panel">
        {/* Top bar */}
        <header className="top-bar">
          <div className="top-bar-left">
            <h2>
              {currentStage === 0 && 'Submit a Case'}
              {currentStage === 1 && 'Stage 1 — FIR Analysis'}
              {currentStage === 2 && 'Stage 2 — Similar Cases'}
              {currentStage >= 3 && 'Stage 3 — Precedent Q&A'}
            </h2>
          </div>

          <div className="top-bar-center">
            <EditableTitle 
              title={chatTitle} 
              onSave={(newTitle) => renameSession(sessionId || loadedSession?.sessionId, newTitle)}
              visible={!!(sessionId || loadedSession?.sessionId)}
            />
          </div>

          <div className="top-bar-actions">
            {stage1 && (
              <>
                <button
                  className="btn btn-pdf-download"
                  onClick={handleExportBrief}
                  title="Export complete case intelligence report as Markdown"
                  style={{ background: 'rgba(59, 130, 246, 0.15)', borderColor: 'rgba(59, 130, 246, 0.4)' }}
                >
                  <FileText size={16} />
                  Export Legal Brief
                </button>
                <button
                  className="btn btn-pdf-download"
                  onClick={handleDownloadPDF}
                  disabled={pdfLoading}
                  title="Download filled FIR form as PDF"
                >
                  <Download size={16} />
                  {pdfLoading ? 'Generating…' : 'Download FIR PDF'}
                </button>
              </>
            )}
            {error && <span className="top-error">{error}</span>}
          </div>
        </header>

        {/* FIR Form (always visible — collapsed after analysis) */}
        <div className="form-container">
          <FIRForm
            onSubmit={startAnalysis}
            disabled={loading || !connected}
            hasAnalysis={hasAnalysis}
            formResetKey={formResetKey}
            isExpanded={isFIRExpanded}
            onToggleExpand={setIsFIRExpanded}
          />
        </div>

        {/* Chat area */}
        <ChatArea
          messages={chatMessages}
          loading={loading}
          loadedSession={loadedSession}
          pipelineProgress={pipelineProgress}
          chatTitle={chatTitle}
          renameSession={renameSession}
          sessionId={sessionId}
        />

        {/* Chat input (shown once Q&A stage is reached) */}
        {qaReady && (
          <ChatInput
            onSend={askQuestion}
            disabled={loading || !connected}
            placeholder="Ask about sections, precedents, punishments, bail..."
          />
        )}
      </main>
    </div>
  );
}

export default App;
