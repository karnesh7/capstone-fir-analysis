import React, { useState, useCallback, useEffect } from 'react';
import { Download, Edit2, Check, X } from 'lucide-react';
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
              <button
                className="btn btn-pdf-download"
                onClick={handleDownloadPDF}
                disabled={pdfLoading}
                title="Download filled FIR form as PDF"
              >
                <Download size={16} />
                {pdfLoading ? 'Generating…' : 'Download FIR PDF'}
              </button>
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
