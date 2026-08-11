import React from 'react';
import { Wifi, WifiOff, Plus, Loader2, Trash2 } from 'lucide-react';
import './Sidebar.css';

function formatSessionTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' });
  } catch {
    return iso;
  }
}

export default function Sidebar({
  connected,
  onNewSession,
  sessions = [],
  activeSessionId,
  onLoadSession,
  onDeleteSession,
  user,
  onLogout,
}) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="brand-icon">⚖</div>
        <h1>LexIR</h1>
        <p>Legal Intelligence & Retrieval</p>
      </div>

      {/* Connection status */}
      <div className={`connection-status ${connected ? 'online' : 'offline'}`}>
        {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Chat history */}
      <div className="sidebar-history">
        <h4>History</h4>
        <button
          type="button"
          className="btn btn-new-session"
          onClick={onNewSession}
          aria-label="+ New Session"
        >
          <Plus size={16} aria-hidden />
          <span>New Session</span>
        </button>
        <div className="history-list">
          {sessions.length === 0 && (
            <p className="history-empty">No saved sessions yet.</p>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`history-row-wrap ${s.id === activeSessionId ? 'active' : ''}`}
            >
              <div
                className="history-row"
                onClick={() => onLoadSession?.(s.id)}
                role="button"
                tabIndex={0}
              >
                <div className="history-row-content">
                  <span className="history-preview">
                    {s.status === 'pending' ? (
                      <span className="history-pending">
                        <Loader2 size={12} className="spin history-spinner" aria-hidden />
                        Analyzing…
                      </span>
                    ) : (
                      s.title || (s.fir_preview || '').slice(0, 60)
                    )}
                  </span>
                  <span className="history-meta">{formatSessionTime(s.created_at)}</span>
                </div>
                <button
                  type="button"
                  className="history-delete"
                  title="Delete session"
                  aria-label="Delete session"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession?.(s.id);
                  }}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* User info & Logout */}
      {user && (
        <div className="sidebar-footer">
          <div className="user-info">
            <div className="avatar-container">
              <img 
                src={user.picture} 
                alt={user.name} 
                className="user-avatar"
                referrerPolicy="no-referrer"
                onError={(e) => {
                  console.error("Avatar failed to load", e);
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'flex';
                }}
              />
              <div className="avatar-fallback" style={{ display: 'none' }}>
                {user.name?.charAt(0).toUpperCase()}
              </div>
            </div>
            <div className="user-details">
              <span className="user-name">{user.name}</span>
              <button className="logout-btn" onClick={onLogout}>Logout</button>
            </div>
          </div>

          {/* Indian Kanoon Data Attribution */}
          <div className="sidebar-attribution">
            <a
              href="https://indiankanoon.org"
              target="_blank"
              rel="noopener noreferrer"
              title="Legal precedent search powered by Indian Kanoon"
              className="kanoon-attribution-link"
            >
              <picture>
                <source media="(max-width: 768px)" srcSet={`${process.env.PUBLIC_URL}/ikanoon_mobile_powered_transparent.png`} />
                <img
                  src={`${process.env.PUBLIC_URL}/ikanoon6_powered_transparent.png`}
                  alt="Powered by Indian Kanoon"
                  className="kanoon-badge-img"
                />
              </picture>
            </a>
          </div>
        </div>
      )}
    </aside>
  );
}
