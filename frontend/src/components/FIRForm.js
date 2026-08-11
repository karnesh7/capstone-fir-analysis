import React, { useState, useEffect, useCallback } from 'react';
import { Send, ChevronUp } from 'lucide-react';
import './FIRForm.css';

const EMPTY = {
  complainant_name: '',
  father_husband_name: '',
  complainant_dob: '',
  nationality: 'Indian',
  occupation: '',
  complainant_address: '',
  accused_names_str: '',
  victim_name: '',
  incident_description: '',
  victim_impact: '',
  evidence: '',
  location: '',
  district: '',
  police_station: '',
  date: new Date().toISOString().slice(0, 10),
  delay_reason: '',
  properties_stolen: '',
  property_value: '',
};

function firToFormFields(fir) {
  if (!fir || typeof fir !== 'object') return { ...EMPTY };
  return {
    ...EMPTY,
    complainant_name: fir.complainant_name ?? '',
    father_husband_name: fir.father_husband_name ?? '',
    complainant_dob: fir.complainant_dob ?? '',
    nationality: fir.nationality || 'Indian',
    occupation: fir.occupation ?? '',
    complainant_address: fir.complainant_address ?? '',
    accused_names_str: Array.isArray(fir.accused_names)
      ? fir.accused_names.join(', ')
      : (fir.accused_names_str ?? ''),
    victim_name: fir.victim_name ?? '',
    incident_description: fir.incident_description ?? '',
    victim_impact: fir.victim_impact ?? '',
    evidence: fir.evidence ?? '',
    location: fir.location ?? '',
    district: fir.district ?? '',
    police_station: fir.police_station ?? '',
    date: fir.date || EMPTY.date,
    delay_reason: fir.delay_reason ?? '',
    properties_stolen: fir.properties_stolen ?? '',
    property_value: fir.property_value ?? '',
  };
}

export default function FIRForm({
  onSubmit,
  disabled,
  hasAnalysis,
  formResetKey = 0,
  viewMode = 'edit',
  firSnapshot = null,
  isExpanded: controlledExpanded,
  onToggleExpand,
}) {
  const [form, setForm] = useState(EMPTY);
  const [internalExpanded, setInternalExpanded] = useState(true);
  const [historyEditUnlocked, setHistoryEditUnlocked] = useState(false);

  const isHistory = viewMode === 'history';

  // Use controlled state if provided, otherwise use internal state
  const isExpanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded;
  const setExpanded = useCallback((val) => {
    if (onToggleExpand) onToggleExpand(val);
    else setInternalExpanded(val);
  }, [onToggleExpand]);

  useEffect(() => {
    setForm(EMPTY);
    setExpanded(true);
    setHistoryEditUnlocked(false);
  }, [formResetKey, setExpanded]);

  useEffect(() => {
    if (isHistory && firSnapshot) {
      setForm(firToFormFields(firSnapshot));
      setExpanded(true);
      setHistoryEditUnlocked(false);
    }
  }, [isHistory, firSnapshot, setExpanded]);

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }));

  const readOnly = isHistory && !historyEditUnlocked;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.incident_description.trim()) return;

    const fir = {
      fir_id: `FIR-${Date.now()}`,
      date: form.date,
      complainant_name: form.complainant_name || null,
      father_husband_name: form.father_husband_name || null,
      complainant_dob: form.complainant_dob || null,
      nationality: form.nationality || 'Indian',
      occupation: form.occupation || null,
      complainant_address: form.complainant_address || null,
      accused_names: form.accused_names_str
        ? form.accused_names_str.split(',').map(s => s.trim()).filter(Boolean)
        : [],
      victim_name: form.victim_name || null,
      incident_description: form.incident_description,
      victim_impact: form.victim_impact,
      evidence: form.evidence,
      location: form.location,
      district: form.district || null,
      police_station: form.police_station,
      delay_reason: form.delay_reason || null,
      properties_stolen: form.properties_stolen || null,
      property_value: form.property_value || null,
    };

    onSubmit(fir);
    if (!isHistory) setExpanded(false);
  };

  const handleSampleFIR = () => {
    onSubmit(null);
    if (!isHistory) setExpanded(false);
  };

  if (!isExpanded && !isHistory) {
    return (
      <div className="fir-form-collapsed" onClick={() => setExpanded(true)}>
        <span className="fir-form-collapsed-text">
          {hasAnalysis ? 'Edit FIR & re-analyze' : 'FIR submitted — click to edit & resubmit'}
        </span>
      </div>
    );
  }

  return (
    <form className={`fir-form ${isHistory ? 'fir-form-history' : ''}`} onSubmit={handleSubmit}>
      <div className="fir-form-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3>
            {isHistory
              ? 'FIR (loaded session)'
              : hasAnalysis
                ? 'Edit & Re-analyze Case'
                : 'Submit a Case for Analysis'}
          </h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {isHistory && (
              <>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '4px 10px', fontSize: 12 }}
                  onClick={() => setHistoryEditUnlocked(true)}
                  disabled={historyEditUnlocked}
                >
                  Edit FIR & re-analyze
                </button>
                {historyEditUnlocked && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ padding: '4px 10px', fontSize: 12 }}
                    onClick={() => setExpanded(false)}
                  >
                    <ChevronUp size={14} /> Collapse
                  </button>
                )}
              </>
            )}
            {/* {hasAnalysis && !isHistory && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ padding: '4px 10px', fontSize: 12 }}
                onClick={() => setExpanded(false)}
              >
                <ChevronUp size={14} /> Collapse
              </button>
            )} */}
            {!isHistory && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ padding: '4px 10px', fontSize: 12 }}
                onClick={() => setExpanded(false)}
              >
                <ChevronUp size={14} /> Collapse
              </button>
            )}
          </div>
        </div>
        <p>
          {isHistory
            ? 'Read-only snapshot of the FIR for this session. Unlock to edit and run a new analysis.'
            : 'Fill in the case details below, or use the sample FIR to try the system.'}
        </p>
      </div>

      {/* ---- Case core ---- */}
      <div className="fir-form-grid">
        <div className="fir-field">
          <label>Complainant Name</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Rajesh Kumar" value={form.complainant_name} onChange={set('complainant_name')} />
        </div>
        <div className="fir-field">
          <label>Father / Husband Name</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Suresh Kumar" value={form.father_husband_name} onChange={set('father_husband_name')} />
        </div>

        <div className="fir-field">
          <label>Complainant DOB</label>
          <input readOnly={readOnly} type="date" value={form.complainant_dob} onChange={set('complainant_dob')} />
        </div>
        <div className="fir-field">
          <label>Nationality</label>
          <input readOnly={readOnly} type="text" placeholder="Indian" value={form.nationality} onChange={set('nationality')} />
        </div>

        <div className="fir-field">
          <label>Occupation</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Software Engineer" value={form.occupation} onChange={set('occupation')} />
        </div>
        <div className="fir-field">
          <label>Complainant Address</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. 12, MG Road, New Delhi" value={form.complainant_address} onChange={set('complainant_address')} />
        </div>
      </div>

      <div className="fir-form-grid">
        <div className="fir-field">
          <label>Victim Name</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Priya Sharma" value={form.victim_name} onChange={set('victim_name')} />
        </div>
        <div className="fir-field">
          <label>Accused Names <span className="hint">(comma-separated)</span></label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Amit Singh, Vikram Patel" value={form.accused_names_str} onChange={set('accused_names_str')} />
        </div>

        <div className="fir-field">
          <label>Date of Incident</label>
          <input readOnly={readOnly} type="date" value={form.date} onChange={set('date')} />
        </div>
        <div className="fir-field">
          <label>Location / Place of Occurrence</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Sector 45, Chandigarh" value={form.location} onChange={set('location')} />
        </div>

        <div className="fir-field">
          <label>District</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Central Delhi" value={form.district} onChange={set('district')} />
        </div>
        <div className="fir-field">
          <label>Police Station</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Sector 41 PS" value={form.police_station} onChange={set('police_station')} />
        </div>
      </div>

      {/* ---- Full-width fields ---- */}
      <div className="fir-field full">
        <label>Incident Description <span className="required">*</span></label>
        <textarea
          readOnly={readOnly}
          rows={4}
          placeholder="Describe what happened in detail — this is the most important field for analysis..."
          value={form.incident_description}
          onChange={set('incident_description')}
          required
        />
      </div>

      <div className="fir-field full">
        <label>Victim Impact</label>
        <textarea
          readOnly={readOnly}
          rows={2}
          placeholder="Physical injuries, psychological trauma, financial loss..."
          value={form.victim_impact}
          onChange={set('victim_impact')}
        />
      </div>

      <div className="fir-field full">
        <label>Evidence</label>
        <textarea
          readOnly={readOnly}
          rows={2}
          placeholder="CCTV footage, witnesses, medical reports, physical evidence..."
          value={form.evidence}
          onChange={set('evidence')}
        />
      </div>

      {/* ---- Property & delay ---- */}
      <div className="fir-form-grid">
        <div className="fir-field">
          <label>Properties Stolen / Involved</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. Gold chain, mobile phone" value={form.properties_stolen} onChange={set('properties_stolen')} />
        </div>
        <div className="fir-field">
          <label>Total Value of Property</label>
          <input readOnly={readOnly} type="text" placeholder="e.g. ₹50,000" value={form.property_value} onChange={set('property_value')} />
        </div>
      </div>

      <div className="fir-field full">
        <label>Reason for Delay in Reporting <span className="hint">(if any)</span></label>
        <input readOnly={readOnly} type="text" placeholder="e.g. Complainant was hospitalised" value={form.delay_reason} onChange={set('delay_reason')} />
      </div>

      {!isHistory && (
        <div className="fir-form-actions">
          <button type="submit" className="btn btn-primary" disabled={disabled || !form.incident_description.trim()}>
            <Send size={16} />
            {hasAnalysis ? 'Re-analyze Case' : 'Analyze Case'}
          </button>
          <button type="button" className="btn btn-secondary" onClick={handleSampleFIR} disabled={disabled}>
            Use Sample FIR
          </button>
        </div>
      )}

      {isHistory && historyEditUnlocked && (
        <div className="fir-form-actions">
          <button type="submit" className="btn btn-primary" disabled={disabled || !form.incident_description.trim()}>
            <Send size={16} />
            Re-analyze Case
          </button>
        </div>
      )}
    </form>
  );
}
