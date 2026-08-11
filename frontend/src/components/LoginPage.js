import React, { useState } from 'react';
import { GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';
import './Login.css';

const LoginPage = ({ onLoginSuccess, hasGoogleAuth }) => {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');

  const handleGoogleSuccess = (credentialResponse) => {
    try {
      const decoded = jwtDecode(credentialResponse.credential);
      console.log('Google Auth Decoded:', decoded);
      const userData = {
        token: credentialResponse.credential,
        name: decoded.name || 'User',
        email: decoded.email,
        picture: decoded.picture || '',
      };
      onLoginSuccess(userData);
    } catch (err) {
      console.error('Failed to decode token:', err);
    }
  };

  const handleDirectLogin = (e) => {
    e.preventDefault();
    const cleanEmail = email.trim() || 'analyst@lexir.local';
    const cleanName = name.trim() || 'Legal Analyst';
    const userData = {
      token: 'local-session-token',
      name: cleanName,
      email: cleanEmail,
      picture: '',
    };
    onLoginSuccess(userData);
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-logo">⚖</div>
        <h1>LexIR</h1>
        <p>Legal Intelligence & Retrieval</p>

        {hasGoogleAuth && (
          <>
            <div className="google-login-btn">
              <GoogleLogin
                onSuccess={handleGoogleSuccess}
                onError={() => {
                  console.log('Google Login Failed');
                  alert('Google Sign-In failed. You can sign in directly below.');
                }}
                useOneTap
              />
            </div>
            <div className="login-divider">
              <span>or sign in directly</span>
            </div>
          </>
        )}

        <form onSubmit={handleDirectLogin} className="direct-login-form">
          <input
            type="text"
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="login-input"
          />
          <input
            type="email"
            placeholder="Email Address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="login-input"
          />
          <button type="submit" className="login-guest-btn">
            Enter LexIR Dashboard
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
