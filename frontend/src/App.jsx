import React from 'react';
import VoiceRecorder from './components/VoiceRecorder';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto py-8">
        <VoiceRecorder />
      </div>
      
      {/* Footer */}
      <footer className="mt-16 py-8 text-center text-muted-foreground border-t bg-white/50 backdrop-blur-sm">
        <p className="text-sm">
          Powered by <strong className="text-primary">Hume AI</strong> • 
          Built with ❤️ for emotional intelligence
        </p>
        <p className="text-xs mt-1 opacity-75">
          Enterprise-grade emotion analysis platform • Secure • Reliable • Accurate
        </p>
      </footer>
    </div>
  );
}

export default App;