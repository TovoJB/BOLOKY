"use client";

import React, { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

interface TTSRecord {
  id: string;
  text: string;
  speed: number;
  audio_file: string;
  audio_url: string;
  created_at: string;
}

interface TTSTabProps {
  /** Texte pré-rempli (depuis le mode texte via bouton Écouter) */
  prefillText?: string;
}

export default function TTSTab({ prefillText = "" }: TTSTabProps) {
  const [text, setText] = useState(prefillText);
  const [speed, setSpeed] = useState(1.0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState("");
  const [error, setError] = useState("");
  const [history, setHistory] = useState<TTSRecord[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (prefillText) setText(prefillText);
  }, [prefillText]);

  const loadHistory = async () => {
    try {
      const res = await fetch(`${API}/tts_list`);
      if (res.ok) setHistory(await res.json());
    } catch { }
  };

  useEffect(() => { loadHistory(); }, []);

  const handleGenerate = async () => {
    if (!text.trim()) return;
    setIsGenerating(true);
    setError("");
    setCurrentAudioUrl("");

    try {
      const res = await fetch(`${API}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), speed }),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Erreur serveur");
      }
      const data = await res.json();
      const fullUrl = `${API}${data.audio_url}`;
      setCurrentAudioUrl(fullUrl);
      await loadHistory();
      // Auto-play
      setTimeout(() => audioRef.current?.play(), 300);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDelete = async (id: string) => {
    await fetch(`${API}/tts/${id}`, { method: "DELETE" });
    setHistory(prev => prev.filter(h => h.id !== id));
  };

  return (
    <div className="space-y-6">
      {/* Textarea */}
      <div className="bg-slate-900/40 rounded-2xl border border-slate-800 overflow-hidden">
        <div className="px-4 pt-4 pb-1">
          <label className="text-xs text-emerald-400 uppercase tracking-wider font-semibold">
            🔊 Texte à synthétiser (Malgache)
          </label>
        </div>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Soraty eto ny teny Malagasy ho synthesizina..."
          className="w-full h-36 px-4 py-3 bg-transparent text-slate-100 placeholder-slate-600 focus:outline-none resize-none text-base leading-relaxed"
          maxLength={500}
        />
        <div className="px-4 pb-3 text-xs text-slate-600 text-right">{text.length} / 500</div>
      </div>

      {/* Vitesse */}
      <div>
        <label className="text-xs text-slate-500 uppercase tracking-wider mb-3 flex items-center justify-between">
          <span>Vitesse de parole</span>
          <span className="text-emerald-400 font-bold">{speed.toFixed(1)}x</span>
        </label>
        <input
          type="range"
          min="0.5"
          max="2.0"
          step="0.1"
          value={speed}
          onChange={e => setSpeed(parseFloat(e.target.value))}
          className="w-full accent-emerald-500"
        />
        <div className="flex justify-between text-[10px] text-slate-600 mt-1">
          <span>0.5x (lent)</span>
          <span>1.0x (normal)</span>
          <span>2.0x (rapide)</span>
        </div>
      </div>

      {/* Bouton */}
      <button
        onClick={handleGenerate}
        disabled={!text.trim() || isGenerating}
        className="w-full py-4 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-30 disabled:cursor-not-allowed text-white font-bold text-base transition-all duration-300 active:scale-[0.98] flex items-center justify-center gap-3 shadow-lg shadow-emerald-500/20"
      >
        {isGenerating ? (
          <>
            <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Génération de l&apos;audio...</span>
          </>
        ) : (
          <span>🔊 Générer l&apos;Audio</span>
        )}
      </button>

      {error && (
        <div className="bg-rose-950/40 border border-rose-800/50 rounded-2xl p-4 text-rose-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Lecteur audio */}
      {currentAudioUrl && (
        <div className="bg-slate-900/60 border border-emerald-800/40 rounded-2xl p-5 space-y-3">
          <div className="text-xs text-emerald-400 font-semibold uppercase tracking-wider mb-2">▶ Résultat Audio</div>
          <audio ref={audioRef} controls src={currentAudioUrl} className="w-full rounded-xl" />
          <a
            href={currentAudioUrl}
            download
            className="inline-flex items-center gap-2 text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Télécharger le WAV
          </a>
        </div>
      )}

      {/* Historique */}
      {history.length > 0 && (
        <div>
          <h3 className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-3">Historique TTS</h3>
          <div className="space-y-2">
            {history.map(item => (
              <div key={item.id} className="bg-slate-900/40 border border-slate-800/60 rounded-2xl p-4 hover:border-slate-700/60 transition-all">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0 space-y-2">
                    <p className="text-slate-300 text-sm truncate">&ldquo;{item.text}&rdquo;</p>
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] bg-slate-800 text-slate-500 px-2 py-0.5 rounded-full">{item.speed}x</span>
                      <span className="text-[10px] text-slate-600">{new Date(item.created_at).toLocaleString("fr-FR")}</span>
                    </div>
                    <audio controls src={`${API}${item.audio_url}`} className="w-full h-8 rounded-lg" style={{ height: "32px" }} />
                  </div>
                  <button
                    onClick={() => handleDelete(item.id)}
                    className="flex-shrink-0 p-2 rounded-xl text-slate-600 hover:text-rose-400 hover:bg-rose-950/30 transition-all"
                    title="Supprimer"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
