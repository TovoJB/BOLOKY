"use client";

import React, { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

interface SpeechTabProps {
  onSendToTTS?: (text: string) => void;
}

export default function SpeechTab({ onSendToTTS }: SpeechTabProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [translation, setTranslation] = useState("");
  const [vezoAudioUrl, setVezoAudioUrl] = useState("");
  const [error, setError] = useState("");
  const [direction, setDirection] = useState<"english-to-vezo" | "english-to-merina" | "english-to-betsileo">("english-to-vezo");
  
  // Audio Vezo parameters (session state, reset on reload)
  const [vezoSpeed, setVezoSpeed] = useState<number>(1.15);
  const [vezoSr, setVezoSr] = useState<number>(22050);
  const [vezoNoiseScale, setVezoNoiseScale] = useState<number>(0.35);
  const [vezoNoiseScaleDuration, setVezoNoiseScaleDuration] = useState<number>(0.6);
  const [vezoHighpassCutoff, setVezoHighpassCutoff] = useState<number>(60);
  const [vezoNoiseGate, setVezoNoiseGate] = useState<number>(0.01);
  const [vezoPeakNorm, setVezoPeakNorm] = useState<number>(0.95);
  const [vezoWarmth, setVezoWarmth] = useState<number>(0.5);

  // Audio Merina parameters (session state, reset on reload)
  const [merinaSpeed, setMerinaSpeed] = useState<number>(1.0);
  const [merinaSr, setMerinaSr] = useState<number>(22050);
  const [merinaHighpassCutoff, setMerinaHighpassCutoff] = useState<number>(60);

  const [showSettings, setShowSettings] = useState<boolean>(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    setError("");
    setTranscription("");
    setTranslation("");
    setVezoAudioUrl("");
    audioChunksRef.current = [];
    setRecordingTime(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());

        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        if (audioBlob.size === 0) {
          setError("Aucun son capturé par le microphone.");
          return;
        }

        await processAudio(audioBlob);
      };

      mediaRecorder.start(200);
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error(err);
      setError("Erreur d'accès au microphone : " + (err.message || "Accès refusé."));
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setError("");

    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");
    formData.append("direction", direction);
    formData.append("language", "en");

    if (direction === "english-to-merina") {
      formData.append("speed", merinaSpeed.toString());
      formData.append("sr", merinaSr.toString());
      formData.append("highpass_cutoff", merinaHighpassCutoff.toString());
    } else {
      formData.append("speed", vezoSpeed.toString());
      formData.append("sr", vezoSr.toString());
      formData.append("noise_scale", vezoNoiseScale.toString());
      formData.append("noise_scale_duration", vezoNoiseScaleDuration.toString());
      formData.append("highpass_cutoff", vezoHighpassCutoff.toString());
      formData.append("noise_gate_threshold", vezoNoiseGate.toString());
      formData.append("peak_norm", vezoPeakNorm.toString());
      formData.append("warmth_level", vezoWarmth.toString());
    }

    try {
      const res = await fetch(`${API}/speech-to-vezo`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Erreur de traitement du serveur");
      }

      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setTranscription(data.transcription || "");
        setTranslation(data.translation || "");
        setVezoAudioUrl(data.vezo_audio_url || data.audio_url || "");
      }
    } catch (err: any) {
      console.error(err);
      setError("Échec de la reconnaissance vocale : " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRegenerateAudio = async () => {
    if (!translation.trim()) return;
    setIsProcessing(true);
    try {
      if (direction === "english-to-merina") {
        const res = await fetch(`${API}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: translation,
            speed: merinaSpeed,
            sr: merinaSr,
            highpass_cutoff: merinaHighpassCutoff
          }),
        });
        const data = await res.json();
        setVezoAudioUrl(data.audio_url);
      } else {
        const res = await fetch(`${API}/tts-vezo`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: translation,
            speed: vezoSpeed,
            sr: vezoSr,
            noise_scale: vezoNoiseScale,
            noise_scale_duration: vezoNoiseScaleDuration,
            highpass_cutoff: vezoHighpassCutoff,
            noise_gate_threshold: vezoNoiseGate,
            peak_norm: vezoPeakNorm,
            warmth_level: vezoWarmth,
          }),
        });
        const data = await res.json();
        setVezoAudioUrl(data.audio_url);
      }
    } catch (err: any) {
      console.error(err);
      setError("Erreur de régénération audio : " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-6">
      {/* ── Selection de direction ── */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 bg-slate-900/40 p-4 rounded-3xl border border-slate-800">
        <div>
          <h2 className="font-semibold text-sm text-indigo-300">🎙️ Reconnaissance Vocale & Traduction Vocale (Anglais)</h2>
          <p className="text-xs text-slate-400">Parlez en Anglais, transcrivez par IA Whisper et générez le son en Vezo ou Malagasy Officiel (Merina).</p>
        </div>

        <div className="inline-flex items-center bg-slate-950/80 p-1.5 rounded-2xl border border-slate-800 gap-1 flex-wrap">
          <button
            onClick={() => setDirection("english-to-vezo")}
            className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all ${direction === "english-to-vezo" ? "bg-teal-600 text-white shadow-lg shadow-teal-600/20" : "text-slate-400 hover:text-slate-200"}`}
          >
            🌊 EN ➔ Vezo
          </button>
          <button
            onClick={() => setDirection("english-to-merina")}
            className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all ${direction === "english-to-merina" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-slate-400 hover:text-slate-200"}`}
          >
            🇲🇬 EN ➔ Malagasy Officiel
          </button>
          <button
            onClick={() => setDirection("english-to-betsileo")}
            className={`px-4 py-2 rounded-xl font-semibold text-xs transition-all ${direction === "english-to-betsileo" ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" : "text-slate-400 hover:text-slate-200"}`}
          >
            ⛰️ EN ➔ Betsileo
          </button>
        </div>
      </div>

      {/* ── Réglages Audio TTS Collapsible ── */}
      <div className="bg-slate-900/60 p-4 rounded-3xl border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-teal-300 flex items-center gap-1.5">
            ⚙️ Paramètres Audio (
            {direction === "english-to-merina"
              ? `Merina: ${merinaSpeed}x - ${merinaSr}Hz`
              : `Vezo: ${vezoSpeed}x - ${vezoSr}Hz - Anti-Bruit Active`}
            )
          </span>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="text-xs text-indigo-400 hover:text-indigo-300 font-medium underline"
          >
            {showSettings ? "Masquer Réglages" : "Ajuster les Réglages Son"}
          </button>
        </div>

        {showSettings && (
          <div className="pt-3 border-t border-slate-800 text-xs space-y-4">
            {direction === "english-to-merina" ? (
              /* Merina Settings */
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="text-slate-400 block mb-1">Vitesse ({merinaSpeed}x)</label>
                  <input
                    type="range" min="0.5" max="2.0" step="0.05"
                    value={merinaSpeed} onChange={(e) => setMerinaSpeed(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Fréquence ({merinaSr} Hz)</label>
                  <div className="flex gap-1">
                    {[16000, 22050, 24000].map((sr) => (
                      <button
                        key={sr}
                        onClick={() => setMerinaSr(sr)}
                        className={`flex-1 py-1 text-[10px] rounded border font-semibold ${
                          merinaSr === sr ? "bg-indigo-600 text-white border-indigo-500" : "bg-slate-950 text-slate-400 border-slate-800"
                        }`}
                      >
                        {sr / 1000}k
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Filtre Passe-Haut</label>
                  <div className="flex gap-1">
                    {[0, 60, 100].map((hp) => (
                      <button
                        key={hp}
                        onClick={() => setMerinaHighpassCutoff(hp)}
                        className={`flex-1 py-1 text-[10px] rounded border font-semibold ${
                          merinaHighpassCutoff === hp ? "bg-indigo-600 text-white border-indigo-500" : "bg-slate-950 text-slate-400 border-slate-800"
                        }`}
                      >
                        {hp === 0 ? "Off" : `${hp}Hz`}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              /* Vezo Anti-Noise Settings */
              <div className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="text-slate-400 block mb-1">Vitesse ({vezoSpeed}x)</label>
                    <input
                      type="range" min="0.5" max="2.0" step="0.05"
                      value={vezoSpeed} onChange={(e) => setVezoSpeed(parseFloat(e.target.value))}
                      className="w-full accent-teal-500 cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Fréquence Sampling (Hz)</label>
                    <div className="flex gap-1">
                      {[16000, 22050, 24000, 32000, 44100].map((sr) => (
                        <button
                          key={sr}
                          onClick={() => setVezoSr(sr)}
                          className={`flex-1 py-1 text-[10px] rounded border font-semibold ${
                            vezoSr === sr ? "bg-teal-600 text-white border-teal-500" : "bg-slate-950 text-slate-400 border-slate-800"
                          }`}
                        >
                          {sr >= 1000 ? `${sr / 1000}k` : sr}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Noise Scale ({vezoNoiseScale})</label>
                    <input
                      type="range" min="0.05" max="1.0" step="0.05"
                      value={vezoNoiseScale} onChange={(e) => setVezoNoiseScale(parseFloat(e.target.value))}
                      className="w-full accent-teal-500 cursor-pointer"
                    />
                    <span className="text-[9px] text-slate-500 block">Faible = réduction grésillement</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-2 border-t border-slate-800/60">
                  <div>
                    <label className="text-slate-400 block mb-1">Duration Noise ({vezoNoiseScaleDuration})</label>
                    <input
                      type="range" min="0.1" max="1.0" step="0.05"
                      value={vezoNoiseScaleDuration} onChange={(e) => setVezoNoiseScaleDuration(parseFloat(e.target.value))}
                      className="w-full accent-teal-500 cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Filtre Anti-Ronflement</label>
                    <div className="flex gap-1 flex-wrap">
                      {[0, 40, 60, 80, 100, 150].map((hp) => (
                        <button
                          key={hp}
                          onClick={() => setVezoHighpassCutoff(hp)}
                          className={`flex-1 py-1 text-[9px] rounded border font-semibold ${
                            vezoHighpassCutoff === hp ? "bg-teal-600 text-white border-teal-500" : "bg-slate-950 text-slate-400 border-slate-800"
                          }`}
                        >
                          {hp === 0 ? "Off" : `${hp}Hz`}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Noise Gate (Souffle)</label>
                    <div className="flex gap-1 flex-wrap">
                      {[
                        { label: "Off", val: 0.0 },
                        { label: "Léger", val: 0.005 },
                        { label: "Moyen", val: 0.01 },
                        { label: "Fort", val: 0.02 }
                      ].map((g) => (
                        <button
                          key={g.label}
                          onClick={() => setVezoNoiseGate(g.val)}
                          className={`flex-1 py-1 text-[9px] rounded border font-semibold ${
                            vezoNoiseGate === g.val ? "bg-teal-600 text-white border-teal-500" : "bg-slate-950 text-slate-400 border-slate-800"
                          }`}
                        >
                          {g.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Peak Norm ({vezoPeakNorm})</label>
                    <input
                      type="range" min="0.70" max="1.00" step="0.05"
                      value={vezoPeakNorm} onChange={(e) => setVezoPeakNorm(parseFloat(e.target.value))}
                      className="w-full accent-teal-500 cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Chaleur/Anti-Métal ({vezoWarmth})</label>
                    <input
                      type="range" min="0.0" max="1.0" step="0.1"
                      value={vezoWarmth} onChange={(e) => setVezoWarmth(parseFloat(e.target.value))}
                      className="w-full accent-teal-500 cursor-pointer"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Zone de Capture Microphone ── */}
      <div className="bg-slate-900/50 rounded-3xl border border-slate-800 p-8 flex flex-col items-center justify-center text-center shadow-2xl relative overflow-hidden backdrop-blur-xl">
        {isRecording && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-48 h-48 rounded-full bg-rose-500/10 animate-ping" />
            <div className="w-32 h-32 rounded-full bg-rose-500/20 animate-pulse" />
          </div>
        )}

        <div className="relative z-10 space-y-6">
          <div className="inline-flex items-center justify-center">
            {!isRecording ? (
              <button
                onClick={startRecording}
                disabled={isProcessing}
                className="w-24 h-24 rounded-full bg-gradient-to-tr from-rose-600 via-rose-500 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white shadow-2xl shadow-rose-600/40 flex items-center justify-center transition-all duration-300 transform hover:scale-105 active:scale-95 disabled:opacity-40"
                title="Démarrer l'enregistrement"
              >
                <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="w-24 h-24 rounded-full bg-rose-600 text-white shadow-2xl shadow-rose-600/60 flex items-center justify-center animate-pulse transition-all duration-300 transform hover:scale-105 active:scale-95"
                title="Arrêter la capture"
              >
                <div className="w-8 h-8 rounded-md bg-white" />
              </button>
            )}
          </div>

          <div>
            <h3 className="text-lg font-bold text-slate-100">
              {isRecording ? "Enregistrement en cours..." : isProcessing ? "Traitement Génération Audio..." : "Cliquez sur le micro pour parler en Anglais"}
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              {isRecording ? `Durée : ${formatTime(recordingTime)}` : "Whisper transcrira votre voix anglaise puis générera la traduction et le son synthétisé."}
            </p>
          </div>

          {isProcessing && (
            <div className="flex items-center justify-center gap-2 text-indigo-400 text-sm animate-pulse">
              <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Reconnaissance Vocale & Traduction en cours...
            </div>
          )}

          {error && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/60 text-rose-300 rounded-xl text-xs max-w-md mx-auto">
              ⚠️ {error}
            </div>
          )}
        </div>
      </div>

      {/* ── Résultats : Transcription & Traduction ── */}
      {(transcription || translation) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Transcription Anglais */}
          <div className="bg-slate-900/50 rounded-3xl border border-slate-800 p-6 flex flex-col justify-between backdrop-blur-xl">
            <div>
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                <span className="font-semibold text-xs tracking-wider text-indigo-400 uppercase flex items-center gap-2">
                  🇬🇧 Transcription Vocale (Whisper EN)
                </span>
                <button
                  onClick={() => navigator.clipboard.writeText(transcription)}
                  className="text-xs text-slate-500 hover:text-slate-300"
                  title="Copier"
                >
                  📋 Copier
                </button>
              </div>
              <p className="text-slate-100 text-lg leading-relaxed">{transcription}</p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/40 text-xs text-slate-500">
              Speech-to-Text Whisper AI (English)
            </div>
          </div>

          {/* Traduction Vezo / Merina / Dialecte */}
          <div className="bg-slate-900/50 rounded-3xl border border-slate-800 p-6 flex flex-col justify-between backdrop-blur-xl">
            <div>
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                <span className={`font-semibold text-xs tracking-wider uppercase flex items-center gap-2 ${direction === "english-to-vezo" ? "text-teal-400" : "text-indigo-400"}`}>
                  {direction === "english-to-vezo" ? "🌊 Traduction Vezo (Fine-Tunée)" : direction === "english-to-merina" ? "🇲🇬 Traduction Malagasy Officiel" : "⛰️ Traduction Betsileo"}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleRegenerateAudio}
                    disabled={isProcessing}
                    className="text-xs text-teal-400 hover:text-teal-300 font-semibold flex items-center gap-1 bg-teal-950/40 border border-teal-800/50 px-2 py-1 rounded-lg"
                    title="Régénérer le son avec les paramètres actuels"
                  >
                    🔄 Régénérer Son
                  </button>
                  <button
                    onClick={() => navigator.clipboard.writeText(translation)}
                    className="text-xs text-slate-500 hover:text-slate-300"
                    title="Copier"
                  >
                    📋 Copier
                  </button>
                </div>
              </div>
              <p className="text-slate-100 text-lg leading-relaxed">{translation}</p>

              {vezoAudioUrl && (
                <div className="mt-4 p-3 bg-teal-950/40 border border-teal-800/60 rounded-2xl flex flex-col gap-2">
                  <div className="flex items-center justify-between text-xs text-teal-300 font-semibold">
                    <span>
                      🔊 Voix Synthèse (
                      {direction === "english-to-merina"
                        ? `Malagasy Officiel MMS-TTS ${merinaSr}Hz - ${merinaSpeed}x`
                        : `Vezo MMS-TTS ${vezoSr}Hz - ${vezoSpeed}x`}
                      )
                    </span>
                    <button
                      onClick={handleRegenerateAudio}
                      className="text-[10px] text-teal-400 bg-teal-900/60 px-2 py-0.5 rounded-full hover:bg-teal-800"
                    >
                      🔄 Réajuster & Régénérer
                    </button>
                  </div>
                  <audio controls autoPlay src={`${API}${vezoAudioUrl}`} className="w-full h-8" />
                </div>
              )}
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/40 flex items-center justify-between text-xs text-slate-500">
              <span>{direction === "english-to-merina" ? "Pipeline : Anglais ➔ Malagasy Officiel (Merina)" : "Pipeline : Anglais ➔ Merina ➔ Vezo"}</span>
              <span className="bg-teal-950 text-teal-300 px-2 py-0.5 rounded-full border border-teal-800/40 text-[10px]">
                {direction === "english-to-merina" ? "Official Malagasy AI" : "Vezo Dialect AI"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
