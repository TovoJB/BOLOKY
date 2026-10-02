"use client";

import React, { useState, useEffect, useRef } from "react";
import TokenCloud3D from "./components/TokenCloud3D";
import TTSTab2 from "./components/TTSTab2";
import MediaTab from "./components/MediaTab";
import TTSTab from "./components/TTSTab";
import SpeechTab from "./components/SpeechTab";

export default function Home() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [direction, setDirection] = useState<"merina-to-betsileo" | "betsileo-to-merina" | "merina-to-vezo" | "english-to-vezo" | "english-to-merina">("merina-to-betsileo");
  const [selectedModel, setSelectedModel] = useState<"model1" | "model1_2" | "model2" | "model3" | "vezo">("model1");
  const [modelTokens, setModelTokens] = useState<string[]>([]);
  const [generationDetails, setGenerationDetails] = useState<Array<{
    token: string;
    prob: number;
    alternatives: Array<{ token: string; prob: number }>;
  }>>([]);
  const [history, setHistory] = useState<Array<{ from: string; to: string; text: string; translation: string }>>([]);
  const [constraintsApplied, setConstraintsApplied] = useState(false);
  const [forcedWords, setForcedWords] = useState<string[]>([]);
  const [view3D, setView3D] = useState(false);

  // Mode switcher: text | speech | media | tts | tts2
  const [appMode, setAppMode] = useState<"text" | "speech" | "media" | "tts" | "tts2">("text");
  const [ttsPreFill, setTtsPreFill] = useState("");
  const [vezoAudioUrl, setVezoAudioUrl] = useState("");
  const [isVezoTtsLoading, setIsVezoTtsLoading] = useState(false);

  // Audio parameters for Vezo TTS (Session state)
  const [vezoSpeed, setVezoSpeed] = useState(1.15);
  const [vezoSr, setVezoSr] = useState(22050);
  const [vezoNoiseScale, setVezoNoiseScale] = useState(0.35);
  const [vezoNoiseScaleDuration, setVezoNoiseScaleDuration] = useState(0.6);
  const [vezoHighpassCutoff, setVezoHighpassCutoff] = useState(60);
  const [vezoNoiseGate, setVezoNoiseGate] = useState(0.01);
  const [vezoPeakNorm, setVezoPeakNorm] = useState(0.95);
  const [vezoWarmth, setVezoWarmth] = useState(0.5);

  // Audio parameters for Official Malagasy (Merina) TTS (Session state)
  const [merinaSpeed, setMerinaSpeed] = useState(1.0);
  const [merinaSr, setMerinaSr] = useState(22050);
  const [merinaHighpassCutoff, setMerinaHighpassCutoff] = useState(60);

  const [showTtsSettings, setShowTtsSettings] = useState(false);

  const handleVezoTts = async (text: string) => {
    if (!text.trim()) return;
    setIsVezoTtsLoading(true);
    try {
      if (direction === "english-to-merina") {
        const res = await fetch("http://localhost:8000/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            speed: merinaSpeed,
            sr: merinaSr,
            highpass_cutoff: merinaHighpassCutoff
          }),
        });
        if (!res.ok) throw new Error("Erreur de synthèse vocale Merina");
        const data = await res.json();
        setVezoAudioUrl(data.audio_url);
      } else {
        const res = await fetch("http://localhost:8000/tts-vezo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            speed: vezoSpeed,
            sr: vezoSr,
            noise_scale: vezoNoiseScale,
            noise_scale_duration: vezoNoiseScaleDuration,
            highpass_cutoff: vezoHighpassCutoff,
            noise_gate_threshold: vezoNoiseGate,
            peak_norm: vezoPeakNorm,
            warmth_level: vezoWarmth
          }),
        });
        if (!res.ok) throw new Error("Erreur de synthèse vocale Vezo");
        const data = await res.json();
        setVezoAudioUrl(data.audio_url);
      }
    } catch (err) {
      console.error("TTS error:", err);
    } finally {
      setIsVezoTtsLoading(false);
    }
  };

  // Tokenizer State
  const [tokenizerInput, setTokenizerInput] = useState("");
  const [tokenizerTokens, setTokenizerTokens] = useState<string[]>([]);
  const [isTokenizing, setIsTokenizing] = useState(false);

  // Check backend server status
  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch("http://localhost:8000/", { method: "GET" }).catch(() => null);
        // FastAPI returns 404 or success on root depending on route, but if we get any response, it's alive.
        if (res) {
          setApiStatus("online");
        } else {
          setApiStatus("offline");
        }
      } catch {
        setApiStatus("offline");
      }
    };
    checkApi();
    const interval = setInterval(checkApi, 10000);
    return () => clearInterval(interval);
  }, []);

  const abortControllerRef = useRef<AbortController | null>(null);

  const handleTranslate = async (textToTranslate = inputText, modelToUse = selectedModel) => {
    const trimmed = textToTranslate.trim();
    if (!trimmed) {
      setOutputText("");
      setModelTokens([]);
      setGenerationDetails([]);
      setConstraintsApplied(false);
      setForcedWords([]);
      return;
    }

    // Annuler la requête précédente si elle est encore en cours
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsTranslating(true);
    try {
      const response = await fetch("http://localhost:8000/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: trimmed, model: modelToUse, direction }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error("Translation failed");
      }

      const data = await response.json();
      if (controller.signal.aborted) return;

      const translationResult = data.translation;
      setOutputText(translationResult);
      if (data.tokens) {
        setModelTokens(data.tokens);
      } else {
        setModelTokens([]);
      }
      if (data.generation_details) {
        setGenerationDetails(data.generation_details);
      } else {
        setGenerationDetails([]);
      }
      setConstraintsApplied(data.constraints_applied || false);
      setForcedWords(data.forced_words || []);

      // Ajouter à l'historique de manière intelligente (pas à chaque touche pressée)
      const wordsCount = trimmed.split(/\s+/).length;
      if (wordsCount >= 3) {
        setHistory((prev) => {
          const alreadyExists = prev.some(
            (h) => h.text.toLowerCase() === trimmed.toLowerCase() && h.translation === translationResult
          );
          if (alreadyExists) return prev;
          return [
            {
              from: direction === "merina-to-betsileo" ? "Merina" : direction === "merina-to-vezo" ? "Merina" : direction === "english-to-vezo" ? "English" : direction === "english-to-merina" ? "English" : "Betsileo",
              to: direction === "merina-to-betsileo" ? "Betsileo" : direction === "merina-to-vezo" || direction === "english-to-vezo" ? "Vezo" : direction === "english-to-merina" ? "Merina" : "Merina",
              text: trimmed,
              translation: translationResult,
            },
            ...prev.slice(0, 4), // Garder les 5 derniers
          ];
        });
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        return; // Requête annulée, rien à faire
      }
      console.error(err);
      setOutputText("Erreur: Impossible de contacter le serveur de traduction.");
    } finally {
      if (!controller.signal.aborted) {
        setIsTranslating(false);
      }
    }
  };

  // Traduction en temps réel au fur et à mesure de la saisie
  useEffect(() => {
    const trimmed = inputText.trim();
    if (!trimmed) {
      setOutputText("");
      setModelTokens([]);
      setGenerationDetails([]);
      setConstraintsApplied(false);
      setForcedWords([]);
      return;
    }

    const timer = setTimeout(() => {
      handleTranslate(inputText, selectedModel);
    }, 400); // 400ms de délai pour ne pas saturer l'API pendant la saisie

    return () => {
      clearTimeout(timer);
    };
  }, [inputText, selectedModel]);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const handleTokenize = async () => {
    if (!tokenizerInput.trim()) return;
    setIsTokenizing(true);
    try {
      const response = await fetch("http://localhost:8000/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: tokenizerInput }),
      });
      if (response.ok) {
        const data = await response.json();
        setTokenizerTokens(data.tokens);
      }
    } catch (err) {
      console.error(err);
      setTokenizerTokens(["Erreur de connexion au serveur"]);
    } finally {
      setIsTokenizing(false);
    }
  };

  const handleClear = () => {
    setInputText("");
    setOutputText("");
    setModelTokens([]);
    setGenerationDetails([]);
    setConstraintsApplied(false);
    setForcedWords([]);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between selection:bg-indigo-500 selection:text-white relative overflow-hidden">
      {/* Background Orbs */}
      <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] rounded-full bg-indigo-500/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-purple-500/10 blur-[120px] pointer-events-none" />

      {/* Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/50 backdrop-blur-md px-6 py-4 relative z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">

            {/* Votre Logo à la place du div avec le "M" */}
            <img
              src="/logo.png"
              alt="Logo"
              className="w-10 h-10 object-contain"
            />

            <div>
              <h1 className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 via-indigo-100 to-purple-200">
                Malagasy Dialect Translator
              </h1>
              <p className="text-xs text-slate-400">Traducteur de Dialectes Malagasy</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <span className="flex items-center space-x-2 text-xs bg-slate-800/60 px-3 py-1.5 rounded-full border border-slate-700/50">
              <span className={`w-2.5 h-2.5 rounded-full ${apiStatus === "online" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : apiStatus === "offline" ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]" : "bg-amber-500 animate-pulse"}`} />
              <span className="text-slate-300 font-medium">
                {apiStatus === "online" ? "Serveur Actif (Port 8000)" : apiStatus === "offline" ? "Serveur Inactif" : "Vérification..."}
              </span>
            </span>
          </div>
        </div>
      </header>

      {/* Content */}
      <section className="flex-grow max-w-7xl w-full mx-auto px-4 md:px-6 py-8 relative z-10 flex flex-col justify-center">
        {/* ── Mode Tabs ── */}
        <div className="flex items-center justify-center mb-8">
          <div className="inline-flex items-center bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800/80 shadow-2xl backdrop-blur-sm gap-1 flex-wrap justify-center">
            {(["text", "speech", "media", "tts", "tts2"] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setAppMode(mode)}
                className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${appMode === mode
                  ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                  : "text-slate-400 hover:text-slate-200"
                  }`}
              >
                {mode === "text" ? "📝 Texte" : mode === "speech" ? "🎙️ Micro Temps Réel" : mode === "media" ? "🎬 Vidéo / Audio" : mode === "tts" ? "🔊 TTS" : "🗣️ TTS2"}
              </button>
            ))}
          </div>
        </div>
        {/* ── Text Mode ── */}
        {appMode === "text" && (<>
          {/* Direction & Model Switchers */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 mb-6 flex-wrap">
            {/* Direction Switcher */}
            <div className="inline-flex items-center bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800/80 shadow-2xl backdrop-blur-sm flex-wrap gap-1">
              <button
                onClick={() => { setDirection("merina-to-betsileo"); if (selectedModel === "vezo") setSelectedModel("model1"); }}
                className={`px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${direction === "merina-to-betsileo" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                Merina ➔ Betsileo
              </button>
              <button
                onClick={() => { setDirection("betsileo-to-merina"); if (selectedModel === "vezo") setSelectedModel("model1"); }}
                className={`px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${direction === "betsileo-to-merina" ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                Betsileo ➔ Merina
              </button>
              <button
                onClick={() => { setDirection("merina-to-vezo"); setSelectedModel("vezo"); }}
                className={`px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${direction === "merina-to-vezo" ? "bg-teal-600 text-white shadow-lg shadow-teal-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                🌊 Merina ➔ Vezo
              </button>
              <button
                onClick={() => { setDirection("english-to-vezo"); setSelectedModel("vezo"); }}
                className={`px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${direction === "english-to-vezo" ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                🇬🇧 Anglais ➔ 🌊 Vezo
              </button>
              <button
                onClick={() => { setDirection("english-to-merina"); if (selectedModel === "vezo") setSelectedModel("model1"); }}
                className={`px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${direction === "english-to-merina" ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                🇬🇧 Anglais ➔ 🇲🇬 Malagasy Officiel
              </button>
            </div>

            {/* Model Switcher — hidden when Vezo or English directions are active */}
            {direction !== "merina-to-vezo" && direction !== "english-to-vezo" && direction !== "english-to-merina" && (
            <div className="inline-flex items-center bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800/80 shadow-2xl backdrop-blur-sm">
              <button
                onClick={() => setSelectedModel("model1")}
                className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${selectedModel === "model1" ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                Modèle 1
              </button>
              <button
                onClick={() => setSelectedModel("model1_2")}
                className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${selectedModel === "model1_2" ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                Modèle 1.2
              </button>
              <button
                onClick={() => setSelectedModel("model2")}
                className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${selectedModel === "model2" ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                Modèle 2
              </button>
              <button
                onClick={() => setSelectedModel("model3")}
                className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${selectedModel === "model3" ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20" : "text-slate-400 hover:text-slate-200"}`}
              >
                Modèle 2.0
              </button>
            </div>
            )}
            {(direction === "merina-to-vezo" || direction === "english-to-vezo") && (
              <div className="inline-flex items-center bg-teal-900/40 border border-teal-700/50 px-4 py-2.5 rounded-2xl text-sm text-teal-300 font-semibold shadow backdrop-blur-sm">
                🌊 Modèle Vezo
              </div>
            )}
            {direction === "english-to-merina" && (
              <div className="inline-flex items-center bg-purple-900/40 border border-purple-700/50 px-4 py-2.5 rounded-2xl text-sm text-purple-300 font-semibold shadow backdrop-blur-sm">
                🇲🇬 Modèle Malagasy Officiel (Helsinki Opus-MT)
              </div>
            )}
          </div>

          {/* Main Workspace Translation Box */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Input Box */}
            <div className="group bg-slate-900/40 hover:bg-slate-900/60 rounded-3xl border border-slate-850 hover:border-slate-800 transition-all duration-300 flex flex-col justify-between overflow-hidden backdrop-blur-xl shadow-xl">
              <div className="p-5 border-b border-slate-800/60 flex items-center justify-between">
                <span className="font-semibold text-sm tracking-wider text-indigo-400 uppercase">
                  {direction === "english-to-vezo" || direction === "english-to-merina" ? "🇬🇧 English" : direction === "merina-to-betsileo" ? "Teny Merina (Standard)" : direction === "merina-to-vezo" ? "🌊 Teny Merina (Vezo)" : "Teny Betsileo"}
                </span>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setAppMode("speech")}
                    className="p-1.5 rounded-lg text-rose-400 hover:text-rose-300 hover:bg-rose-950/40 transition-all flex items-center gap-1 text-xs font-semibold"
                    title="Parler au microphone (Speech-to-Text Anglais -> Vezo/Merina)"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    Micro
                  </button>
                  <button
                    onClick={handleClear}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 transition-all"
                    title="Effacer le texte"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>

              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Soraty eto ny andinin-tsoratra masina..."
                className="w-full h-64 p-6 bg-transparent text-slate-100 placeholder-slate-500 focus:outline-none resize-none text-lg leading-relaxed border-none focus:ring-0"
                maxLength={1500}
              />

              <div className="p-4 bg-slate-900/30 border-t border-slate-800/30 flex items-center justify-between text-xs text-slate-500">
                <span>{inputText.length} / 1500 caractères</span>
                <div className="flex items-center gap-3">
                  {isTranslating && (
                    <span className="flex items-center gap-1.5 text-indigo-400 text-xs animate-pulse">
                      <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Traduction…
                    </span>
                  )}
                  {!isTranslating && outputText && (
                    <span className="flex items-center gap-1.5 text-emerald-400 text-xs">
                      <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                      Traduit
                    </span>
                  )}
                  <button
                    onClick={() => handleTranslate(inputText, selectedModel)}
                    disabled={isTranslating || !inputText.trim()}
                    className="px-4 py-2 rounded-xl bg-indigo-600/20 hover:bg-indigo-600/40 border border-indigo-800/40 hover:border-indigo-600/60 disabled:opacity-30 text-indigo-300 font-semibold flex items-center gap-2 transition-all duration-200 active:scale-95"
                    title="Forcer la re-traduction"
                  >
                    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Relancer
                  </button>
                </div>
              </div>
            </div>

            {/* Output Box */}
            <div className="bg-slate-900/40 rounded-3xl border border-slate-850 flex flex-col justify-between overflow-hidden backdrop-blur-xl shadow-xl">
              <div className="p-5 border-b border-slate-800/60 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className={`font-semibold text-sm tracking-wider uppercase ${direction === "merina-to-vezo" || direction === "english-to-vezo" ? "text-teal-400" : direction === "english-to-merina" ? "text-purple-400" : "text-purple-400"}`}>
                    {direction === "merina-to-betsileo" ? "Teny Betsileo" : direction === "merina-to-vezo" || direction === "english-to-vezo" ? "🌊 Teny Vezo" : "🇲🇬 Malagasy Officiel (Merina)"}
                  </span>
                  <span className={`flex items-center gap-1.5 text-[10px] font-bold px-2 py-0.5 rounded-full border transition-all duration-300 ${isTranslating
                    ? "text-amber-300 bg-amber-950/30 border-amber-700/40 animate-pulse"
                    : inputText.trim()
                      ? "text-emerald-300 bg-emerald-950/30 border-emerald-700/40"
                      : "text-slate-500 bg-slate-900/30 border-slate-800/40"
                    }`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${isTranslating ? "bg-amber-400 animate-ping" : inputText.trim() ? "bg-emerald-400" : "bg-slate-600"
                      }`} />
                    {isTranslating ? "EN COURS" : inputText.trim() ? "LIVE" : "EN ATTENTE"}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  {(direction === "merina-to-vezo" || direction === "english-to-vezo" || direction === "english-to-merina") && (
                    <>
                      <button
                        onClick={() => handleVezoTts(outputText)}
                        disabled={!outputText || isVezoTtsLoading}
                        className={`px-2.5 py-1 rounded-lg border text-xs font-semibold flex items-center gap-1.5 transition-all disabled:opacity-40 ${
                          direction === "english-to-merina"
                            ? "bg-purple-950/60 hover:bg-purple-900/80 border-purple-700/50 text-purple-300"
                            : "bg-teal-950/60 hover:bg-teal-900/80 border-teal-700/50 text-teal-300"
                        }`}
                        title="Synthétiser le texte en voix"
                      >
                        {isVezoTtsLoading ? (
                          <span className="flex items-center gap-1">
                            <svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Synthèse...
                          </span>
                        ) : (
                          <>{direction === "english-to-merina" ? "🇲🇬 Voix Merina" : "🌊 Voix Vezo"}</>
                        )}
                      </button>

                      <button
                        onClick={() => setShowTtsSettings(!showTtsSettings)}
                        className={`p-1.5 rounded-lg border transition-all text-xs flex items-center gap-1 font-semibold ${
                          showTtsSettings
                            ? "bg-indigo-600 text-white border-indigo-500"
                            : "text-slate-400 hover:text-slate-200 bg-slate-800/60 border-slate-700/50"
                        }`}
                        title="Réglages Audio TTS"
                      >
                        ⚙️ Réglages
                      </button>
                    </>
                  )}
                  <button
                    onClick={() => { setTtsPreFill(outputText); setAppMode("tts"); }}
                    disabled={!outputText}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-emerald-400 hover:bg-emerald-950/30 transition-all disabled:opacity-40"
                    title="Écouter via TTS Standard"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M12 6a7 7 0 010 12M9 9a3 3 0 000 6" />
                    </svg>
                  </button>
                  <button
                    onClick={() => handleCopy(outputText)}
                    disabled={!outputText}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 transition-all disabled:opacity-40"
                    title="Copier le texte"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* TTS Settings Collapsible Panel */}
              {showTtsSettings && (
                <div className="p-4 bg-slate-950/90 border-b border-slate-800/80 space-y-4 text-xs animate-fade-in-up">
                  {/* Panel Header */}
                  <div className="flex items-center justify-between">
                    <span className="font-semibold flex items-center gap-1.5 text-slate-200">
                      {direction === "english-to-merina" ? (
                        <span className="text-purple-300">⚙️ Configuration Vocale (Malagasy Officiel Merina)</span>
                      ) : (
                        <span className="text-teal-300">⚙️ Configuration Vocale & Anti-Bruit (Dialecte Vezo)</span>
                      )}
                    </span>
                    <button
                      onClick={() => {
                        if (direction === "english-to-merina") {
                          setMerinaSpeed(1.0);
                          setMerinaSr(22050);
                          setMerinaHighpassCutoff(60);
                        } else {
                          setVezoSpeed(1.15);
                          setVezoSr(22050);
                          setVezoNoiseScale(0.35);
                          setVezoNoiseScaleDuration(0.6);
                          setVezoHighpassCutoff(60);
                          setVezoNoiseGate(0.01);
                          setVezoPeakNorm(0.95);
                          setVezoWarmth(0.5);
                        }
                      }}
                      className="text-[10px] text-slate-400 hover:text-slate-200 underline"
                    >
                      Réinitialiser
                    </button>
                  </div>

                  {/* 🇲🇬 Merina Configuration */}
                  {direction === "english-to-merina" ? (
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {/* Vitesse */}
                      <div>
                        <label className="block text-[11px] text-slate-400 mb-1">
                          Vitesse: <span className="text-purple-300 font-mono font-bold">{merinaSpeed}x</span>
                        </label>
                        <input
                          type="range"
                          min="0.5"
                          max="2.0"
                          step="0.05"
                          value={merinaSpeed}
                          onChange={(e) => setMerinaSpeed(parseFloat(e.target.value))}
                          className="w-full accent-purple-500 bg-slate-800 rounded cursor-pointer"
                        />
                      </div>

                      {/* Fréquence */}
                      <div>
                        <label className="block text-[11px] text-slate-400 mb-1">Fréquence (Hz)</label>
                        <div className="flex gap-1">
                          {[16000, 22050, 24000].map((sr) => (
                            <button
                              key={sr}
                              onClick={() => setMerinaSr(sr)}
                              className={`flex-1 py-1 text-[10px] rounded border font-semibold ${
                                merinaSr === sr
                                  ? "bg-purple-600 text-white border-purple-500"
                                  : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                              }`}
                            >
                              {sr / 1000}k
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Filtre Passe-Haut */}
                      <div>
                        <label className="block text-[11px] text-slate-400 mb-1">Filtre Passe-Haut</label>
                        <div className="flex gap-1">
                          {[0, 60, 100].map((hp) => (
                            <button
                              key={hp}
                              onClick={() => setMerinaHighpassCutoff(hp)}
                              className={`flex-1 py-1 text-[10px] rounded border font-semibold ${
                                merinaHighpassCutoff === hp
                                  ? "bg-purple-600 text-white border-purple-500"
                                  : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                              }`}
                            >
                              {hp === 0 ? "Off" : `${hp}Hz`}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* 🌊 Vezo Extended Anti-Noise Configuration */
                    <div className="space-y-3">
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        {/* Vitesse */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1">
                            Vitesse: <span className="text-teal-300 font-mono font-bold">{vezoSpeed}x</span>
                          </label>
                          <input
                            type="range"
                            min="0.5"
                            max="2.0"
                            step="0.05"
                            value={vezoSpeed}
                            onChange={(e) => setVezoSpeed(parseFloat(e.target.value))}
                            className="w-full accent-teal-500 bg-slate-800 rounded cursor-pointer"
                          />
                        </div>

                        {/* Fréquence */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1">Fréquence Sampling (Hz)</label>
                          <div className="flex gap-1">
                            {[16000, 22050, 24000, 32000, 44100].map((sr) => (
                              <button
                                key={sr}
                                onClick={() => setVezoSr(sr)}
                                className={`flex-1 py-1 text-[10px] rounded border font-semibold ${
                                  vezoSr === sr
                                    ? "bg-teal-600 text-white border-teal-500"
                                    : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                                }`}
                              >
                                {sr >= 1000 ? `${sr / 1000}k` : sr}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Noise Scale (Bruit de synthèse/Timbre) */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1 flex justify-between">
                            <span>Noise Scale:</span>
                            <span className="text-teal-300 font-mono font-bold">{vezoNoiseScale}</span>
                          </label>
                          <input
                            type="range"
                            min="0.05"
                            max="1.0"
                            step="0.05"
                            value={vezoNoiseScale}
                            onChange={(e) => setVezoNoiseScale(parseFloat(e.target.value))}
                            className="w-full accent-teal-500 bg-slate-800 rounded cursor-pointer"
                          />
                          <span className="text-[9px] text-slate-500 block">Faible = réduction du grésillement</span>
                        </div>
                      </div>

                      {/* Second Row: Advanced Noise Correction */}
                      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-2 border-t border-slate-800/60">
                        {/* Duration Noise Scale */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1 flex justify-between">
                            <span>Variabilité Rythme:</span>
                            <span className="text-teal-300 font-mono font-bold">{vezoNoiseScaleDuration}</span>
                          </label>
                          <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            value={vezoNoiseScaleDuration}
                            onChange={(e) => setVezoNoiseScaleDuration(parseFloat(e.target.value))}
                            className="w-full accent-teal-500 bg-slate-800 rounded cursor-pointer"
                          />
                        </div>

                        {/* Highpass Cutoff */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1">Filtre Anti-Ronflement</label>
                          <div className="flex gap-1 flex-wrap">
                            {[0, 40, 60, 80, 100, 150].map((hp) => (
                              <button
                                key={hp}
                                onClick={() => setVezoHighpassCutoff(hp)}
                                className={`flex-1 py-1 text-[9px] rounded border font-semibold ${
                                  vezoHighpassCutoff === hp
                                    ? "bg-teal-600 text-white border-teal-500"
                                    : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                                }`}
                              >
                                {hp === 0 ? "Off" : `${hp}Hz`}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Noise Gate */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1">Suppression Souffle (Gate)</label>
                          <div className="flex gap-1 flex-wrap">
                            {[
                              { label: "Off", val: 0.0 },
                              { label: "Léger", val: 0.005 },
                              { label: "Moyen", val: 0.01 },
                              { label: "Fort", val: 0.02 },
                            ].map((g) => (
                              <button
                                key={g.label}
                                onClick={() => setVezoNoiseGate(g.val)}
                                className={`flex-1 py-1 text-[9px] rounded border font-semibold ${
                                  vezoNoiseGate === g.val
                                    ? "bg-teal-600 text-white border-teal-500"
                                    : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                                }`}
                              >
                                {g.label}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Volume Peak Norm */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1 flex justify-between">
                            <span>Normalisation:</span>
                            <span className="text-teal-300 font-mono font-bold">{vezoPeakNorm}</span>
                          </label>
                          <input
                            type="range"
                            min="0.70"
                            max="1.00"
                            step="0.05"
                            value={vezoPeakNorm}
                            onChange={(e) => setVezoPeakNorm(parseFloat(e.target.value))}
                            className="w-full accent-teal-500 bg-slate-800 rounded cursor-pointer"
                          />
                        </div>

                        {/* Adoucisseur Anti-Son Métallique */}
                        <div>
                          <label className="block text-[11px] text-slate-400 mb-1 flex justify-between">
                            <span>Chaleur Naturelle (Anti-Métal):</span>
                            <span className="text-teal-300 font-mono font-bold">{vezoWarmth}</span>
                          </label>
                          <input
                            type="range"
                            min="0.0"
                            max="1.0"
                            step="0.1"
                            value={vezoWarmth}
                            onChange={(e) => setVezoWarmth(parseFloat(e.target.value))}
                            className="w-full accent-teal-500 bg-slate-800 rounded cursor-pointer"
                          />
                          <span className="text-[9px] text-slate-500 block">Élimine le son robotique/métallique</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {outputText && (
                    <button
                      onClick={() => handleVezoTts(outputText)}
                      disabled={isVezoTtsLoading}
                      className={`w-full py-2 font-semibold rounded-xl transition-all shadow text-center flex items-center justify-center gap-2 ${
                        direction === "english-to-merina"
                          ? "bg-purple-600 hover:bg-purple-500 text-white"
                          : "bg-teal-600 hover:bg-teal-500 text-white"
                      }`}
                    >
                      🔄 Régénérer l'Audio avec ces paramètres
                    </button>
                  )}
                </div>
              )}

              <div className="w-full h-64 p-6 overflow-y-auto text-slate-100 text-lg leading-relaxed bg-transparent select-text">
                {outputText ? (
                  outputText
                ) : (
                  <span className="text-slate-500 italic text-base">Ny dikanteny dia hiseho eto...</span>
                )}
              </div>

              {vezoAudioUrl && (direction === "merina-to-vezo" || direction === "english-to-vezo" || direction === "english-to-merina") && (
                <div className="mx-6 mb-4 p-3 bg-indigo-950/40 border border-indigo-800/60 rounded-2xl flex flex-col gap-2">
                  <div className="flex items-center justify-between text-xs text-indigo-300 font-semibold">
                    <span>
                      🔊 Audio {direction === "english-to-merina" ? "Malagasy Officiel" : "Vezo"} Généré (
                      {direction === "english-to-merina" ? `${merinaSr}Hz - ${merinaSpeed}x` : `${vezoSr}Hz - ${vezoSpeed}x`}
                      )
                    </span>
                    <span className="text-[10px] text-indigo-400 bg-indigo-900/60 px-2 py-0.5 rounded-full">
                      {direction === "english-to-merina" ? "MMS-TTS-MLG" : "Fine-tuned Vezo"}
                    </span>
                  </div>
                  <audio controls autoPlay src={`http://localhost:8000${vezoAudioUrl}`} className="w-full h-8" />
                </div>
              )}

              <div className="p-4 bg-slate-900/30 border-t border-slate-800/30 flex items-center justify-between text-xs text-slate-500">
                <span>{outputText.length} caractères</span>
                <span className={`text-[10px] tracking-wide px-2.5 py-1 rounded-full border ${direction === "merina-to-vezo" || direction === "english-to-vezo" ? "bg-teal-950 text-teal-300 border-teal-800/40" : "bg-purple-950 text-purple-300 border-purple-800/40"}`}>
                  {direction === "merina-to-vezo" || direction === "english-to-vezo" ? "🌊 Vezo Dialect AI" : direction === "english-to-merina" ? "🇲🇬 Merina Standard AI" : "AI Dialect Convert"}
                </span>
              </div>
            </div>
          </div>

          {/* Model Tokenization Visualizer */}
          {modelTokens.length > 0 && (
            <div className="mb-8 bg-slate-900/40 border border-slate-800/80 p-6 rounded-3xl shadow-xl backdrop-blur-xl">
              <h2 className="text-sm font-semibold tracking-wider text-indigo-400 uppercase mb-4">
                Tokenisation par le modèle ({selectedModel === "model1" ? "Modèle 1" : selectedModel === "model1_2" ? "Modèle 1.2" : selectedModel === "model2" ? "Modèle 2" : selectedModel === "vezo" ? "🌊 Vezo" : "Modèle 2.0"})
              </h2>
              <div className="flex flex-wrap gap-2">
                {modelTokens.map((token, idx) => (
                  <span key={idx} className="bg-slate-950/80 text-indigo-300 px-3 py-1.5 rounded-xl text-sm font-mono border border-slate-800/60 shadow-inner">
                    {token}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Model Generation & Probability Decision Visualizer / Constraints Info */}
          {(generationDetails.length > 0 || constraintsApplied) && (
            <div className="mb-8 bg-slate-900/40 border border-slate-800/80 p-6 rounded-3xl shadow-xl backdrop-blur-xl animate-fade-in-up">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold tracking-wider text-indigo-400 uppercase flex items-center gap-2">
                  {constraintsApplied ? "🔒 Contraintes Lexicales Appliquées" : "🔮 Analyse de Décision & Probabilités (Génération Pas-à-Pas)"}
                </h2>
                <span className="text-[10px] bg-indigo-950 text-indigo-300 px-2.5 py-1 rounded-full border border-indigo-800/40">
                  {constraintsApplied ? "Mode Dictionnaire Forcé" : "Visualiseur Décisionnel AI"}
                </span>
              </div>

              {constraintsApplied ? (
                <div className="space-y-4">
                  <p className="text-xs text-slate-400">
                    Des termes de votre dictionnaire vérifié ont été détectés dans la phrase source. Le traducteur a activé la recherche par faisceau (Beam Search) pour garantir leur présence exacte dans le résultat Betsileo.
                  </p>
                  <div className="flex flex-wrap gap-3 items-center">
                    <span className="text-xs text-slate-500 font-medium">Mots forcés dans la traduction :</span>
                    {forcedWords.map((word, idx) => (
                      <span key={idx} className="bg-indigo-950/60 text-indigo-300 px-3 py-1 rounded-lg text-xs font-mono border border-indigo-850 flex items-center gap-1.5 shadow-inner">
                        <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-ping" />
                        {word}
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {/* Controls row */}
                  <div className="flex items-center justify-between mb-5">
                    <p className="text-xs text-slate-400">
                      La barre <span className="text-emerald-400 font-semibold">surlignée</span> = chemin choisi. Les autres = alternatives rejetées.
                    </p>
                    <div className="flex items-center gap-1 bg-slate-950/60 border border-slate-800 rounded-xl p-1">
                      <button
                        onClick={() => setView3D(false)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${!view3D ? "bg-indigo-600 text-white shadow" : "text-slate-400 hover:text-slate-200"
                          }`}
                      >
                        2D
                      </button>
                      <button
                        onClick={() => setView3D(true)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${view3D ? "bg-indigo-600 text-white shadow" : "text-slate-400 hover:text-slate-200"
                          }`}
                      >
                        3D ✨
                      </button>
                    </div>
                  </div>

                  {/* Path header — chosen tokens in sequence */}
                  <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-2 scrollbar-thin">
                    {generationDetails.map((step, idx) => {
                      const color =
                        step.prob >= 80 ? "from-emerald-500 to-emerald-600 shadow-emerald-500/20" :
                          step.prob >= 50 ? "from-amber-500 to-amber-600 shadow-amber-500/20" :
                            "from-rose-500 to-rose-600 shadow-rose-500/20";
                      return (
                        <React.Fragment key={idx}>
                          <div className="flex flex-col items-center gap-1 animate-fade-in-up" style={{ animationDelay: `${idx * 60}ms` }}>
                            <span className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold text-white bg-gradient-to-b ${color} shadow-md whitespace-nowrap`}>
                              {step.token}
                            </span>
                            <span className="text-[9px] text-slate-500">{step.prob}%</span>
                          </div>
                          {idx < generationDetails.length - 1 && (
                            <svg width="20" height="16" viewBox="0 0 20 16" className="flex-shrink-0 opacity-40">
                              <path d="M0 8 L14 8 M10 4 L14 8 L10 12" stroke="#6366f1" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </div>

                  {/* ===== 3D WORD CLOUD VIEW ===== */}
                  {view3D ? (
                    <TokenCloud3D generationDetails={generationDetails} />
                  ) : (

                    /* ===== 2D VIEW ===== */
                    <div className="space-y-5">
                      {generationDetails.map((step, idx) => (
                        <div key={idx} className="animate-fade-in-up" style={{ animationDelay: `${idx * 60}ms` }}>
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-[10px] font-mono text-slate-500 w-5 text-right flex-shrink-0">t{idx + 1}</span>
                            <div className="h-px flex-1 bg-slate-800/60" />
                          </div>
                          <div className="space-y-1.5 pl-7">
                            {step.alternatives.map((alt, altIdx) => {
                              const isChosen = alt.token === step.token;
                              const barColor =
                                isChosen
                                  ? step.prob >= 80 ? "bg-emerald-500" : step.prob >= 50 ? "bg-amber-500" : "bg-rose-500"
                                  : "bg-slate-700/40";
                              const borderColor = step.prob >= 80 ? "#10b981" : step.prob >= 50 ? "#f59e0b" : "#f43f5e";
                              return (
                                <div key={altIdx} className="flex items-center gap-3">
                                  <div className="w-4 flex-shrink-0 flex justify-center">
                                    {isChosen && (
                                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                                        <circle cx="5" cy="5" r="4"
                                          className={step.prob >= 80 ? "fill-emerald-500" : step.prob >= 50 ? "fill-amber-500" : "fill-rose-500"}
                                        />
                                      </svg>
                                    )}
                                  </div>
                                  <span className={`font-mono text-xs w-28 flex-shrink-0 truncate ${isChosen ? "font-bold text-slate-100" : "text-slate-500"
                                    }`}>{alt.token}</span>
                                  <div className="flex-1 h-5 bg-slate-950/60 rounded-md overflow-hidden relative">
                                    <div
                                      className={`h-full rounded-md animate-bar-grow ${barColor} ${isChosen ? "opacity-90" : "opacity-40"}`}
                                      style={{ width: `${alt.prob}%`, animationDelay: `${idx * 60 + altIdx * 80 + 150}ms` }}
                                    />
                                    {isChosen && (
                                      <div
                                        className="absolute inset-0 border-b-2 rounded-md pointer-events-none"
                                        style={{ borderColor }}
                                      />
                                    )}
                                  </div>
                                  <span className={`text-xs font-semibold w-10 text-right flex-shrink-0 tabular-nums ${isChosen ? "text-slate-100" : "text-slate-500"
                                    }`}>{alt.prob}%</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* History / Info Segment */}
          {history.length > 0 && (
            <div className="bg-slate-900/20 border border-slate-800 p-6 rounded-3xl shadow-xl backdrop-blur-xl">
              <h2 className="text-sm font-semibold tracking-wider text-slate-400 uppercase mb-4">Dikanteny Farany (Historique)</h2>
              <div className="space-y-3">
                {history.map((item, idx) => (
                  <div key={idx} className="bg-slate-900/60 p-4 rounded-2xl border border-slate-800/50 flex flex-col md:flex-row justify-between md:items-center gap-2 hover:border-slate-700/50 transition-all">
                    <div className="space-y-1">
                      <p className="text-sm text-slate-200 font-medium">{item.text}</p>
                      <p className="text-sm text-indigo-400 font-semibold">{item.translation}</p>
                    </div>
                    <div className="flex items-center justify-between md:justify-end gap-3 text-xs text-slate-500">
                      <span>{item.from} ➔ {item.to}</span>
                      <button
                        onClick={() => {
                          setInputText(item.text);
                          setOutputText(item.translation);
                        }}
                        className="text-indigo-500 hover:text-indigo-400 font-medium transition"
                      >
                        Réutiliser
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tokenizer Test Section */}
          <div className="mt-8 bg-slate-900/40 border border-slate-800 p-6 rounded-3xl shadow-xl backdrop-blur-xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold tracking-wider text-emerald-400 uppercase">Test du Nouveau Tokenizer (SentencePiece)</h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="flex flex-col space-y-3">
                <textarea
                  value={tokenizerInput}
                  onChange={(e) => setTokenizerInput(e.target.value)}
                  placeholder="Entrez une phrase en malgache pour tester le découpage..."
                  className="w-full h-32 p-4 bg-slate-950/50 rounded-2xl text-slate-200 border border-slate-800 focus:outline-none focus:border-emerald-500/50 transition-colors resize-none"
                />
                <button
                  onClick={handleTokenize}
                  disabled={isTokenizing || !tokenizerInput.trim()}
                  className="px-4 py-2 rounded-xl bg-emerald-600/80 hover:bg-emerald-600 disabled:bg-slate-800 text-white font-medium transition-all"
                >
                  {isTokenizing ? "Découpage..." : "Tester le découpage"}
                </button>
              </div>

              <div className="bg-slate-950/50 rounded-2xl border border-slate-800 p-4 overflow-y-auto h-48 flex flex-wrap gap-2 items-start content-start">
                {tokenizerTokens.length > 0 ? (
                  tokenizerTokens.map((token, idx) => (
                    <span key={idx} className="bg-slate-800 text-emerald-300 px-2 py-1 rounded-md text-sm font-mono border border-slate-700">
                      {token}
                    </span>
                  ))
                ) : (
                  <span className="text-slate-500 italic text-sm w-full text-center mt-10">Les jetons (tokens) s'afficheront ici...</span>
                )}
              </div>
            </div>
          </div>

        </>)}{/* end text mode */}

        {/* ── Speech Mode ── */}
        {appMode === "speech" && (
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-3xl p-6 shadow-xl backdrop-blur-xl">
            <SpeechTab onSendToTTS={(t) => { setTtsPreFill(t); setAppMode("tts"); }} />
          </div>
        )}

        {/* ── Media Mode ── */}
        {appMode === "media" && (
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-3xl p-6 shadow-xl backdrop-blur-xl">
            <h2 className="text-sm font-semibold tracking-wider text-indigo-400 uppercase mb-6">🎬 Transcription Vidéo / Audio</h2>
            <MediaTab />
          </div>
        )}

        {/* ── TTS Mode ── */}
        {appMode === "tts" && (
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-3xl p-6 shadow-xl backdrop-blur-xl">
            <h2 className="text-sm font-semibold tracking-wider text-emerald-400 uppercase mb-6">🔊 Synthèse Vocale Malagasy</h2>
            <TTSTab prefillText={ttsPreFill} />
          </div>
        )}
        {appMode === "tts2" && (
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-3xl p-6 shadow-xl backdrop-blur-xl">
            <h2 className="text-sm font-semibold tracking-wider text-emerald-400 uppercase mb-6">🗣️ Synthèse Vocale — Kokoro TTS2 Malagasy</h2>
            <TTSTab2 prefillText={ttsPreFill} />
          </div>
        )}

      </section>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 px-6 py-6 text-center text-xs text-slate-500 relative z-10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <p>© {new Date().getFullYear()} Malagasy NLP. Tous droits réservés.</p>
          <div className="flex items-center space-x-4">
            <span className="text-[10px] bg-slate-900 px-3 py-1 rounded-full border border-slate-800 text-indigo-400">
              Modèle: modele2.0 (Merina-to-Betsileo)
            </span>
          </div>
        </div>
      </footer>
    </main>
  );
}
