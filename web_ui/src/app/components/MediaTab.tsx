"use client";

import React, { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";
interface Transcription {
  id: string;
  filename: string;
  transcription: string;
  translation: string;
  model_used: string;
  direction: string;
  vtt_url?: string;
  dubbed_url?: string;
  original_dubbed_url?: string;
  kokoro_dubbed_url?: string;
  created_at: string;
}

interface PhonemeSegment {
  segment_idx: number;
  text: string;
  phonemes: string;
  start: number;
  end: number;
}

const ACCEPTED_MEDIA = ".mp4,.mkv,.avi,.mov,.webm,.mp3,.wav,.ogg,.flac";
const VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov", ".webm"];

interface SubtitledVideoProps extends React.VideoHTMLAttributes<HTMLVideoElement> {
  srcUrl: string;
  vttUrl?: string;
  videoType?: string;
}

function SubtitledVideo({ srcUrl, vttUrl, videoType, ...props }: SubtitledVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const existingTracks = video.querySelectorAll("track");
    existingTracks.forEach(t => t.remove());

    if (vttUrl) {
      const track = document.createElement("track");
      track.kind = "subtitles";
      track.label = "Traduction";
      track.srclang = "mg";
      track.src = vttUrl;
      track.default = true;
      video.appendChild(track);
    }
  }, [srcUrl, vttUrl]);

  return (
    <video ref={videoRef} {...props}>
      <source src={srcUrl} type={videoType} />
    </video>
  );
}


export default function MediaTab() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<"model1" | "model1_2" | "model2" | "model3" | "vezo">("model1");
  const [direction, setDirection] = useState<"merina-to-betsileo" | "betsileo-to-merina" | "merina-to-vezo">("merina-to-betsileo");
  const [transcription, setTranscription] = useState("");
  const [translation, setTranslation] = useState("");
  const [currentVttUrl, setCurrentVttUrl] = useState("");
  const [currentDubbedUrl, setCurrentDubbedUrl] = useState("");
  const [currentOriginalDubbedUrl, setCurrentOriginalDubbedUrl] = useState("");
  const [localVideoUrl, setLocalVideoUrl] = useState("");
  const [dubbing, setDubbing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDubbing, setIsDubbing] = useState(false);
  const [dubbingJobId, setDubbingJobId] = useState("");
  const [error, setError] = useState("");
  const [history, setHistory] = useState<Transcription[]>([]);
  // ── Kokoro state ──
  const [currentJobId, setCurrentJobId] = useState("");
  const [kokoroDubUrl, setKokoroDubUrl] = useState("");
  const [kokoroPhonemes, setKokoroPhonemes] = useState<PhonemeSegment[]>([]);
  const [editedPhonemes, setEditedPhonemes] = useState<PhonemeSegment[]>([]);
  const [isKokoro, setIsKokoro] = useState(false);
  const [isKokoroRegen, setIsKokoroRegen] = useState(false);
  const [manualPhoneme, setManualPhoneme] = useState("");
  const [useManualPhoneme, setUseManualPhoneme] = useState(false);
  const [isKokoroManual, setIsKokoroManual] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadHistory = async () => {
    try {
      const res = await fetch(`${API}/transcriptions`);
      if (res.ok) setHistory(await res.json());
    } catch { }
  };

  useEffect(() => { loadHistory(); }, []);

  // Polling pour le doublage asynchrone
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isDubbing && dubbingJobId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API}/transcriptions`);
          if (res.ok) {
            const hist = await res.json();
            setHistory(hist);
            const job = hist.find((h: Transcription) => h.id === dubbingJobId);
            if (job && (job.dubbed_url || job.original_dubbed_url)) {
              if (job.dubbed_url) setCurrentDubbedUrl(job.dubbed_url);
              if (job.original_dubbed_url) setCurrentOriginalDubbedUrl(job.original_dubbed_url);
              setIsDubbing(false);
              setDubbingJobId("");
            }
          }
        } catch { }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [isDubbing, dubbingJobId]);

  const handleProcess = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError("");
    setTranscription("");
    setTranslation("");
    setCurrentVttUrl("");
    setCurrentDubbedUrl("");
    setCurrentOriginalDubbedUrl("");
    setLocalVideoUrl("");

    const form = new FormData();
    form.append("file", file);
    form.append("model", model);
    form.append("direction", direction);
    if (dubbing) form.append("dubbing", "true");

    try {
      const res = await fetch(`${API}/transcribe`, { method: "POST", body: form });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Erreur serveur");
      }
      const data = await res.json();
      setTranscription(data.transcription);
      setTranslation(data.translation);
      setCurrentJobId(data.id);
      setKokoroDubUrl("");
      setKokoroPhonemes([]);
      setEditedPhonemes([]);
      if (data.vtt_url) setCurrentVttUrl(data.vtt_url);

      if (data.is_dubbing && (!data.dubbed_url || !data.original_dubbed_url)) {
        setIsDubbing(true);
        setDubbingJobId(data.id);
      } else {
        if (data.dubbed_url) setCurrentDubbedUrl(data.dubbed_url);
        if (data.original_dubbed_url) setCurrentOriginalDubbedUrl(data.original_dubbed_url);
      }

      const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
      if (VIDEO_EXTS.includes(ext)) {
        setLocalVideoUrl(URL.createObjectURL(file));
      }

      await loadHistory();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDelete = async (id: string) => {
    await fetch(`${API}/transcriptions/${id}`, { method: "DELETE" });
    setHistory(prev => prev.filter(h => h.id !== id));
  };

  const handleKokoroDub = async () => {
    if (!currentJobId) return;
    setIsKokoro(true);
    try {
      const res = await fetch(`${API}/kokoro_dub`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: currentJobId }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || "Erreur Kokoro"); }
      const data = await res.json();
      setKokoroDubUrl(data.kokoro_dubbed_url);
      setKokoroPhonemes(data.phonemes);
      setEditedPhonemes(data.phonemes);
    } catch (e: any) { setError(e.message); }
    finally { setIsKokoro(false); }
  };

  const handleKokoroRegen = async () => {
    if (!currentJobId) return;
    setIsKokoroRegen(true);
    try {
      const res = await fetch(`${API}/kokoro_dub_regen`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: currentJobId, phonemes: editedPhonemes }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || "Erreur régénération"); }
      const data = await res.json();
      setKokoroDubUrl(data.kokoro_dubbed_url);
      setKokoroPhonemes(data.phonemes);
      setEditedPhonemes(data.phonemes);
    } catch (e: any) { setError(e.message); }
    finally { setIsKokoroRegen(false); }
  };

  const handleKokoroManual = async () => {
    if (!currentJobId || !manualPhoneme.trim()) return;
    setIsKokoroManual(true);
    try {
      const res = await fetch(`${API}/kokoro_dub_manual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: currentJobId, manual_phoneme: manualPhoneme.trim() }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || "Erreur phonème manuel"); }
      const data = await res.json();
      setKokoroDubUrl(data.kokoro_dubbed_url);
      setKokoroPhonemes(data.phonemes);
      setEditedPhonemes(data.phonemes);
    } catch (e: any) { setError(e.message); }
    finally { setIsKokoroManual(false); }
  };

  const fileExt = file ? file.name.substring(file.name.lastIndexOf(".")).toLowerCase() : "";
  const isVideo = VIDEO_EXTS.includes(fileExt);

  return (
    <div className="space-y-6">
      {/* Upload zone */}
      <div
        onClick={() => fileInputRef.current?.click()}
        className="group border-2 border-dashed border-slate-700 hover:border-indigo-500/60 rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 bg-slate-900/30 hover:bg-slate-900/50"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_MEDIA}
          className="hidden"
          onChange={e => {
            setFile(e.target.files?.[0] || null);
            setLocalVideoUrl("");
            setCurrentVttUrl("");
            setCurrentDubbedUrl("");
            setCurrentOriginalDubbedUrl("");
          }}
        />
        {file ? (
          <div className="space-y-2">
            <div className="text-3xl">{isVideo ? "🎬" : "🎵"}</div>
            <p className="text-slate-200 font-semibold">{file.name}</p>
            <p className="text-slate-500 text-xs">{(file.size / 1024 / 1024).toFixed(2)} MB — Cliquer pour changer</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-4xl opacity-40">📂</div>
            <p className="text-slate-400 font-medium">Cliquer pour choisir un fichier</p>
            <p className="text-slate-600 text-xs">Vidéo : .mp4 .mkv .avi .mov | Audio : .mp3 .wav .ogg</p>
          </div>
        )}
      </div>

      {/* Options */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider mb-2 block">Modèle de traduction</label>
          <div className="flex gap-2 flex-wrap">
            {(["model1", "model1_2", "model2", "model3"] as const).map(m => (
              <button
                key={m}
                onClick={() => { setModel(m); if (direction === "merina-to-vezo") setDirection("merina-to-betsileo"); }}
                className={`flex-1 py-2 rounded-xl text-xs font-semibold transition-all ${model === m && direction !== "merina-to-vezo" ? "bg-purple-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
              >
                {m === "model1" ? "Standard" : m === "model1_2" ? "Modèle 1.2" : m === "model2" ? "Optimisé" : "2.0"}
              </button>
            ))}
            <button
              onClick={() => { setModel("vezo"); setDirection("merina-to-vezo"); }}
              className={`flex-1 py-2 rounded-xl text-xs font-semibold transition-all ${model === "vezo" ? "bg-teal-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
            >
              🌊 Vezo
            </button>
          </div>
        </div>
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider mb-2 block">Direction</label>
          <div className="flex gap-2 flex-wrap">
            {(["merina-to-betsileo", "betsileo-to-merina"] as const).map(d => (
              <button
                key={d}
                onClick={() => { setDirection(d); if (model === "vezo") setModel("model1"); }}
                className={`flex-1 py-2 rounded-xl text-xs font-semibold transition-all ${direction === d ? "bg-indigo-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
              >
                {d === "merina-to-betsileo" ? "Mer → Bet" : "Bet → Mer"}
              </button>
            ))}
            <button
              onClick={() => { setDirection("merina-to-vezo"); setModel("vezo"); }}
              className={`flex-1 py-2 rounded-xl text-xs font-semibold transition-all ${direction === "merina-to-vezo" ? "bg-teal-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
            >
              🌊 Mer → Vezo
            </button>
          </div>
        </div>
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider mb-2 block">Audio</label>
          <label className="flex items-center gap-2 bg-slate-800 p-2 rounded-xl border border-slate-700 cursor-pointer hover:bg-slate-700 transition-colors h-[32px] mt-1">
            <input
              type="checkbox"
              checked={dubbing}
              onChange={e => setDubbing(e.target.checked)}
              className="w-4 h-4 rounded bg-slate-900 border-slate-600 text-purple-600 focus:ring-purple-600 focus:ring-offset-slate-800"
            />
            <span className="text-xs font-semibold text-slate-300">Générer Voice Over</span>
          </label>
        </div>
      </div>

      {/* Résultats */}
      {/* Résultats Texte */}
      {(transcription || translation) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-4">
            <div className="text-xs text-indigo-400 uppercase tracking-wider font-semibold mb-2">📝 Transcription MMS</div>
            <p className="text-slate-200 text-sm leading-relaxed">{transcription}</p>
          </div>
          <div className={`bg-slate-900/60 rounded-2xl border border-slate-800 p-4 ${direction === "merina-to-vezo" ? "border-teal-800/40" : ""}`}>
            <div className={`text-xs uppercase tracking-wider font-semibold mb-2 ${direction === "merina-to-vezo" ? "text-teal-400" : "text-purple-400"}`}>
              {direction === "merina-to-vezo" ? "🌊 Traduction Vezo" : `🔄 Traduction ${direction === "merina-to-betsileo" ? "Betsileo" : "Merina"}`}
            </div>
            <p className="text-slate-200 text-sm leading-relaxed">{translation}</p>
          </div>
        </div>
      )}

      {/* Lecteurs Média */}
      {(localVideoUrl || currentDubbedUrl || currentOriginalDubbedUrl) && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

          {/* Média 1: Original avec sous-titres */}
          {(localVideoUrl && currentVttUrl && isVideo) && (
            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-4">
              <div className="text-xs text-emerald-400 uppercase tracking-wider font-semibold mb-3 flex items-center gap-2">
                ▶ 1. Vidéo Originale (Sous-titrée)
                {isDubbing && (
                  <span className="flex items-center gap-2 text-purple-400 bg-purple-900/30 px-2 py-1 rounded-full ml-auto">
                    <div className="w-3 h-3 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                    Doublage en cours...
                  </span>
                )}
              </div>
              <SubtitledVideo
                key={`orig-${localVideoUrl}-${currentVttUrl}`}
                srcUrl={localVideoUrl}
                vttUrl={`${API}${currentVttUrl}`}
                videoType={file?.type}
                controls
                className="w-full rounded-xl"
                crossOrigin="anonymous"
              />
              <div className="mt-3 text-right">
                <a href={`${API}${currentVttUrl}`} download className="text-xs text-emerald-400 hover:text-emerald-300">
                  📥 Télécharger Sous-titres (.vtt)
                </a>
              </div>
            </div>
          )}

          {/* Média 2: Doublé (Traduction) */}
          {currentDubbedUrl && (
            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-4">
              <div className="text-xs text-purple-400 uppercase tracking-wider font-semibold mb-3">
                ▶ 2. Doublage (Traduction)
              </div>
              {currentDubbedUrl.endsWith(".wav") ? (
                <audio key={currentDubbedUrl} controls className="w-full" src={`${API}${currentDubbedUrl}`} />
              ) : (
                <SubtitledVideo
                  key={`dubbed-${currentDubbedUrl}-${currentVttUrl}`}
                  srcUrl={`${API}${currentDubbedUrl}`}
                  vttUrl={currentVttUrl ? `${API}${currentVttUrl}` : undefined}
                  videoType={file?.type}
                  controls
                  className="w-full rounded-xl"
                  crossOrigin="anonymous"
                />
              )}
              <div className="mt-3 text-right">
                <a href={`${API}${currentDubbedUrl}`} download className="text-xs text-purple-400 hover:text-purple-300">
                  📥 Télécharger Doublage Trad.
                </a>
              </div>
            </div>
          )}

          {/* Média 3: Doublé (Original) */}
          {currentOriginalDubbedUrl && (
            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-4">
              <div className="text-xs text-blue-400 uppercase tracking-wider font-semibold mb-3">
                ▶ 3. Doublage (Original)
              </div>
              {currentOriginalDubbedUrl.endsWith(".wav") ? (
                <audio key={currentOriginalDubbedUrl} controls className="w-full" src={`${API}${currentOriginalDubbedUrl}`} />
              ) : (
                <video key={currentOriginalDubbedUrl} controls className="w-full rounded-xl" crossOrigin="anonymous">
                  <source src={`${API}${currentOriginalDubbedUrl}`} type={file?.type} />
                </video>
              )}
              <div className="mt-3 text-right">
                <a href={`${API}${currentOriginalDubbedUrl}`} download className="text-xs text-blue-400 hover:text-blue-300">
                  📥 Télécharger Doublage Orig.
                </a>
              </div>
            </div>
          )}

        </div>
      )}

      {/* ── Section 4 : Kokoro TTS Dubbing ── */}
      {currentJobId && (
        <div className="bg-slate-900/40 border border-emerald-900/40 rounded-3xl p-6 space-y-5">

          {/* Header */}
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold tracking-wider text-emerald-400 uppercase">🗣️ 4. Doublage Kokoro TTS</h3>
            <button
              onClick={handleKokoroDub}
              disabled={isKokoro || useManualPhoneme}
              className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-xs font-bold transition-all flex items-center gap-2"
            >
              {isKokoro ? <><svg className="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg><span>Génération...</span></> : <span>🎙️ Générer depuis texte</span>}
            </button>
          </div>

          {/* ── Mode : Phonème Manuel ── */}
          <div className="bg-slate-950/50 border border-slate-800 rounded-2xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">✍️ Phonème manuel</span>
              <button
                onClick={() => setUseManualPhoneme(v => !v)}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-300 focus:outline-none ${
                  useManualPhoneme ? "bg-emerald-600" : "bg-slate-700"
                }`}
              >
                <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform duration-300 ${
                  useManualPhoneme ? "translate-x-4" : "translate-x-0.5"
                }`} />
              </button>
            </div>

            <textarea
              value={manualPhoneme}
              onChange={e => setManualPhoneme(e.target.value)}
              placeholder="Entrez le phonème ici (ex: helo wɜːld, fə netʃ...)..."
              disabled={!useManualPhoneme}
              rows={3}
              className={`w-full bg-slate-900 text-slate-100 text-sm rounded-xl px-3 py-2 border focus:outline-none focus:ring-1 focus:ring-emerald-500 resize-none transition-opacity ${
                useManualPhoneme ? "border-emerald-700/60 opacity-100" : "border-slate-800 opacity-40 cursor-not-allowed"
              }`}
            />

            {useManualPhoneme && (
              <button
                onClick={handleKokoroManual}
                disabled={isKokoroManual || !manualPhoneme.trim()}
                className="w-full py-2.5 rounded-xl bg-gradient-to-r from-emerald-700 to-teal-700 hover:from-emerald-600 hover:to-teal-600 disabled:opacity-40 text-white text-sm font-bold transition-all flex items-center justify-center gap-2"
              >
                {isKokoroManual
                  ? <><svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg><span>Génération...</span></>
                  : <span>⚡ Générer directement avec ce phonème</span>
                }
              </button>
            )}
          </div>

          {kokoroPhonemes.length > 0 && (
            <div className="space-y-3">
              <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">📊 Segments phonétiques éditables</div>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {editedPhonemes.map((seg, idx) => (
                  <div key={idx} className="bg-slate-900/60 rounded-xl border border-slate-800 p-3 space-y-1">
                    <div className="text-[10px] text-slate-600 flex gap-2">
                      <span className="text-emerald-600">Seg {seg.segment_idx + 1}</span>
                      <span>{seg.start.toFixed(1)}s → {seg.end.toFixed(1)}s</span>
                    </div>
                    <textarea
                      value={seg.phonemes}
                      onChange={e => { const u = [...editedPhonemes]; u[idx] = { ...u[idx], phonemes: e.target.value }; setEditedPhonemes(u); }}
                      className="w-full bg-slate-800 text-slate-100 text-xs rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-emerald-500 resize-none"
                      rows={2}
                    />
                  </div>
                ))}
              </div>
              <button
                onClick={handleKokoroRegen}
                disabled={isKokoroRegen}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-teal-700 to-emerald-700 hover:from-teal-600 hover:to-emerald-600 disabled:opacity-40 text-white text-sm font-bold transition-all flex items-center justify-center gap-2"
              >
                {isKokoroRegen ? <><svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg><span>Régénération...</span></> : <span>🔄 Régénérer avec phonèmes modifiés</span>}
              </button>
            </div>
          )}

          {kokoroDubUrl && (
            <div className="space-y-3">
              <div className="text-xs text-emerald-400 font-semibold uppercase tracking-wider">▶ Résultat — Vidéo + Audio Kokoro</div>
              {kokoroDubUrl.endsWith(".wav") ? (
                <audio key={kokoroDubUrl} controls className="w-full" src={`${API}${kokoroDubUrl}`} />
              ) : (
                <SubtitledVideo
                  key={`kokoro-${kokoroDubUrl}-${currentVttUrl}`}
                  srcUrl={`${API}${kokoroDubUrl}`}
                  vttUrl={currentVttUrl ? `${API}${currentVttUrl}` : undefined}
                  controls
                  className="w-full rounded-xl"
                  crossOrigin="anonymous"
                />
              )}
              <div className="text-right">
                <a href={`${API}${kokoroDubUrl}`} download className="text-xs text-emerald-400 hover:text-emerald-300">📥 Télécharger Doublage Kokoro</a>
              </div>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="bg-rose-950/40 border border-rose-800/50 rounded-2xl p-4 text-rose-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Bouton */}
      <button
        onClick={handleProcess}
        disabled={!file || isProcessing}
        className="w-full py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-30 disabled:cursor-not-allowed text-white font-bold text-base transition-all duration-300 active:scale-[0.98] flex items-center justify-center gap-3 shadow-lg shadow-indigo-500/20"
      >
        {isProcessing ? (
          <>
            <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Traitement en cours (peut prendre quelques minutes)...</span>
          </>
        ) : (
          <span>{isVideo ? "🎬" : "🎵"} Générer la Transcription + Traduction</span>
        )}
      </button>

      {/* Historique */}
      {history.length > 0 && (
        <div>
          <h3 className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-3">Historique des transcriptions</h3>
          <div className="space-y-2">
            {history.map(item => (
              <div key={item.id} className="bg-slate-900/40 border border-slate-800/60 rounded-2xl p-4 flex items-start justify-between gap-4 hover:border-slate-700/60 transition-all">
                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-300 text-sm font-medium truncate">{item.filename}</span>
                    <span className="text-[10px] bg-slate-800 text-slate-500 px-2 py-0.5 rounded-full flex-shrink-0">
                      {item.direction === "merina-to-betsileo" ? "Mer→Bet" : "Bet→Mer"}
                    </span>
                  </div>
                  <p className="text-xs text-indigo-400 truncate">{item.transcription}</p>
                  <p className="text-xs text-purple-400 truncate">{item.translation}</p>

                  <div className="flex items-center gap-3">
                    <p className="text-[10px] text-slate-600">{new Date(item.created_at).toLocaleString("fr-FR")}</p>
                    {item.vtt_url && (
                      <a href={`${API}${item.vtt_url}`} download onClick={e => e.stopPropagation()} className="text-[10px] text-emerald-500 hover:text-emerald-400">
                        (📥 .vtt)
                      </a>
                    )}
                    {item.dubbed_url && (
                      <a href={`${API}${item.dubbed_url}`} download onClick={e => e.stopPropagation()} className="text-[10px] text-purple-500 hover:text-purple-400">
                        (📥 Doublage Trad.)
                      </a>
                    )}
                    {item.original_dubbed_url && (
                      <a href={`${API}${item.original_dubbed_url}`} download onClick={e => e.stopPropagation()} className="text-[10px] text-blue-500 hover:text-blue-400">
                        (📥 Doublage Orig.)
                      </a>
                    )}
                  </div>
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
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
