#!/usr/bin/env python3
"""
Interface graphique de découpage audio+texte assisté
pour le dataset Vezo TTS/ASR.

Usage:
    cd /home/tovo/Bureau/scrap
    python alignement/interface_decoupage.py

Fonctionnalités:
- Liste les entrées du dataset avec marquage visuel (🔴 trop long, 🟡 court, ✅ OK)
- Lecture audio avec waveform visualisée
- Curseurs pour marquer les points de découpe
- Sauvegarde automatique dans le dataset (metadata + WAV)
"""

import os
import csv
import json
import wave
import struct
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading
import subprocess

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_vezo"
METADATA_ASR = DATASET_DIR / "metadata" / "metadata_asr_bible.csv"
METADATA_TTS = DATASET_DIR / "metadata" / "metadata_tts_bible.csv"
SEGMENTS_DIR = DATASET_DIR / "audio" / "segments"
BACKUP_DIR   = DATASET_DIR / "audio" / "segments_backup"

MAX_CHARS    = 300
MAX_DURATION = 15.0
MIN_DURATION = 1.0
MIN_CHARS    = 5

# ─── COULEURS ──────────────────────────────────────────────────────────────────
COLOR_BG       = "#1e1e2e"
COLOR_PANEL    = "#2a2a3e"
COLOR_ACCENT   = "#7c6af7"
COLOR_OK       = "#a6e3a1"
COLOR_WARN     = "#f9e2af"
COLOR_DANGER   = "#f38ba8"
COLOR_TEXT     = "#cdd6f4"
COLOR_SUBTEXT  = "#6c7086"
COLOR_BUTTON   = "#313244"
COLOR_BUTTON_H = "#45475a"
COLOR_WAVEFORM = "#89b4fa"
COLOR_CUT      = "#f38ba8"
COLOR_PLAYHEAD = "#a6e3a1"


def load_wav_samples(path, max_samples=8000):
    """Charge les échantillons d'un WAV pour la waveform (sans dépendances externes)."""
    try:
        with wave.open(str(path), 'r') as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            duration   = n_frames / framerate
            raw        = wf.readframes(n_frames)
        fmt = {1: 'b', 2: 'h', 4: 'i'}.get(sampwidth, 'h')
        total   = len(raw) // sampwidth
        samples = list(struct.unpack(f'<{total}{fmt}', raw))
        if n_channels > 1:
            samples = samples[::n_channels]
        if len(samples) > max_samples:
            step    = len(samples) // max_samples
            samples = samples[::step][:max_samples]
        maxval     = max(abs(s) for s in samples) or 1
        normalized = [s / maxval for s in samples]
        return normalized, duration, framerate
    except Exception:
        return [], 0.0, 16000


def get_wav_duration(path):
    try:
        with wave.open(str(path), 'r') as wf:
            return wf.getnframes() / wf.getframerate()
    except:
        return 0.0


def status_of(row):
    try:
        dur  = float(row.get('duration', 0))
        nch  = len(row.get('transcript', ''))
    except:
        return 'ok'
    if dur > MAX_DURATION or nch > MAX_CHARS:
        return 'long'
    if dur < MIN_DURATION or nch < MIN_CHARS:
        return 'short'
    return 'ok'


def status_icon(s):
    return {'long': '🔴', 'short': '🟡', 'ok': '✅'}.get(s, '❓')


def status_color(s):
    return {'long': COLOR_DANGER, 'short': COLOR_WARN, 'ok': COLOR_OK}.get(s, COLOR_TEXT)


# ─── I/O DATASET ───────────────────────────────────────────────────────────────

def load_dataset():
    rows = []
    try:
        with open(METADATA_ASR, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de charger le dataset:\n{e}")
    return rows


def save_dataset(rows):
    with open(METADATA_ASR, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_path', 'duration', 'transcript', 'language', 'source'])
        writer.writeheader()
        writer.writerows(rows)
    with open(METADATA_TTS, 'w', encoding='utf-8') as f:
        for row in rows:
            clip_id = Path(row['audio_path']).stem
            text    = row['transcript']
            f.write(f"{clip_id}|{text}|{text}\n")


# ─── DÉCOUPAGE AUDIO ───────────────────────────────────────────────────────────

def slice_wav(src, dst, t_start, t_end):
    cmd = ['ffmpeg', '-y', '-i', str(src),
           '-ss', str(t_start), '-to', str(t_end),
           '-c', 'copy', str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


# ─── APPLICATION PRINCIPALE ────────────────────────────────────────────────────

class DecoupageApp:
    def __init__(self, root):
        self.root      = root
        self.root.title("✂️  Interface de Découpage — Dataset Vezo")
        self.root.geometry("1400x820")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(1100, 700)

        self.dataset  = load_dataset()
        self.filtered = []
        self.sel_idx  = None
        self.cut_pos  = 0.5   # position relative (0..1)
        self.samples  = []
        self.duration = 0.0
        self.framerate= 16000

        self._build_ui()
        self._apply_filter('all')
        self._update_stats()

    # ── Build ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=COLOR_BG,
                            sashwidth=5, sashrelief=tk.FLAT)
        pw.pack(fill=tk.BOTH, expand=True)

        # ── GAUCHE : liste ──────────────────────────────────────────────────────
        left = tk.Frame(pw, bg=COLOR_PANEL, width=390)
        pw.add(left, minsize=280)

        # Filtres
        ff = tk.Frame(left, bg=COLOR_PANEL, pady=8)
        ff.pack(fill=tk.X, padx=8)
        tk.Label(ff, text="Filtre :", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 6))
        self.filter_var = tk.StringVar(value='all')
        for val, lbl, col in [
            ('all',   'Tout',      COLOR_TEXT),
            ('long',  '🔴 Longs',  COLOR_DANGER),
            ('short', '🟡 Courts', COLOR_WARN),
            ('ok',    '✅ OK',     COLOR_OK),
        ]:
            tk.Radiobutton(ff, text=lbl, variable=self.filter_var, value=val,
                           command=lambda v=val: self._apply_filter(v),
                           bg=COLOR_PANEL, fg=col, selectcolor=COLOR_BG,
                           activebackground=COLOR_PANEL, activeforeground=col,
                           font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)

        self.stats_label = tk.Label(left, text="", bg=COLOR_PANEL,
                                     fg=COLOR_SUBTEXT, font=('Segoe UI', 8))
        self.stats_label.pack(fill=tk.X, padx=10, pady=(0, 4))

        lf = tk.Frame(left, bg=COLOR_PANEL)
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        sb = ttk.Scrollbar(lf)
        self.listbox = tk.Listbox(lf, yscrollcommand=sb.set, selectmode=tk.SINGLE,
                                   bg=COLOR_BG, fg=COLOR_TEXT,
                                   selectbackground=COLOR_ACCENT, selectforeground='white',
                                   font=('Courier', 9), activestyle='none',
                                   bd=0, highlightthickness=0)
        sb.config(command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        # ── DROITE : éditeur ────────────────────────────────────────────────────
        right = tk.Frame(pw, bg=COLOR_BG)
        pw.add(right, minsize=680)

        # En-tête
        hdr = tk.Frame(right, bg=COLOR_PANEL, pady=8)
        hdr.pack(fill=tk.X)
        self.entry_label  = tk.Label(hdr, text="← Sélectionnez une entrée",
                                      bg=COLOR_PANEL, fg=COLOR_TEXT,
                                      font=('Segoe UI', 11, 'bold'), anchor='w')
        self.entry_label.pack(side=tk.LEFT, padx=14)
        self.status_badge = tk.Label(hdr, text="", bg=COLOR_PANEL,
                                      fg=COLOR_TEXT, font=('Segoe UI', 10))
        self.status_badge.pack(side=tk.LEFT, padx=6)

        # Métriques
        mf = tk.Frame(right, bg=COLOR_BG, pady=4)
        mf.pack(fill=tk.X, padx=12)
        self.dur_label   = tk.Label(mf, text="⏱ —", bg=COLOR_BG, fg=COLOR_TEXT, font=('Segoe UI', 9))
        self.dur_label.pack(side=tk.LEFT, padx=(0, 16))
        self.char_label  = tk.Label(mf, text="📝 —", bg=COLOR_BG, fg=COLOR_TEXT, font=('Segoe UI', 9))
        self.char_label.pack(side=tk.LEFT, padx=(0, 16))
        self.token_label = tk.Label(mf, text="🔤 ~—", bg=COLOR_BG, fg=COLOR_TEXT, font=('Segoe UI', 9))
        self.token_label.pack(side=tk.LEFT)

        # ── Canvas waveform ─────────────────────────────────────────────────────
        cf = tk.Frame(right, bg=COLOR_BG, padx=12, pady=4)
        cf.pack(fill=tk.X)
        self.canvas = tk.Canvas(cf, height=160, bg="#11111b",
                                 highlightthickness=1, highlightbackground=COLOR_PANEL,
                                 cursor='crosshair')
        self.canvas.pack(fill=tk.X)
        self.canvas.bind('<Configure>',  lambda e: self._draw_waveform())
        self.canvas.bind('<Button-1>',   self._on_canvas_click)
        self.canvas.bind('<B1-Motion>',  self._on_canvas_drag)

        # Légende
        lf2 = tk.Frame(right, bg=COLOR_BG, padx=12)
        lf2.pack(fill=tk.X)
        for c, t in [(COLOR_CUT, "✂ Point de découpe (glisser pour déplacer)  "),
                     (COLOR_WAVEFORM, "Waveform audio")]:
            tk.Label(lf2, text="─", bg=COLOR_BG, fg=c, font=('Segoe UI', 9)).pack(side=tk.LEFT)
            tk.Label(lf2, text=t,   bg=COLOR_BG, fg=COLOR_SUBTEXT, font=('Segoe UI', 8)).pack(side=tk.LEFT)

        # Indicateurs de segments
        sif = tk.Frame(right, bg=COLOR_BG, padx=12, pady=4)
        sif.pack(fill=tk.X)
        tk.Label(sif, text="✂ Point :", bg=COLOR_BG, fg=COLOR_TEXT, font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.cut_time_lbl = tk.Label(sif, text="—", bg=COLOR_BG, fg=COLOR_DANGER,
                                      font=('Segoe UI', 9, 'bold'), width=8)
        self.cut_time_lbl.pack(side=tk.LEFT)
        self.seg1_lbl = tk.Label(sif, text="Seg.A: —", bg=COLOR_BG, fg=COLOR_WARN, font=('Segoe UI', 8))
        self.seg1_lbl.pack(side=tk.LEFT, padx=(10, 6))
        self.seg2_lbl = tk.Label(sif, text="Seg.B: —", bg=COLOR_BG, fg=COLOR_WARN, font=('Segoe UI', 8))
        self.seg2_lbl.pack(side=tk.LEFT)

        # ── Texte ───────────────────────────────────────────────────────────────
        tlf = tk.LabelFrame(right, text=" 📝 Transcription ", bg=COLOR_BG, fg=COLOR_TEXT,
                             font=('Segoe UI', 9), padx=8, pady=6, bd=1, relief=tk.FLAT)
        tlf.pack(fill=tk.X, padx=12, pady=(8, 4))
        self.text_entry = tk.Text(tlf, height=3, wrap=tk.WORD,
                                   bg=COLOR_PANEL, fg=COLOR_TEXT,
                                   insertbackground=COLOR_TEXT,
                                   font=('Segoe UI', 10), bd=0, padx=6, pady=6, relief=tk.FLAT)
        self.text_entry.pack(fill=tk.X)
        self.text_entry.bind('<KeyRelease>', self._on_text_change)

        # ── Boutons ─────────────────────────────────────────────────────────────
        bf = tk.Frame(right, bg=COLOR_BG, padx=12, pady=10)
        bf.pack(fill=tk.X)
        for lbl, cmd, bg, fg in [
            ("▶  Lire",          self._play_audio,   COLOR_ACCENT,  '#fff'),
            ("⏹  Stop",          self._stop_audio,   COLOR_BUTTON,  COLOR_TEXT),
            ("✂️  Découper ici", self._do_cut,        COLOR_DANGER,  '#fff'),
            ("💾  Sauver texte", self._save_text,     COLOR_BUTTON,  COLOR_TEXT),
            ("🗑  Supprimer",    self._delete_entry,  "#3b2020",     COLOR_DANGER),
        ]:
            tk.Button(bf, text=lbl, command=cmd,
                      bg=bg, fg=fg, activebackground=COLOR_BUTTON_H,
                      font=('Segoe UI', 9), bd=0, padx=14, pady=7,
                      cursor='hand2', relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 8))

        # Barre de statut
        self.status_bar = tk.Label(right, text="Prêt.", bg=COLOR_PANEL,
                                    fg=COLOR_SUBTEXT, font=('Segoe UI', 8),
                                    anchor='w', padx=10)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Filtre ──────────────────────────────────────────────────────────────────

    def _apply_filter(self, filt):
        self.listbox.delete(0, tk.END)
        self.filtered = []
        for i, row in enumerate(self.dataset):
            s = status_of(row)
            if filt != 'all' and s != filt:
                continue
            self.filtered.append(i)
            dur  = float(row.get('duration', 0))
            nch  = len(row.get('transcript', ''))
            stem = Path(row.get('audio_path', 'unknown')).stem
            line = f"{status_icon(s)} {stem:<38} {dur:>6.1f}s {nch:>4}ch"
            self.listbox.insert(tk.END, line)
            self.listbox.itemconfig(tk.END, fg=status_color(s))

    def _update_stats(self):
        n_long  = sum(1 for r in self.dataset if status_of(r) == 'long')
        n_short = sum(1 for r in self.dataset if status_of(r) == 'short')
        n_ok    = len(self.dataset) - n_long - n_short
        self.stats_label.config(
            text=f"Total: {len(self.dataset)}  |  🔴 {n_long}  🟡 {n_short}  ✅ {n_ok}"
        )

    # ── Sélection ───────────────────────────────────────────────────────────────

    def _on_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.sel_idx = sel[0]
        row = self.dataset[self.filtered[self.sel_idx]]
        s   = status_of(row)

        stem = Path(row.get('audio_path', '')).stem
        dur  = float(row.get('duration', 0))
        text = row.get('transcript', '')
        nch  = len(text)
        ntok = nch * 2

        self.entry_label.config(text=stem)
        self.status_badge.config(text=f"{status_icon(s)}  {s.upper()}", fg=status_color(s))
        self.dur_label.config(
            text=f"⏱ {dur:.2f}s",
            fg=COLOR_DANGER if dur > MAX_DURATION else COLOR_WARN if dur < MIN_DURATION else COLOR_OK)
        self.char_label.config(
            text=f"📝 {nch} chars",
            fg=COLOR_DANGER if nch > MAX_CHARS else COLOR_WARN if nch < MIN_CHARS else COLOR_OK)
        self.token_label.config(
            text=f"🔤 ~{ntok} tokens",
            fg=COLOR_DANGER if ntok > 600 else COLOR_OK)

        self.text_entry.delete('1.0', tk.END)
        self.text_entry.insert('1.0', text)

        wav = SEGMENTS_DIR / (stem + '.wav')
        if wav.exists():
            self.samples, self.duration, self.framerate = load_wav_samples(wav)
        else:
            self.samples, self.duration, self.framerate = [], dur, 16000

        self.cut_pos = 0.5
        self._draw_waveform()
        self._update_cut_labels()
        self._status(f"Chargé : {stem}  ({dur:.2f}s, {nch} chars, ~{ntok} tokens)")

    # ── Waveform ────────────────────────────────────────────────────────────────

    def _draw_waveform(self):
        c = self.canvas
        c.delete('all')
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 2 or H < 2:
            return

        mid   = H // 2
        cut_x = int(self.cut_pos * W)

        # Fond
        c.create_rectangle(0,     0, cut_x, H, fill="#181825", outline='')
        c.create_rectangle(cut_x, 0, W,     H, fill="#1e1e2e", outline='')

        # Waveform
        if self.samples:
            n   = len(self.samples)
            amp = (H // 2) - 6
            pts = []
            for i, s in enumerate(self.samples):
                x = int(i * W / n)
                y = int(mid - s * amp)
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(pts, fill=COLOR_WAVEFORM, width=1)

        # Ligne médiane
        c.create_line(0, mid, W, mid, fill=COLOR_SUBTEXT, dash=(2, 4))

        # Ligne de découpe
        c.create_line(cut_x, 0, cut_x, H, fill=COLOR_CUT, width=2)
        c.create_text(min(cut_x + 4, W - 70), 14,
                      text=f"✂ {self.cut_pos * self.duration:.2f}s",
                      fill=COLOR_CUT, anchor='w', font=('Segoe UI', 8))

        # Timecodes bas
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x = int(frac * W)
            t = frac * self.duration
            c.create_text(max(x, 14), H - 6, text=f"{t:.1f}s",
                          fill=COLOR_SUBTEXT, font=('Segoe UI', 7), anchor='s')

    def _on_canvas_click(self, event):
        W = self.canvas.winfo_width()
        if W > 0:
            self.cut_pos = max(0.02, min(0.98, event.x / W))
            self._draw_waveform()
            self._update_cut_labels()

    def _on_canvas_drag(self, event):
        self._on_canvas_click(event)

    def _update_cut_labels(self):
        cut_t = self.cut_pos * self.duration
        self.cut_time_lbl.config(text=f"{cut_t:.2f}s")
        self.seg1_lbl.config(text=f"Seg.A: {cut_t:.1f}s")
        self.seg2_lbl.config(text=f"Seg.B: {self.duration - cut_t:.1f}s")

    # ── Texte ───────────────────────────────────────────────────────────────────

    def _on_text_change(self, event=None):
        text = self.text_entry.get('1.0', tk.END).strip()
        nch  = len(text)
        ntok = nch * 2
        self.char_label.config(
            text=f"📝 {nch} chars",
            fg=COLOR_DANGER if nch > MAX_CHARS else COLOR_WARN if nch < MIN_CHARS else COLOR_OK)
        self.token_label.config(
            text=f"🔤 ~{ntok} tokens",
            fg=COLOR_DANGER if ntok > 600 else COLOR_OK)

    # ── Actions ─────────────────────────────────────────────────────────────────

    def _get_row(self):
        if self.sel_idx is None:
            return None, None
        ds_idx = self.filtered[self.sel_idx]
        return self.dataset[ds_idx], ds_idx

    def _play_audio(self):
        row, _ = self._get_row()
        if not row:
            return
        wav = SEGMENTS_DIR / (Path(row['audio_path']).stem + '.wav')
        if not wav.exists():
            self._status(f"WAV introuvable: {wav}", error=True)
            return
        threading.Thread(target=self._play_thread, args=(wav,), daemon=True).start()

    def _play_thread(self, wav):
        try:
            subprocess.run(['aplay', str(wav)], capture_output=True)
        except FileNotFoundError:
            subprocess.run(['ffplay', '-nodisp', '-autoexit', str(wav)], capture_output=True)

    def _stop_audio(self):
        subprocess.run(['pkill', '-f', 'aplay'],  capture_output=True)
        subprocess.run(['pkill', '-f', 'ffplay'], capture_output=True)

    def _save_text(self):
        row, ds_idx = self._get_row()
        if row is None:
            return
        new_text = self.text_entry.get('1.0', tk.END).strip()
        if not new_text:
            messagebox.showwarning("Attention", "Le texte est vide!")
            return
        self.dataset[ds_idx]['transcript'] = new_text
        save_dataset(self.dataset)
        self._apply_filter(self.filter_var.get())
        self._update_stats()
        self._status(f"✅ Texte sauvegardé pour {Path(row['audio_path']).stem}")

    def _do_cut(self):
        row, ds_idx = self._get_row()
        if row is None:
            return
        cut_t   = self.cut_pos * self.duration
        stem    = Path(row['audio_path']).stem
        wav_src = SEGMENTS_DIR / (stem + '.wav')

        if not wav_src.exists():
            self._status(f"WAV introuvable: {wav_src}", error=True)
            return
        if cut_t < 0.5 or cut_t > self.duration - 0.5:
            messagebox.showwarning("Position invalide",
                "Le point de découpe doit être à au moins 0.5s des extrémités.")
            return

        # Dialogue de division du texte
        dlg = CutTextDialog(self.root, row.get('transcript', ''), cut_t, self.duration)
        self.root.wait_window(dlg.win)
        if dlg.result is None:
            return
        text1, text2 = dlg.result

        # Backup
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_src, BACKUP_DIR / wav_src.name)

        # Découpe audio
        wav_a = SEGMENTS_DIR / (stem + '_a.wav')
        wav_b = SEGMENTS_DIR / (stem + '_b.wav')
        ok_a  = slice_wav(wav_src, wav_a, 0,     cut_t)
        ok_b  = slice_wav(wav_src, wav_b, cut_t, self.duration)
        if not (ok_a and ok_b):
            messagebox.showerror("Erreur ffmpeg", "Le découpage audio a échoué.")
            return

        dur_a = get_wav_duration(wav_a)
        dur_b = get_wav_duration(wav_b)

        # Mise à jour dataset
        lang = row.get('language', 'skg-x-vz')
        src  = row.get('source', '')
        row_a = {'audio_path': f"audio/segments/{stem}_a.wav",
                 'duration': f"{dur_a:.3f}", 'transcript': text1,
                 'language': lang, 'source': src + '_a'}
        row_b = {'audio_path': f"audio/segments/{stem}_b.wav",
                 'duration': f"{dur_b:.3f}", 'transcript': text2,
                 'language': lang, 'source': src + '_b'}

        self.dataset.pop(ds_idx)
        self.dataset.insert(ds_idx, row_b)
        self.dataset.insert(ds_idx, row_a)
        wav_src.unlink()

        save_dataset(self.dataset)
        self.sel_idx = None
        self._apply_filter(self.filter_var.get())
        self._update_stats()
        self._status(f"✅ {stem} → {stem}_a ({dur_a:.1f}s) + {stem}_b ({dur_b:.1f}s)")

    def _delete_entry(self):
        row, ds_idx = self._get_row()
        if row is None:
            return
        stem = Path(row['audio_path']).stem
        if not messagebox.askyesno("Confirmer", f"Supprimer définitivement:\n{stem} ?"):
            return
        wav = SEGMENTS_DIR / (stem + '.wav')
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if wav.exists():
            shutil.move(str(wav), BACKUP_DIR / wav.name)
        self.dataset.pop(ds_idx)
        save_dataset(self.dataset)
        self.sel_idx = None
        self._apply_filter(self.filter_var.get())
        self._update_stats()
        self._status(f"🗑 Supprimé: {stem}")

    def _status(self, msg, error=False):
        self.status_bar.config(text=msg, fg=COLOR_DANGER if error else COLOR_SUBTEXT)


# ─── DIALOGUE RÉPARTITION DU TEXTE ────────────────────────────────────────────

class CutTextDialog:
    def __init__(self, parent, text, cut_t, total_t):
        self.result = None
        self.win = tk.Toplevel(parent)
        self.win.title("✂️  Répartir le texte")
        self.win.configure(bg=COLOR_BG)
        self.win.geometry("720x430")
        self.win.transient(parent)
        self.win.grab_set()

        tk.Label(self.win,
                 text=f"Audio découpé à  {cut_t:.2f}s  sur  {total_t:.2f}s",
                 bg=COLOR_BG, fg=COLOR_TEXT, font=('Segoe UI', 10, 'bold')
                 ).pack(pady=(14, 2))
        tk.Label(self.win,
                 text="Répartissez le texte entre les deux segments audio ci-dessous.",
                 bg=COLOR_BG, fg=COLOR_SUBTEXT, font=('Segoe UI', 9)
                 ).pack(pady=(0, 10))

        # Suggestion automatique proportionnelle
        frac = cut_t / total_t if total_t > 0 else 0.5
        mid  = max(1, int(len(text) * frac))
        # Chercher la ponctuation la plus proche
        best = mid
        for delta in range(0, min(80, mid)):
            for d in [delta, -delta]:
                p = mid + d
                if 0 < p < len(text) and text[p] in ' ,;.!?-\n':
                    best = p + 1
                    break

        sug1 = text[:best].strip()
        sug2 = text[best:].strip()

        self.texts = []
        for i, (lbl, default) in enumerate([
            (f"📌 Segment A  (0 → {cut_t:.1f}s)", sug1),
            (f"📌 Segment B  ({cut_t:.1f}s → {total_t:.1f}s)", sug2),
        ]):
            tk.Label(self.win, text=lbl, bg=COLOR_BG, fg=COLOR_TEXT,
                     font=('Segoe UI', 9, 'bold'), anchor='w').pack(fill=tk.X, padx=16)
            t = tk.Text(self.win, height=3, wrap=tk.WORD,
                        bg=COLOR_PANEL, fg=COLOR_TEXT, insertbackground=COLOR_TEXT,
                        font=('Segoe UI', 10), bd=0, padx=6, pady=6, relief=tk.FLAT)
            t.insert('1.0', default)
            t.pack(fill=tk.X, padx=16, pady=(2, 10))
            self.texts.append(t)

        bf = tk.Frame(self.win, bg=COLOR_BG)
        bf.pack(pady=10)
        tk.Button(bf, text="✂️  Confirmer",
                  command=self._confirm, bg=COLOR_DANGER, fg='white',
                  font=('Segoe UI', 10), bd=0, padx=16, pady=8,
                  relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=8)
        tk.Button(bf, text="✕ Annuler",
                  command=self.win.destroy, bg=COLOR_BUTTON, fg=COLOR_TEXT,
                  font=('Segoe UI', 10), bd=0, padx=12, pady=8,
                  relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT)

    def _confirm(self):
        t1 = self.texts[0].get('1.0', tk.END).strip()
        t2 = self.texts[1].get('1.0', tk.END).strip()
        if not t1 or not t2:
            messagebox.showwarning("Attention", "Les deux segments doivent avoir du texte!")
            return
        self.result = (t1, t2)
        self.win.destroy()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except:
        pass
    style.configure('TScrollbar', background=COLOR_PANEL,
                    troughcolor=COLOR_BG, arrowcolor=COLOR_TEXT)
    app = DecoupageApp(root)
    root.mainloop()
