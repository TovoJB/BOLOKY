"use client";
import { useRef, useEffect, useCallback } from "react";

type Alt = { token: string; prob: number };
type Step = { token: string; prob: number; alternatives: Alt[] };

interface TokenCloud3DProps {
  generationDetails: Step[];
}

// ─── 3D math helpers ─────────────────────────────────────────────────────────
function rotateX(pt: [number, number, number], a: number): [number, number, number] {
  const cos = Math.cos(a), sin = Math.sin(a);
  return [pt[0], pt[1] * cos - pt[2] * sin, pt[1] * sin + pt[2] * cos];
}
function rotateY(pt: [number, number, number], a: number): [number, number, number] {
  const cos = Math.cos(a), sin = Math.sin(a);
  return [pt[0] * cos + pt[2] * sin, pt[1], -pt[0] * sin + pt[2] * cos];
}
function project(pt: [number, number, number], fov: number, cx: number, cy: number): [number, number, number] {
  const z = pt[2] + fov;
  const scale = fov / Math.max(z, 1);
  return [pt[0] * scale + cx, pt[1] * scale + cy, scale];
}

// ─── Color helpers ────────────────────────────────────────────────────────────
function probColor(prob: number, alpha = 1): string {
  if (prob >= 80) return `rgba(16,185,129,${alpha})`;
  if (prob >= 50) return `rgba(245,158,11,${alpha})`;
  return `rgba(244,63,94,${alpha})`;
}

export default function TokenCloud3D({ generationDetails }: TokenCloud3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rotRef = useRef({ x: -0.35, y: 0.5 });
  const dragRef = useRef<{ active: boolean; lastX: number; lastY: number }>({
    active: false, lastX: 0, lastY: 0,
  });
  const animRef = useRef<number | null>(null);

  // Build 3D point cloud from data
  const buildPoints = useCallback(() => {
    const X_SPREAD = 160;
    const Y_SCALE = 2.2;   // prob 0→100 mapped to -220→0
    const Z_SPREAD = 80;
    const steps = generationDetails.length;
    const cx = ((steps - 1) * X_SPREAD) / 2;

    type Point3D = {
      pos: [number, number, number];
      token: string;
      prob: number;
      isChosen: boolean;
      stepIdx: number;
      altIdx: number;
    };

    const points: Point3D[] = [];
    generationDetails.forEach((step, si) => {
      step.alternatives.forEach((alt, ai) => {
        const isChosen = alt.token === step.token;
        // Spread alternatives in Z; chosen centered
        const zOff = (ai - Math.floor(step.alternatives.length / 2)) * Z_SPREAD;
        points.push({
          pos: [si * X_SPREAD - cx, -(alt.prob * Y_SCALE), zOff],
          token: alt.token,
          prob: alt.prob,
          isChosen,
          stepIdx: si,
          altIdx: ai,
        });
      });
    });
    return points;
  }, [generationDetails]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H * 0.62;
    const FOV = 420;
    const rx = rotRef.current.x;
    const ry = rotRef.current.y;

    ctx.clearRect(0, 0, W, H);

    // Draw subtle grid on "floor"
    const gridY = 20; // flat ground at prob≈0 → y=0 in 3D
    const gridColor = "rgba(99,102,241,0.07)";
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;

    const points = buildPoints();

    // Sort by depth (painter's algorithm)
    const projected = points.map((p) => {
      let pt = rotateX(p.pos, rx);
      pt = rotateY(pt, ry);
      const [px, py, scale] = project(pt, FOV, cx, cy);
      return { ...p, px, py, scale, depth: pt[2] };
    });
    projected.sort((a, b) => a.depth - b.depth);

    // ── Draw connecting path for chosen tokens ────────────────────────────
    const chosenPts = projected
      .filter((p) => p.isChosen)
      .sort((a, b) => a.stepIdx - b.stepIdx);

    if (chosenPts.length > 1) {
      // Glow under line
      ctx.shadowBlur = 18;
      ctx.shadowColor = "rgba(99,102,241,0.7)";
      ctx.beginPath();
      chosenPts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.px, p.py);
        else ctx.lineTo(p.px, p.py);
      });
      ctx.strokeStyle = "rgba(99,102,241,0.85)";
      ctx.lineWidth = 2.5;
      ctx.lineJoin = "round";
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // ── Draw dots & labels ────────────────────────────────────────────────
    projected.forEach((p) => {
      const r = p.isChosen
        ? Math.max(7, 16 * p.scale)
        : Math.max(3, 8 * p.scale);

      // Dot
      ctx.beginPath();
      ctx.arc(p.px, p.py, r, 0, Math.PI * 2);
      if (p.isChosen) {
        // Glow ring
        ctx.shadowBlur = 20;
        ctx.shadowColor = probColor(p.prob, 0.9);
        ctx.fillStyle = probColor(p.prob, 0.92);
        ctx.fill();
        // White inner core
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(p.px, p.py, r * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.fill();
      } else {
        ctx.shadowBlur = 6;
        ctx.shadowColor = "rgba(99,102,241,0.2)";
        ctx.fillStyle = `rgba(71,85,105,${0.3 + p.prob / 250})`;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Token label
      const fontSize = p.isChosen
        ? Math.max(10, Math.round(14 * p.scale))
        : Math.max(8, Math.round(10 * p.scale));
      ctx.font = `${p.isChosen ? "700" : "400"} ${fontSize}px monospace`;
      ctx.fillStyle = p.isChosen
        ? probColor(p.prob, 1)
        : `rgba(148,163,184,${0.4 + p.prob / 200})`;
      ctx.textAlign = "center";
      ctx.fillText(
        p.token.length > 10 ? p.token.slice(0, 9) + "…" : p.token,
        p.px,
        p.py - r - 4
      );

      // Prob % under chosen dots
      if (p.isChosen) {
        ctx.font = `500 ${Math.max(8, Math.round(9 * p.scale))}px monospace`;
        ctx.fillStyle = probColor(p.prob, 0.7);
        ctx.fillText(`${p.prob}%`, p.px, p.py + r + 12);
      }
    });

    // ── Axis step labels ──────────────────────────────────────────────────
    generationDetails.forEach((step, si) => {
      const X_SPREAD = 160;
      const steps = generationDetails.length;
      const centerX = ((steps - 1) * X_SPREAD) / 2;
      let pt: [number, number, number] = [si * X_SPREAD - centerX, 30, 0];
      pt = rotateX(pt, rx) as [number, number, number];
      pt = rotateY(pt, ry) as [number, number, number];
      const [px, py] = project(pt, FOV, cx, cy);
      ctx.font = "500 9px monospace";
      ctx.fillStyle = "rgba(100,116,139,0.7)";
      ctx.textAlign = "center";
      ctx.fillText(`t${si + 1}`, px, py);
    });
  }, [buildPoints, generationDetails]);

  useEffect(() => {
    let running = true;
    const animate = () => {
      if (!running) return;
      draw();
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => {
      running = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [draw]);

  // ── Mouse / Touch drag ────────────────────────────────────────────────────
  const onMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragRef.current.active) return;
    const dx = e.clientX - dragRef.current.lastX;
    const dy = e.clientY - dragRef.current.lastY;
    rotRef.current.y += dx * 0.008;
    rotRef.current.x += dy * 0.008;
    dragRef.current.lastX = e.clientX;
    dragRef.current.lastY = e.clientY;
  };
  const onMouseUp = () => { dragRef.current.active = false; };

  const onTouchStart = (e: React.TouchEvent) => {
    dragRef.current = { active: true, lastX: e.touches[0].clientX, lastY: e.touches[0].clientY };
  };
  const onTouchMove = (e: React.TouchEvent) => {
    if (!dragRef.current.active) return;
    const dx = e.touches[0].clientX - dragRef.current.lastX;
    const dy = e.touches[0].clientY - dragRef.current.lastY;
    rotRef.current.y += dx * 0.008;
    rotRef.current.x += dy * 0.008;
    dragRef.current.lastX = e.touches[0].clientX;
    dragRef.current.lastY = e.touches[0].clientY;
  };

  return (
    <div className="relative w-full rounded-2xl overflow-hidden bg-slate-950/70 border border-slate-800/60 shadow-2xl"
      style={{ height: 340 }}
    >
      {/* Hint */}
      <div className="absolute top-3 right-4 text-[10px] text-slate-600 select-none pointer-events-none">
        ↺ Glisser pour tourner
      </div>
      {/* Legend */}
      <div className="absolute top-3 left-4 flex items-center gap-3 text-[10px] text-slate-500 select-none pointer-events-none">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400" /> Chemin choisi
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-slate-600" /> Alternatives
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-6 h-px bg-indigo-500" /> Trajectoire
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={900}
        height={340}
        className="w-full h-full cursor-grab active:cursor-grabbing select-none"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onMouseUp}
      />
    </div>
  );
}
