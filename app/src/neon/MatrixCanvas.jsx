import React, { useEffect, useRef } from 'react';

// Matrix rain animation as a React component
export default function MatrixCanvas({ enabled = true }) {
  const canvasRef = useRef(null);
  const runningRef = useRef(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let w, h, columns, drops;

    const glyphs = '01あいうえおカキクケコｱｲｳｴｵ01ΛλξπΣσµΩ<>[]{}/*+-=|';
    const fontSize = 14; // px
    const opacityTrail = 0.08; // lower = longer trails

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      w = canvas.clientWidth = window.innerWidth;
      h = canvas.clientHeight = window.innerHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      columns = Math.ceil(w / fontSize);
      drops = Array.from({ length: columns }, () => Math.random() * -20);
      ctx.font = `${fontSize}px ui-monospace, monospace`;
    }

    function draw() {
      if (!runningRef.current || !enabled) return;
      // fade rect for trails
      ctx.fillStyle = `rgba(0, 0, 0, ${opacityTrail})`;
      ctx.fillRect(0, 0, w, h);

      for (let i = 0; i < columns; i++) {
        const char = glyphs[Math.floor(Math.random() * glyphs.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;
        const hue = 188 + (i % 20);
        ctx.fillStyle = `hsl(${hue} 100% 60%)`;
        ctx.fillText(char, x, y);

        if (y > h && Math.random() > 0.975) drops[i] = 0;
        drops[i]++;
      }
      requestAnimationFrame(draw);
    }

    resize();
    window.addEventListener('resize', resize);
    const onVis = () => {
      runningRef.current = document.visibilityState === 'visible';
      if (runningRef.current && enabled) requestAnimationFrame(draw);
    };
    document.addEventListener('visibilitychange', onVis);

    // kick off
    requestAnimationFrame(draw);

    return () => {
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('resize', resize);
      runningRef.current = false;
    };
  }, [enabled]);

  return (
    <canvas
      ref={canvasRef}
      className="matrix-bg"
      style={{ display: enabled ? 'block' : 'none' }}
      aria-hidden="true"
    />
  );
}
