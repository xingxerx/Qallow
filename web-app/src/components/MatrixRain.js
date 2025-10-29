import { useEffect, useRef } from 'react';

const CHARACTERS = '01';
const FONT_SIZE = 16;
const TRAIL_ALPHA = 0.05;
const DROP_RESET_PROBABILITY = 0.975;

const useMatrixRain = (canvasRef) => {
  const animationFrameId = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return undefined;
    }

    let width = 0;
    let height = 0;
    let columns = 0;
    let drops = [];

    const setup = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
      columns = Math.floor(width / FONT_SIZE);
      drops = Array.from({ length: columns }, () => 1);
      ctx.font = `${FONT_SIZE}px monospace`;
    };

    const draw = () => {
      ctx.fillStyle = `rgba(0, 0, 0, ${TRAIL_ALPHA})`;
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = '#0F0';

      for (let i = 0; i < drops.length; i += 1) {
        const text = CHARACTERS.charAt(Math.floor(Math.random() * CHARACTERS.length));
        ctx.fillText(text, i * FONT_SIZE, drops[i] * FONT_SIZE);

        if (drops[i] * FONT_SIZE > height && Math.random() > DROP_RESET_PROBABILITY) {
          drops[i] = 0;
        }

        drops[i] += 1;
      }

      animationFrameId.current = requestAnimationFrame(draw);
    };

    const start = () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      animationFrameId.current = requestAnimationFrame(draw);
    };

    const handleResize = () => {
      setup();
      start();
    };

    setup();
    start();

    const handleVisibility = () => {
      if (document.visibilityState === 'visible') {
        start();
      } else if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
    };

    window.addEventListener('resize', handleResize);
    document.addEventListener('visibilitychange', handleVisibility);

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [canvasRef]);
};

const MatrixRain = () => {
  const canvasRef = useRef(null);
  useMatrixRain(canvasRef);

  return <canvas ref={canvasRef} className="matrix-bg" aria-hidden="true" />;
};

export default MatrixRain;
