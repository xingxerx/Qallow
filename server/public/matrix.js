(function(){
  const canvas = document.getElementById('matrix');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, columns, drops, running = true;

  const glyphs = '01あいうえおカキクケコｱｲｳｴｵ01ΛλξπΣσµΩ<>[]{}/*+-=|';
  const fontSize = 14; // px
  const opacityTrail = 0.08; // lower = longer trails

  function resize(){
    const dpr = window.devicePixelRatio || 1;
    w = canvas.clientWidth = window.innerWidth;
    h = canvas.clientHeight = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    columns = Math.ceil(w / fontSize);
    drops = Array.from({length: columns}, () => Math.random() * -20);
    ctx.font = `${fontSize}px ui-monospace, monospace`;
  }

  function draw(){
    if(!running) return;
    // fade rect for trails
    ctx.fillStyle = `rgba(0, 0, 0, ${opacityTrail})`;
    ctx.fillRect(0,0,w,h);

    for(let i=0;i<columns;i++){
      const char = glyphs[Math.floor(Math.random() * glyphs.length)];
      const x = i * fontSize;
      const y = drops[i] * fontSize;
      // neon cyan → blue gradient per row
      const hue = 188 + (i % 20);
      ctx.fillStyle = `hsl(${hue} 100% 60%)`;
      ctx.fillText(char, x, y);

      if(y > h && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    }
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener('resize', resize);
  document.addEventListener('visibilitychange', () => {
    running = document.visibilityState === 'visible';
    if(running) requestAnimationFrame(draw);
  });

  const toggle = document.getElementById('toggleMatrix');
  if(toggle){
    toggle.addEventListener('change', (e) => {
      canvas.style.display = e.target.checked ? 'block' : 'none';
      if(e.target.checked && running) requestAnimationFrame(draw);
    });
  }

  requestAnimationFrame(draw);
})();
