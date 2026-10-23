// Record the 3D viewer to image frames, frame by frame (deterministic, any machine speed).
//
//   npm install puppeteer-core@23            (once; uses your installed Google Chrome)
//   python -m http.server 8766 -d viewer &   (serve the viewer)
//   node scripts/record_viewer.mjs /tmp/frames
//   python scripts/encode_video.py /tmp/frames
//
// Each shot: [scene index, controller to follow, camera, start (s from takeover), end, playback speed]
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const OUT = process.argv[2] || 'frames';
const URL = process.env.VIEWER_URL || 'http://localhost:8766/_record.html';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const FPS = 30;
const W = 1600, H = 900;
const SHOTS = [
  { scene: 0, follow: 'learned_full', cam: 'chase', from: -1.4, to: 3.4, speed: 0.5, xray: true,
    caption: 'Stopped car ahead on a wet road at 107 km/h. The network takes over 1.4 s before impact.' },
  { scene: 1, follow: 'learned_full', cam: 'wheel', from: -0.2, to: 2.6, speed: 0.4, xray: true,
    caption: 'Brake close-up: each disc glows with its own brake force. Slippery patch under one side.' },
  { scene: 0, follow: 'learned_brakes_only', cam: 'chase', from: -0.2, to: 3.0, speed: 0.6, xray: false,
    caption: 'Steering frozen: the network changes lanes by braking one side harder than the other.' },
];
const VIDEO_CSS = `#runs, #transport, #scenes, #scene-info { display: none !important; }
#top { width: auto !important; }
#cap { position: fixed; right: 24px; bottom: 24px; max-width: 520px; padding: 12px 16px; border-radius: 10px;
  background: rgba(14,19,26,0.84); border: 1px solid #263140; color: #e9eef3; font: 600 19px/1.35 Archivo, sans-serif; }
#cluster { bottom: 24px !important; }`;

fs.mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', `--window-size=${W},${H}`],
});
const page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
page.on('pageerror', e => console.error('page error:', e.message));
await page.goto(URL, { waitUntil: 'load' });
await page.waitForFunction(() => window.__crashai && document.getElementById('loading').hidden, { timeout: 120000 });
await page.addStyleTag({ content: VIDEO_CSS });

let n = 0;
for (const [k, shot] of SHOTS.entries()) {
  const tk = await page.evaluate(sh => {
    const A = window.__crashai;
    A.loadScene(sh.scene); A.follow(sh.follow); A.setCam(sh.cam);
    if (A.state.xray !== sh.xray) A.setX(sh.xray);
    let cap = document.getElementById('cap');
    if (!cap) { cap = document.createElement('div'); cap.id = 'cap'; document.body.appendChild(cap); }
    cap.textContent = sh.caption;
    return window.CRASHAI.scenes[sh.scene].runs[0].tTakeover;
  }, shot);
  const frames = Math.round((shot.to - shot.from) / shot.speed * FPS);
  for (let i = -8; i < frames; i++) { // 8 warm-up frames let the camera settle
    const t = tk + shot.from + Math.max(0, i) * shot.speed / FPS;
    await page.evaluate((t, dt) => window.__crashai.renderAt(t, dt), t, 1 / FPS);
    if (i < 0) continue;
    await page.screenshot({ path: path.join(OUT, `f${String(n++).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 92 });
  }
  console.log(`shot ${k + 1}: ${frames} frames`);
}
await browser.close();
console.log(`${n} frames in ${OUT}`);
