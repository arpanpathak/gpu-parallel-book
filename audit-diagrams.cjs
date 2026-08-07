#!/usr/bin/env node
/**
 * Pixel-accurate SVG diagram audit via Chrome DevTools Protocol.
 *
 * Loads each SVG file, then for every <text> element queries the REAL
 * rendered bounding box (getBBox). Reports:
 *   1. text-vs-text overlaps  (two labels painted on top of each other)
 *   2. text escaping its nearest enclosing <rect> (label outside its box)
 *   3. text escaping the viewBox entirely
 *
 * Usage: node audit-real.mjs <dir-with-svgs>
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9240;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    let id = 0;
    const pending = new Map();
    const listeners = [];
    ws.onopen = () => resolve({
      send(method, params = {}) {
        return new Promise((res, rej) => {
          const mid = ++id;
          pending.set(mid, { res, rej });
          ws.send(JSON.stringify({ id: mid, method, params }));
        });
      },
      on(method, cb) { listeners.push({ method, cb }); },
    });
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        const { res, rej } = pending.get(msg.id);
        pending.delete(msg.id);
        if (msg.error) rej(new Error(msg.error.message));
        else res(msg.result);
      } else if (msg.method) {
        for (const { method, cb } of listeners) if (method === msg.method) cb(msg.params);
      }
    };
    ws.onerror = (e) => reject(new Error('ws error ' + e.message));
  });
}

async function auditOne(cdp, file) {
  const svgContent = fs.readFileSync(file, 'utf-8');
  const html = `<!DOCTYPE html><html><body style="margin:0">${svgContent}</body></html>`;
  const url = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
  const loadPromise = new Promise(res => cdp.on('Page.loadEventFired', res));
  await cdp.send('Page.navigate', { url });
  await Promise.race([loadPromise, sleep(8000)]);
  await sleep(600);

  const r = await cdp.send('Runtime.evaluate', {
    expression: `(function(){
      const svg = document.querySelector('svg');
      if (!svg) return { error: 'no svg' };
      const vbs = (svg.getAttribute('viewBox') || '').trim().split(/[\\s,]+/).map(Number);
      const W = (vbs.length === 4) ? vbs[2] : (svg.clientWidth || 860);
      const H = (vbs.length === 4) ? vbs[3] : (svg.clientHeight || 600);
      const texts = [];
      svg.querySelectorAll('text').forEach(t => {
        const b = t.getBBox();
        const bb = { x: b.x, y: b.y, w: b.width, h: b.height,
                     text: t.textContent.trim().slice(0, 40) };
        texts.push(bb);
      });
      const rects = [];
      svg.querySelectorAll('rect').forEach(rr => {
        const b = rr.getBBox();
        rects.push({ x: b.x, y: b.y, w: b.width, h: b.height });
      });
      return { W, H, texts, rects };
    })()`,
    returnByValue: true,
  });
  const data = r.result && r.result.value;
  if (!data || data.error) return { file, error: (data && data.error) || 'no data' };

  const problems = [];

  // 1. text vs text overlaps (allow 1px tolerance)
  const T = data.texts;
  for (let i = 0; i < T.length; i++) {
    for (let j = i + 1; j < T.length; j++) {
      const a = T[i], b = T[j];
      const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (ox > 2 && oy > 2) {
        const area = ox * oy;
        const minArea = Math.min(a.w * a.h, b.w * b.h);
        if (area > 0.35 * minArea) {  // meaningful overlap, not mere touching
          problems.push(`OVERLAP '${a.text}' [${a.x.toFixed(0)},${a.y.toFixed(0)}] vs '${b.text}' [${b.x.toFixed(0)},${b.y.toFixed(0)}]`);
        }
      }
    }
  }

  // 2. text escaping its nearest enclosing rect (by center point)
  const R = data.rects;
  for (const t of T) {
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2;
    const inside = R.filter(r => cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h);
    if (inside.length > 0) {
      // pick the smallest containing rect
      const best = inside.reduce((m, r) => (r.w * r.h < m.w * m.h ? r : m));
      if (t.x < best.x - 2 || t.x + t.w > best.x + best.w + 2 ||
          t.y < best.y - 2 || t.y + t.h > best.y + best.h + 2) {
        problems.push(`TEXT EXCEEDS BOX '${t.text}' text=[${t.x.toFixed(0)},${t.y.toFixed(0)},${(t.x+t.w).toFixed(0)},${(t.y+t.h).toFixed(0)}] box=[${best.x.toFixed(0)},${best.y.toFixed(0)},${(best.x+best.w).toFixed(0)},${(best.y+best.h).toFixed(0)}]`);
      }
    }
  }

  // 3. text outside viewBox
  const vbW = data.W, vbH = data.H;
  for (const t of T) {
    if (t.x < -1 || t.x + t.w > vbW + 1 || t.y < -1 || t.y + t.h > vbH + 1) {
      problems.push(`TEXT OUT OF VIEWBOX '${t.text}' [${t.x.toFixed(0)},${t.y.toFixed(0)},${(t.x+t.w).toFixed(0)},${(t.y+t.h).toFixed(0)}] (vb ${vbW}x${vbH})`);
    }
  }

  return { file: path.basename(file), W: data.W, H: data.H, problems };
}

async function main() {
  const dir = process.argv[2];
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.svg')).sort();

  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-sandbox',
    `--remote-debugging-port=${PORT}`,
    '--user-data-dir=/tmp/chrome-audit-profile',
    'about:blank',
  ], { stdio: 'ignore' });

  try {
    let targets = null;
    for (let i = 0; i < 50; i++) {
      try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); break; }
      catch { await sleep(200); }
    }
    const page = targets.find(t => t.type === 'page');
    const cdp = await connect(page.webSocketDebuggerUrl);
    await cdp.send('Page.enable');

    let total = 0;
    for (const f of files) {
      const res = await auditOne(cdp, path.join(dir, f));
      if (res.error) { console.log(`=== ${f}: ERROR ${res.error}`); continue; }
      if (res.problems.length === 0) {
        console.log(`=== ${f}: OK (${res.W}x${res.H})`);
      } else {
        console.log(`=== ${f}: ${res.problems.length} PROBLEMS (${res.W}x${res.H})`);
        for (const p of res.problems.slice(0, 40)) console.log('   ', p);
        total += res.problems.length;
      }
    }
    console.log(`\nTOTAL PROBLEMS: ${total}`);
    process.exit(0);
  } catch (e) {
    console.error("ERROR:", e.message); console.error(e.stack);
    process.exit(1);
  } finally {
    chrome.kill();
  }
}

main();
