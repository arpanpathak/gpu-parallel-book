#!/usr/bin/env node
/**
 * Render a URL to PDF via Chrome DevTools Protocol with printBackground: true.
 * (This is the switch that actually controls whether background colors
 * survive printing; Chrome's CLI --print-to-pdf defaults it to false, which
 * is why the Night Owl theme previously rendered as white pages.)
 *
 * Usage: node print-to-pdf.cjs <url> <outfile>
 */
const { spawn } = require('child_process');
const fs = require('fs');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9223;
const [url, outfile] = process.argv.slice(2);

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

async function main() {
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-sandbox',
    `--remote-debugging-port=${PORT}`,
    '--user-data-dir=/tmp/chrome-cdp-profile2',
    'about:blank',
  ], { stdio: 'ignore' });

  try {
    let targets = null;
    for (let i = 0; i < 50; i++) {
      try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); break; }
      catch { await sleep(200); }
    }
    if (!targets) throw new Error('Chrome debugging endpoint not reachable');

    const page = targets.find(t => t.type === 'page');
    const cdp = await connect(page.webSocketDebuggerUrl);

    const loadPromise = new Promise(res => cdp.on('Page.loadEventFired', res));
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Page.navigate', { url });
    await Promise.race([loadPromise, sleep(40000)]);
    await sleep(1500);

    // Poll until MathJax finishes typesetting or timeout (120 s).
    let mathDone = false;
    for (let i = 0; i < 120; i++) {
      const r = await cdp.send('Runtime.evaluate', {
        expression: `(function(){
          if (typeof MathJax === 'undefined' || !MathJax.Hub) return 'done';
          const q = MathJax.Hub.queue;
          return (q.pending === 0 && q.running === 0) ? 'done' : 'busy';
        })()`,
        returnByValue: true,
      });
      if (r.result && r.result.value === 'done') { mathDone = true; break; }
      await sleep(1000);
    }
    console.log(mathDone ? 'MathJax typeset complete' : 'MathJax timeout (continuing)');

    const pdf = await cdp.send('Page.printToPDF', {
      printBackground: true,
      displayHeaderFooter: false,
      preferCSSPageSize: true,
    });
    fs.writeFileSync(outfile, Buffer.from(pdf.data, 'base64'));
    console.log('PDF written:', outfile, fs.statSync(outfile).size, 'bytes');
  } catch (e) {
    console.error('ERROR:', e.message);
    process.exitCode = 1;
  } finally {
    chrome.kill();
  }
}

main();
