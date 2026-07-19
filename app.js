/* =========================================================================
   DreamFace — client-side face swap engine
   - Neural path: inswapper_128.onnx via onnxruntime-web (WASM SIMD)
   - Fallback path: JS Delaunay warp (light, no model needed)
   Everything runs on-device. Nothing is uploaded.
   ========================================================================= */

'use strict';

const MODELS_URL = './models';
const INSWAPPER_URL = 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx';
const ORT = window.ort || self.ort;

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const srcDrop = $('srcDrop'), srcFile = $('srcFile'), srcPreview = $('srcPreview');
const tgtDrop = $('tgtDrop'), tgtFile = $('tgtFile'), tgtPreview = $('tgtPreview'), tgtBadge = $('tgtBadge');
const optSmooth = $('optSmooth'), optBlend = $('optBlend'), blendVal = $('blendVal');
const runBtn = $('runBtn'), progressWrap = $('progressWrap'), progressFill = $('progressFill'), progressText = $('progressText');
const resultStep = $('resultStep'), resultMedia = $('resultMedia'), downloadBtn = $('downloadBtn'), resetBtn = $('resetBtn');
const modelDot = $('modelDot'), modelText = $('modelText');
const toastEl = $('toast');

// ── state ────────────────────────────────────────────────────────────────
let srcImg = null;          // HTMLImageElement of source face
let tgtImgOrVideo = null;   // HTMLImageElement | HTMLVideoElement of target
let tgtIsVideo = false;
let neuralReady = false;
let session = null;         // onnx session
let busy = false;

// ── toast ─────────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, isError) {
  toastEl.textContent = msg;
  toastEl.className = 'toast show' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.className = 'toast'; }, 3200);
}

// ── face-api setup ─────────────────────────────────────────────────────────
function setModelStatus(state, text) {
  modelDot.className = 'dot ' + (state || '');
  modelText.textContent = text;
}

async function initFaceApi() {
  setModelStatus('load', 'טוען זיהוי פנים…');
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_URL);
  setModelStatus('ok', 'זיהוי פנים מוכן');
}

// ── onnxruntime session (neural path) ───────────────────────────────────────
async function initNeural() {
  try {
    if (!window.ort) { toast('ספריית עיבוד לא נטענה', true); return; }
    setModelStatus('load', 'טוען מודל נוירוני (~530MB, פעם ראשונה מקוון)…');
    ORT.env.wasm.wasmPaths = './vendor/';
    ORT.env.wasm.simd = true;
    ORT.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    let buf;
    // try cache-first
    try {
      const cache = await caches.open('dreamface-v1');
      let res = await cache.match(INSWAPPER_URL);
      if (!res) {
        res = await fetch(INSWAPPER_URL);
        if (res.ok) cache.put(INSWAPPER_URL, res.clone());
      }
      buf = await res.arrayBuffer();
    } catch (e) {
      const r = await fetch(INSWAPPER_URL);
      buf = await r.arrayBuffer();
    }
    session = await ORT.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    neuralReady = true;
    setModelStatus('ok', 'מודל נוירוני מוכן ✓');
  } catch (e) {
    console.warn('Neural init failed, using fallback:', e);
    neuralReady = false;
    setModelStatus('ok', 'זיהוי פנים מוכן (ללא מודל נוירוני)');
  }
}

// ── file pickers ──────────────────────────────────────────────────────────
function bindPicker(drop, input, onPicked) {
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', () => { if (input.files[0]) onPicked(input.files[0]); });
  ['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('active'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('active'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) { input.files = e.dataTransfer.files; onPicked(f); } });
}

function showPreview(previewEl, url, isVideo) {
  previewEl.innerHTML = '<span class="badge">' + (previewEl.querySelector('.badge') ? previewEl.querySelector('.badge').textContent : '') + '</span>';
  const el = isVideo ? document.createElement('video') : document.createElement('img');
  if (isVideo) { el.muted = true; el.loop = true; el.autoplay = true; el.controls = true; }
  el.src = url;
  previewEl.appendChild(el);
  previewEl.classList.add('show');
  if (isVideo) el.play().catch(() => {});
}

bindPicker(srcDrop, srcFile, (file) => {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { srcImg = img; showPreview(srcPreview, url, false); refreshRun(); };
  img.onerror = () => toast('שגיאה בטעינת התמונה', true);
  img.src = url;
  srcPreview.querySelector('.badge').textContent = 'מקור';
});

bindPicker(tgtDrop, tgtFile, (file) => {
  const url = URL.createObjectURL(file);
  const isVid = file.type.startsWith('video/');
  tgtIsVideo = isVid;
  if (isVid) {
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.onloadeddata = () => { tgtImgOrVideo = v; showPreview(tgtPreview, url, true); tgtBadge.textContent = 'סרטון'; refreshRun(); };
    v.src = url;
  } else {
    const img = new Image();
    img.onload = () => { tgtImgOrVideo = img; showPreview(tgtPreview, url, false); tgtBadge.textContent = 'תמונה'; refreshRun(); };
    img.src = url;
  }
});

optBlend.addEventListener('input', () => { blendVal.textContent = optBlend.value + '%'; });

function refreshRun() {
  runBtn.disabled = !(srcImg && tgtImgOrVideo && !busy);
}

// ── math helpers ───────────────────────────────────────────────────────────
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

// expand a landmark box with margin, align to square
function faceSquare(landmarks, imgW, imgH, margin = 0.35) {
  const pts = landmarks.positions;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
  const w = maxX - minX, h = maxY - minY;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const size = Math.max(w, h) * (1 + margin);
  let x = cx - size / 2, y = cy - size / 2;
  x = clamp(x, 0, imgW); y = clamp(y, 0, imgH);
  const side = Math.min(size, imgW - x, imgH - y);
  return { x, y, size: side };
}

// bilinear crop of a square region into given size → Uint8ClampedArray RGBA
function cropSquare(ctx, box, size) {
  const ix = Math.round(box.x), iy = Math.round(box.y), s = Math.round(box.size);
  const img = ctx.getImageData(ix, iy, s, s);
  // resize to `size` via temp canvas
  const tmp = document.createElement('canvas'); tmp.width = s; tmp.height = s;
  tmp.getContext('2d').putImageData(img, 0, 0);
  const out = document.createElement('canvas'); out.width = size; out.height = size;
  out.getContext('2d').drawImage(tmp, 0, 0, size, size);
  return out;
}

// draw a square region (canvas) back into ctx at box, with feather blend
function pasteSquare(ctx, canvas, box, feather, smooth, blendAmt) {
  const ix = Math.round(box.x), iy = Math.round(box.y), s = Math.round(box.size);
  ctx.save();
  ctx.beginPath();
  ctx.rect(ix, iy, s, s);
  ctx.clip();
  ctx.drawImage(canvas, ix, iy, s, s);
  ctx.restore();

  if (smooth || feather > 0) {
    // feather: blend edges using a soft mask
    const f = Math.max(2, feather);
    const src = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
    const dst = ctx.getImageData(ix, iy, s, s);
    const W = dst.width, H = dst.height;
    const mask = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = Math.min(x, W - 1 - x), dy = Math.min(y, H - 1 - y);
        const d = Math.min(dx, dy);
        let m = d >= f ? 1 : (d / f);
        m = m * m * (3 - 2 * m); // smoothstep
        mask[y * W + x] = m;
      }
    }
    const sd = src.data, dd = dst.data;
    for (let i = 0; i < dd.length; i += 4) {
      const m = mask[i / 4] * (blendAmt / 100);
      for (let c = 0; c < 3; c++) dd[i + c] = sd[i + c] * m + dd[i + c] * (1 - m);
      dd[i + 3] = 255;
    }
    ctx.putImageData(dst, ix, iy);
  }
}

// color/light match: match mean+luminance of swapped face to target region
function matchColor(targetCtx, box, swappedCanvas) {
  const ix = Math.round(box.x), iy = Math.round(box.y), s = Math.round(box.size);
  const tgt = targetCtx.getImageData(ix, iy, s, s);
  const sw = swappedCanvas.getContext('2d').getImageData(0, 0, swappedCanvas.width, swappedCanvas.height);
  const n = tgt.data.length / 4;
  let tm = [0, 0, 0], sm = [0, 0, 0];
  for (let i = 0; i < tgt.data.length; i += 4) {
    for (let c = 0; c < 3; c++) { tm[c] += tgt.data[i + c]; sm[c] += sw.data[i + c]; }
  }
  for (let c = 0; c < 3; c++) { tm[c] /= n; sm[c] /= n; }
  const sd = sw.data;
  for (let i = 0; i < sd.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      let v = sd[i + c] + (tm[c] - sm[c]);
      sd[i + c] = clamp(v, 0, 255);
    }
  }
  swappedCanvas.getContext('2d').putImageData(sw, 0, 0);
}

// ── NEURAL swap (one face) ─────────────────────────────────────────────────
async function neuralSwapFace(sourceCanvas, targetCanvas, tBox) {
  // prepare model input tensors
  const sz = 128;
  // source: crop+bilinear to 128
  const sTmp = document.createElement('canvas'); sTmp.width = sz; sTmp.height = sz;
  sTmp.getContext('2d').drawImage(sourceCanvas, 0, 0, sz, sz);
  // target: crop box to 128
  const tTmp = document.createElement('canvas'); tTmp.width = sz; tTmp.height = sz;
  tTmp.getContext('2d').drawImage(targetCanvas, Math.round(tBox.x), Math.round(tBox.y), Math.round(tBox.size), Math.round(tBox.size), 0, 0, sz, sz);

  const sourceTensor = imgToTensor(sTmp);
  const targetTensor = imgToTensor(tTmp);

  const feeds = {};
  feeds[session.inputNames[0]] = sourceTensor;
  feeds[session.inputNames[1]] = targetTensor;
  const out = await session.run(feeds);
  const result = out[session.outputNames[0]];
  // result shape [1,3,128,128] float -> canvas
  const outCanvas = tensorToCanvas(result, sz);
  return outCanvas;
}

function imgToTensor(canvas) {
  const ctx = canvas.getContext('2d');
  const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const N = canvas.width * canvas.height;
  const data = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    data[i] = id.data[i * 4] / 255;                 // R
    data[N + i] = id.data[i * 4 + 1] / 255;         // G
    data[2 * N + i] = id.data[i * 4 + 2] / 255;     // B
  }
  return new ORT.Tensor('float32', data, [1, 3, canvas.height, canvas.width]);
}

function tensorToCanvas(tensor, size) {
  const data = tensor.data; // float32 [1,3,128,128]
  const N = size * size;
  const canvas = document.createElement('canvas'); canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const id = ctx.createImageData(size, size);
  for (let i = 0; i < N; i++) {
    id.data[i * 4] = clamp(data[i] * 255, 0, 255);
    id.data[i * 4 + 1] = clamp(data[N + i] * 255, 0, 255);
    id.data[i * 4 + 2] = clamp(data[2 * N + i] * 255, 0, 255);
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return canvas;
}

// ── FALLBACK swap (JS Delaunay warp) ────────────────────────────────────────
function getTriangles(pts) {
  const coords = [];
  for (const p of pts) coords.push([p.x, p.y]);
  const d = new Delaunator(coords.flat());
  const tris = [];
  for (let i = 0; i < d.triangles.length; i += 3) {
    tris.push([d.triangles[i], d.triangles[i + 1], d.triangles[i + 2]]);
  }
  return tris;
}

function affine(src, dst) {
  // solve 2x3 transform mapping src tri -> dst tri
  const [x1, y1] = [src[0].x, src[0].y], [x2, y2] = [src[1].x, src[1].y], [x3, y3] = [src[2].x, src[2].y];
  const [u1, v1] = [dst[0].x, dst[0].y], [u2, v2] = [dst[1].x, dst[1].y], [u3, v3] = [dst[2].x, dst[2].y];
  const den = (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
  const a = ((u1 - u3) * (y2 - y3) - (u2 - u3) * (y1 - y3)) / den;
  const b = ((u1 - u3) * (x2 - x3) - (u2 - u3) * (x1 - x3)) / den;
  const c = u1 - a * x1 - b * y1;
  const d2 = ((v1 - v3) * (y2 - y3) - (v2 - v3) * (y1 - y3)) / den;
  const e = ((v1 - v3) * (x2 - x3) - (v2 - v3) * (x1 - x3)) / den;
  const f = v1 - d2 * x1 - e * y1;
  return { a, b, c, d: d2, e, f };
}

function warpTriangle(srcCanvas, dstCtx, srcTri, dstTri) {
  const srcCtx = srcCanvas.getContext('2d');
  // bounding box of dst tri
  const minX = Math.floor(Math.min(dstTri[0].x, dstTri[1].x, dstTri[2].x));
  const minY = Math.floor(Math.min(dstTri[0].y, dstTri[1].y, dstTri[2].y));
  const maxX = Math.ceil(Math.max(dstTri[0].x, dstTri[1].x, dstTri[2].x));
  const maxY = Math.ceil(Math.max(dstTri[0].y, dstTri[1].y, dstTri[2].y));
  const W = Math.max(1, maxX - minX), H = Math.max(1, maxY - minY);
  const t = affine(dstTri, srcTri); // map dst->src
  const out = document.createElement('canvas'); out.width = W; out.height = H;
  const octx = out.getContext('2d');
  const id = octx.createImageData(W, H);
  const sData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height).data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const gx = minX + x, gy = minY + y;
      const sx = Math.round(t.a * gx + t.b * gy + t.c);
      const sy = Math.round(t.d * gx + t.e * gy + t.f);
      if (sx < 0 || sy < 0 || sx >= srcCanvas.width || sy >= srcCanvas.height) continue;
      const si = (sy * srcCanvas.width + sx) * 4;
      const di = (y * W + x) * 4;
      id.data[di] = sData[si]; id.data[di + 1] = sData[si + 1];
      id.data[di + 2] = sData[si + 2]; id.data[di + 3] = 255;
    }
  }
  octx.putImageData(id, 0, 0);
  dstCtx.drawImage(out, minX, minY);
}

function fallbackSwapFace(sourceCanvas, targetCanvas, sPts, tPts) {
  const out = document.createElement('canvas'); out.width = targetCanvas.width; out.height = targetCanvas.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(targetCanvas, 0, 0);
  const sTris = getTriangles(sPts);
  const tTris = getTriangles(tPts);
  ctx.globalCompositeOperation = 'source-over';
  for (let i = 0; i < sTris.length; i++) {
    warpTriangle(sourceCanvas, ctx, sTris[i].map(j => sPts[j]), tTris[i].map(j => tPts[j]));
  }
  return out;
}

// ── detect helper ───────────────────────────────────────────────────────────
async function detectFace(imgOrCanvas, isVideo) {
  const opt = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 });
  const det = await faceapi.detectSingleFace(imgOrCanvas, opt).withFaceLandmarks();
  return det;
}

// ── MAIN process ────────────────────────────────────────────────────────────
async function run() {
  if (busy || !srcImg || !tgtImgOrVideo) return;
  busy = true; refreshRun();
  progressWrap.classList.add('show');
  setProgress(2, 'מזהה פנים…');

  try {
    const blendAmt = parseInt(optBlend.value, 10);
    const smooth = optSmooth.checked;

    // draw source to canvas
    const sCanvas = document.createElement('canvas'); sCanvas.width = srcImg.naturalWidth; sCanvas.height = srcImg.naturalHeight;
    sCanvas.getContext('2d').drawImage(srcImg, 0, 0);

    const sDet = await detectFace(sCanvas);
    if (!sDet) { toast('לא זוהו פנים בתמונת המקור', true); return finish(); }

    if (tgtIsVideo) {
      await runVideo(sCanvas, sDet, blendAmt, smooth);
    } else {
      await runImage(sCanvas, sDet, blendAmt, smooth);
    }
  } catch (e) {
    console.error(e);
    toast('שגיאה בעיבוד: ' + (e.message || e), true);
  } finally {
    finish();
  }
}

function finish() {
  busy = false; refreshRun();
}

function setProgress(pct, msg) {
  progressFill.style.width = pct + '%';
  progressText.textContent = msg;
}

async function runImage(sCanvas, sDet, blendAmt, smooth) {
  const tCanvas = document.createElement('canvas'); tCanvas.width = tgtImgOrVideo.naturalWidth; tCanvas.height = tgtImgOrVideo.naturalHeight;
  tCanvas.getContext('2d').drawImage(tgtImgOrVideo, 0, 0);
  const tDet = await detectFace(tCanvas);
  if (!tDet) { toast('לא זוהו פנים בתמונת היעד', true); return; }

  setProgress(20, 'מחליף פנים…');
  const result = await swapOne(sCanvas, tCanvas, sDet, tDet, blendAmt, smooth);

  setProgress(95, 'מסיים…');
  resultMedia.innerHTML = '';
  const out = document.createElement('img');
  out.src = result.toDataURL('image/jpeg', 0.95);
  resultMedia.appendChild(out);
  resultStep.classList.add('show');
  downloadBtn.onclick = () => downloadCanvas(result, 'dreamface_image.jpg');
  setProgress(100, 'הסתיים ✓');
}

async function runVideo(sCanvas, sDet, blendAmt, smooth) {
  const video = tgtImgOrVideo;
  await new Promise(r => { if (video.readyState >= 2) r(); else video.onloadeddata = r; });
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { toast('לא ניתן לקרוא את הסרטון', true); return; }

  const outCanvas = document.createElement('canvas'); outCanvas.width = vw; outCanvas.height = vh;
  const octx = outCanvas.getContext('2d');

  // setup MediaRecorder to capture outCanvas stream
  const stream = outCanvas.captureStream(0);
  let recorder, chunks = [];
  try {
    recorder = new MediaRecorder(stream, { mimeType: pickMime() });
  } catch (e) {
    toast('הקלטת וידאו לא נתמכת במכשיר — יציג תוצאה פריים בודד', true);
  }
  if (recorder) {
    recorder.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
    recorder.start();
  }

  const tmp = document.createElement('canvas'); tmp.width = vw; tmp.height = vh;
  const tctx = tmp.getContext('2d');

  const fps = 15;
  const frameDt = 1 / fps;
  let frameIdx = 0;
  const maxFrames = 2000;

  while (video.currentTime < video.duration && frameIdx < maxFrames) {
    const target = video.currentTime + frameDt;
    await seekVideo(video, target);
    tctx.drawImage(video, 0, 0, vw, vh);
    const tDet = await detectFace(tmp);
    octx.drawImage(tmp, 0, 0);
    if (tDet) {
      const swapped = await swapOne(sCanvas, tmp, sDet, tDet, blendAmt, smooth);
      octx.drawImage(swapped, 0, 0);
    }
    // push frame to recorder
    if (recorder) {
      // draw to outCanvas already; flush a frame
      outCanvas.getContext('2d').drawImage(octx.canvas, 0, 0);
      stream.getVideoTracks(); // noop
    }
    frameIdx++;
    setProgress(Math.min(95, Math.round((video.currentTime / (video.duration || 1)) * 95)), 'מעבד פריים ' + frameIdx);
    // allow UI to breathe
    await new Promise(r => setTimeout(r, 0));
  }

  let blobUrl = null;
  if (recorder) {
    await new Promise(r => { recorder.onstop = r; recorder.stop(); });
    const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
    blobUrl = URL.createObjectURL(blob);
  }

  setProgress(98, 'בונה תוצאה…');
  resultMedia.innerHTML = '';
  if (blobUrl) {
    const v = document.createElement('video'); v.src = blobUrl; v.controls = true; v.loop = true;
    resultMedia.appendChild(v);
    downloadBtn.onclick = () => { const a = document.createElement('a'); a.href = blobUrl; a.download = 'dreamface_video.webm'; a.click(); };
  } else {
    const img = document.createElement('img'); img.src = outCanvas.toDataURL('image/jpeg', 0.9);
    resultMedia.appendChild(img);
    downloadBtn.onclick = () => downloadCanvas(outCanvas, 'dreamface_frame.jpg');
  }
  resultStep.classList.add('show');
  setProgress(100, 'הסתיים ✓');
}

function pickMime() {
  const cands = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'];
  for (const c of cands) if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  return '';
}

function seekVideo(video, t) {
  return new Promise((resolve) => {
    const onSeek = () => { video.removeEventListener('seeked', onSeek); resolve(); };
    video.addEventListener('seeked', onSeek);
    try { video.currentTime = Math.min(t, video.duration - 0.01); } catch (e) { resolve(); }
  });
}

async function swapOne(sCanvas, tCanvas, sDet, tDet, blendAmt, smooth) {
  if (neuralReady && session) {
    try {
      const sBox = faceSquare(sDet.landmarks, sCanvas.width, sCanvas.height, 0.30);
      const tBox = faceSquare(tDet.landmarks, tCanvas.width, tCanvas.height, 0.30);
      const swapped = await neuralSwapFace(sCanvas, tCanvas, tBox);
      matchColor(tCanvas.getContext('2d'), tBox, swapped);
      const out = document.createElement('canvas'); out.width = tCanvas.width; out.height = tCanvas.height;
      const octx = out.getContext('2d');
      octx.drawImage(tCanvas, 0, 0);
      pasteSquare(octx, swapped, tBox, Math.round(tBox.size * 0.06), smooth, blendAmt);
      return out;
    } catch (e) {
      console.warn('neural swap failed, fallback:', e);
    }
  }
  // fallback JS path
  const sPts = sDet.landmarks.positions;
  const tPts = tDet.landmarks.positions;
  const out = fallbackSwapFace(sCanvas, tCanvas, sPts, tPts);
  if (blendAmt < 100) {
    // reduce strength by blending over original
    const tctx2 = document.createElement('canvas'); tctx2.width = tCanvas.width; tctx2.height = tCanvas.height;
    tctx2.getContext('2d').drawImage(tCanvas, 0, 0);
    const octx = out.getContext('2d');
    octx.globalAlpha = blendAmt / 100;
    octx.drawImage(tCanvas, 0, 0);
    octx.globalAlpha = 1;
  }
  if (smooth) { /* feather already embedded in paste; fallback uses hard paste */ }
  return out;
}

function downloadCanvas(canvas, name) {
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/jpeg', 0.95);
  a.download = name; a.click();
}

resetBtn.addEventListener('click', () => {
  resultStep.classList.remove('show');
  resultMedia.innerHTML = '';
  progressWrap.classList.remove('show');
  progressFill.style.width = '0%';
});

runBtn.addEventListener('click', run);

// ── boot ────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
  try {
    await initFaceApi();
    // load neural in parallel (do not block UI)
    initNeural();
  } catch (e) {
    console.error(e);
    setModelStatus('', 'שגיאה בטעינת מודלים');
    toast('שגיאה בטעינת מודלים — בדוק חיבור לאינטרנט בפעם הראשונה', true);
  }
});

// register service worker for offline
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}
