"use strict";

// Token disuntik ke <meta> oleh backend (same-origin). Dikirim via header untuk
// API, dan via ?token= untuk EventSource/<video> yang tak bisa set header.
const TOKEN = document.querySelector('meta[name="vidnote-token"]').content;
const H = { "X-VidNote-Token": TOKEN };
const STAGE_LABEL = {
  input: "Cek video", audio: "Ekstrak audio", asr: "Transkripsi",
  diarize: "Pemisahan pembicara", segment: "Transkrip & subtitle",
  summary: "Ringkasan", burn: "Hardsub",
};
const STAGE_DESC = {
  input: "Memeriksa sumber dan durasi video",
  audio: "Menyiapkan audio untuk transkripsi",
  asr: "Mengubah ucapan menjadi teks",
  diarize: "Mengenali siapa yang berbicara",
  segment: "Menyusun segmen waktu dan file SRT",
  summary: "Menyusun kesimpulan dan poin penting",
  burn: "Menambahkan subtitle ke video",
};
const LANG_LABEL = { auto: "Deteksi otomatis", id: "Indonesia", en: "English" };
const STAGES = Object.keys(STAGE_LABEL);

const $ = (sel) => document.querySelector(sel);
let currentTab = "url";
let jobId = null;
let evtSource = null;
let currentFile = null;
let jobSourceUrl = null;   // URL yang disubmit (untuk link sumber di layar Hasil)

function fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function parseTs(t) {
  if (!t) return null;
  const p = String(t).split(":").map(Number);
  if (p.some(isNaN)) return null;
  return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p.length === 2 ? p[0] * 60 + p[1] : p[0];
}
const SERVER_DOWN = "VidNote tidak aktif. Nyalakan ulang (klik VidNote.bat) lalu buka kembali halamannya.";
const SESSION_EXPIRED = "Sesi berakhir karena server sempat dimatikan. Buka ulang http://127.0.0.1:8765 lewat jendela VidNote yang baru.";

function friendlyFetchError(e) {
  // "Failed to fetch" / TypeError = browser tak bisa menghubungi server (biasanya mati).
  if (e instanceof TypeError) return SERVER_DOWN;
  return "Gagal mengirim: " + (e && e.message ? e.message : e);
}

function showError(msg) {
  const e = $("#error");
  e.textContent = msg;
  e.classList.remove("hidden");
}
function clearError() { $("#error").classList.add("hidden"); }

// Sorot langkah alur kerja di header (1 Sumber, 2 Proses, 3 Hasil).
function setStep(n) {
  document.querySelectorAll(".wf-step").forEach((s) =>
    s.classList.toggle("active", Number(s.dataset.step) === n));
}

// ---------- info perangkat ----------
async function loadSystem() {
  try {
    const r = await fetch("/system", { headers: H });
    const s = await r.json();
    const gpuRadio = document.querySelector('input[name="device"][value="gpu"]');
    const gpuNote = $("#gpu-note");
    if (s.gpu_available) {
      gpuRadio.disabled = false;
      gpuRadio.checked = true;
      gpuNote.textContent = s.gpu_name || "";
      gpuNote.classList.remove("badge-muted");
    } else {
      gpuRadio.disabled = true;
      gpuNote.textContent = "Tidak tersedia di komputer ini";
      gpuNote.classList.add("badge-muted");   // badge abu-abu, bukan hijau "rekomendasi"
    }
    const gb = (s.max_upload_bytes / 1024 ** 3).toFixed(0);
    // Durasi maks sudah ditampilkan sebagai hint statis di heading section;
    // di sini tampilkan info yang hanya diketahui runtime (batas upload).
    $("#limits").textContent = `Upload file maks ${gb} GB. Satu video diproses dalam satu waktu.`;
    $("#process").disabled = false;
  } catch (e) {
    $("#limits").textContent = SERVER_DOWN;
    $("#process").disabled = true;
  }
}

// ---------- tab & input ----------
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    currentTab = t.dataset.tab;
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".tab-panel").forEach((p) =>
      p.classList.toggle("hidden", p.dataset.panel !== currentTab));
  });
});

const pasteBtn = $("#paste-url");
if (pasteBtn) {
  pasteBtn.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) { $("#url").value = text.trim(); $("#url").focus(); }
    } catch (_) { $("#url").focus(); } // clipboard ditolak: fokuskan input saja
  });
}

const dz = $("#dropzone");
$("#file").addEventListener("change", (e) => { currentFile = e.target.files[0] || null; updateDropText(); });
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
dz.addEventListener("drop", (e) => {
  currentFile = (e.dataTransfer.files || [])[0] || null;
  updateDropText();
});
function updateDropText() {
  $("#dropzone-text").textContent = currentFile ? currentFile.name : "Atau, seret file video ke sini";
}

// ---------- submit ----------
$("#process").addEventListener("click", startJob);

function selectedOptions() {
  const device = document.querySelector('input[name="device"]:checked').value;
  const lang = $("#lang").value;
  const ns = $("#num-speakers").value;
  return { device, lang, num_speakers: ns ? Number(ns) : null };
}

async function startJob() {
  clearError();
  const opts = selectedOptions();
  try {
    if (currentTab === "url") {
      const url = $("#url").value.trim();
      if (!url) return showError("Masukkan link YouTube dulu.");
      jobSourceUrl = url;
      const r = await fetch("/jobs/url", {
        method: "POST",
        headers: { ...H, "Content-Type": "application/json" },
        body: JSON.stringify({ url, ...opts }),
      });
      await handleSubmitResponse(r);
    } else {
      if (!currentFile) return showError("Pilih file video dulu.");
      jobSourceUrl = null;
      await uploadFile(currentFile, opts);
    }
  } catch (e) {
    showError(friendlyFetchError(e));
  }
}

function uploadFile(file, opts) {
  return new Promise((resolve) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("device", opts.device);
    fd.append("lang", opts.lang);
    if (opts.num_speakers) fd.append("num_speakers", opts.num_speakers);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/jobs/upload");
    xhr.setRequestHeader("X-VidNote-Token", TOKEN);
    enterProgress();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setBar("Mengunggah", e.loaded / e.total);
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status === 200) { jobId = data.job_id; listenEvents(); }
      else if (xhr.status === 401) { exitProgress(); showError(SESSION_EXPIRED); }
      else { exitProgress(); showError(detailOf(data) || `Upload gagal (${xhr.status}).`); }
      resolve();
    };
    xhr.onerror = () => { exitProgress(); showError(SERVER_DOWN); resolve(); };
    xhr.send(fd);
  });
}

function detailOf(data) {
  if (!data || !data.detail) return "";
  return Array.isArray(data.detail) ? data.detail.join("; ") : String(data.detail);
}

async function handleSubmitResponse(r) {
  if (r.status === 401) { showError(SESSION_EXPIRED); return; }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { showError(detailOf(data) || `Gagal (${r.status}).`); return; }
  jobId = data.job_id;
  enterProgress();
  listenEvents();
}

// ---------- progres (SSE) ----------
function fillJobDetails() {
  const opts = selectedOptions();
  const srcLabel = $("#job-source-label"), srcVal = $("#job-source-value");
  if (currentTab === "url") {
    if (srcLabel) srcLabel.textContent = "Video dari YouTube";
    if (srcVal) srcVal.textContent = $("#url").value.trim();
  } else {
    if (srcLabel) srcLabel.textContent = "File lokal";
    if (srcVal) srcVal.textContent = currentFile ? currentFile.name : "";
  }
  const dev = $("#job-device"); if (dev) dev.textContent = opts.device.toUpperCase();
  const lang = $("#job-lang"); if (lang) lang.textContent = LANG_LABEL[opts.lang] || opts.lang;
  const spk = $("#job-speakers");
  if (spk) spk.textContent = opts.num_speakers ? `${opts.num_speakers} pembicara` : "Deteksi otomatis";
}

function enterProgress() {
  setStep(2);
  $("#process").disabled = true;
  $("#result-card").classList.add("hidden");
  $("#source-view").classList.add("hidden");
  $("#process-view").classList.remove("hidden");
  fillJobDetails();
  const ol = $("#stages");
  ol.innerHTML = "";
  STAGES.forEach((s, i) => {
    const li = document.createElement("li");
    li.dataset.stage = s;
    const num = document.createElement("span");
    num.className = "stage-num";
    num.textContent = String(i + 1).padStart(2, "0");
    const text = document.createElement("span");
    text.className = "stage-text";
    text.textContent = STAGE_LABEL[s];
    li.append(num, text);
    ol.appendChild(li);
  });
  $("#stage-counter").textContent = `Tahap 1 dari ${STAGES.length}`;
  setBar("", 0);
}
function exitProgress() {
  $("#process-view").classList.add("hidden");
  $("#source-view").classList.remove("hidden");
  $("#process").disabled = false;
  if (evtSource) { evtSource.close(); evtSource = null; }
}
// Catatan: setStep(3) dipanggil di loadResult() saat hasil tampil;
// jika job batal/error, exitProgress kembali ke tampilan Sumber & step ke 1.
function setBar(label, frac) {
  const pct = Math.round(frac * 100);
  $("#bar-fill").style.width = pct + "%";
  const pctEl = $("#progress-pct");
  if (pctEl) pctEl.textContent = pct + "%";
  $("#progress-label").textContent = label || "";
}
function markStage(stage) {
  const items = [...document.querySelectorAll("#stages li")];
  const idx = STAGES.indexOf(stage);
  const counter = $("#stage-counter");
  if (counter && idx >= 0) counter.textContent = `Tahap ${idx + 1} dari ${STAGES.length}`;
  const desc = $("#stage-desc");
  if (desc && idx >= 0) desc.textContent = STAGE_DESC[stage] || "";
  items.forEach((li, i) => {
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
    const text = li.querySelector(".stage-text");
    if (!text) return;
    text.textContent = STAGE_LABEL[STAGES[i]];   // selalu nama tahap saja, tanpa pesan model
  });
}

function listenEvents() {
  evtSource = new EventSource(`/jobs/${jobId}/events?token=${encodeURIComponent(TOKEN)}`);
  evtSource.addEventListener("info", (e) => {
    const d = JSON.parse(e.data);
    const [lo, hi] = d.estimate_min || [];
    setBar(`${d.title} — perkiraan ~${lo}-${hi} menit`, 0);
  });
  evtSource.addEventListener("stage", (e) => {
    const d = JSON.parse(e.data);
    markStage(d.stage);
  });
  evtSource.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    markStage(d.stage);
    setBar(STAGE_LABEL[d.stage] || "", d.fraction);
  });
  evtSource.addEventListener("done", () => { exitProgress(); loadResult(); });
  evtSource.addEventListener("cancelled", () => { exitProgress(); setStep(1); showError("Job dibatalkan."); });
  // Event "error" dari job (punya data JSON) vs koneksi SSE putus (tanpa data).
  evtSource.addEventListener("error", (e) => {
    let msg = null;
    try { const d = JSON.parse(e.data); msg = detailOf(d) || d.message; } catch (_) {}
    const dropped = !msg && evtSource && evtSource.readyState === EventSource.CLOSED;
    exitProgress();
    setStep(1);
    if (msg) showError(msg);                 // error job sebenarnya dari backend
    else if (dropped) showError(SERVER_DOWN); // koneksi putus: server kemungkinan mati
    else showError("Koneksi ke server terputus saat memproses. Coba nyalakan ulang VidNote.");
  });
}

$("#cancel").addEventListener("click", async () => {
  if (!jobId) return;
  await fetch(`/jobs/${jobId}/cancel`, { method: "POST", headers: { ...H, "Origin": location.origin } });
});

// ---------- hasil ----------
async function loadResult() {
  let r;
  try {
    r = await fetch(`/jobs/${jobId}/result`, { headers: H });
  } catch (e) {
    return showError(friendlyFetchError(e));
  }
  if (r.status === 401) return showError(SESSION_EXPIRED);
  if (!r.ok) return showError("Gagal mengambil hasil.");
  const res = await r.json();
  setStep(3);
  $("#process-view").classList.add("hidden");
  $("#source-view").classList.add("hidden");
  $("#result-card").classList.remove("hidden");

  const player = $("#player");
  player.src = `/jobs/${jobId}/video?token=${encodeURIComponent(TOKEN)}`;
  $("#dl-video").href = player.src;
  $("#dl-srt").href = `/jobs/${jobId}/srt?token=${encodeURIComponent(TOKEN)}`;
  $("#preview-note").textContent = res.available.includes("video_hardsub.mp4")
    ? "" : "Video hardsub tidak tersedia.";

  const srcLink = $("#source-link");
  if (srcLink) {
    const row = srcLink.closest(".source-link");
    if (jobSourceUrl) {
      srcLink.href = jobSourceUrl;
      srcLink.textContent = jobSourceUrl.replace(/^https?:\/\//, "");
      if (row) row.classList.remove("hidden");
    } else if (row) {
      row.classList.add("hidden");   // sumber upload: tak ada link
    }
  }

  renderSummary(res.summary);
  const boundaries = (res.summary && res.summary.points || [])
    .map((p) => (p.seconds >= 0 ? p.seconds : parseTs(p.time)))
    .filter((s) => s != null && s > 0);
  renderTranscript(res.transcript, boundaries);
  $("#result-card").scrollIntoView({ behavior: "smooth" });
}

function tsLink(seconds, label) {
  const a = document.createElement("a");
  a.className = "ts-link";
  a.textContent = label;              // textContent: aman XSS
  a.addEventListener("click", (e) => { e.preventDefault(); seekTo(seconds); });
  return a;
}

function renderSummary(summary) {
  const ul = $("#summary-points");
  ul.innerHTML = "";
  if (!summary || !summary.points || !summary.points.length) {
    $("#summary-conclusion").textContent = "Ringkasan tidak tersedia.";
    return;
  }
  summary.points.forEach((p) => {
    const li = document.createElement("li");
    const sec = p.seconds >= 0 ? p.seconds : parseTs(p.time);
    if (sec != null) { li.appendChild(tsLink(sec, `[${p.time || fmtTime(sec)}] `)); }
    li.appendChild(document.createTextNode(p.text));   // teks LLM via textNode
    ul.appendChild(li);
  });
  $("#summary-conclusion").textContent = summary.conclusion || "";
}

// Kelompokkan ulang transkrip: blok baru dimulai saat ganti pembicara ATAU saat
// melewati timestamp poin penting (boundaries). Pakai cues (potongan halus) bila ada,
// supaya pemotongan per poin penting tetap jalan walau cuma satu pembicara.
function groupTranscript(t, boundaries) {
  const units = (t.cues && t.cues.length) ? t.cues : (t.paragraphs || []);
  const bounds = [...(boundaries || [])].filter((s) => s > 0).sort((a, b) => a - b);
  // Nomor "ruas" waktu tempat sebuah detik berada = jumlah boundary yang <= detik itu.
  // Unit yang berada di ruas berbeda dari blok sebelumnya memulai blok baru.
  const seg = (sec) => bounds.filter((b) => b <= sec).length;
  const blocks = [];
  for (const u of units) {
    const prev = blocks[blocks.length - 1];
    const newBlock = !prev || prev.speaker !== u.speaker || seg(u.start) !== seg(prev.start);
    if (newBlock) {
      blocks.push({ start: u.start, end: u.end, speaker: u.speaker, text: u.text });
    } else {
      prev.end = u.end;
      prev.text = `${prev.text} ${u.text}`;
    }
  }
  return blocks;
}

function renderTranscript(t, boundaries) {
  const box = $("#transcript");
  box.innerHTML = "";
  if (!t || (!t.paragraphs && !t.cues)) { box.textContent = "Transkrip tidak tersedia."; return; }
  const labels = {};
  (t.speakers || []).forEach((s) => { labels[s.id] = s.label; });
  const blocks = groupTranscript(t, boundaries);
  blocks.forEach((p) => {
    const div = document.createElement("div");
    div.className = "seg";
    div.dataset.start = p.start;
    div.dataset.end = p.end;

    const time = document.createElement("span");
    time.className = "time";
    time.textContent = fmtTime(p.start);

    const spk = document.createElement("span");
    spk.className = "spk";
    spk.textContent = (labels[p.speaker] || `Pembicara ${p.speaker + 1}`) + ":";
    // Warna label dibiarkan default (gelap) agar terbaca di panel terang.
    // Palet warna-warni hanya untuk subtitle di video (backend/pipeline/subtitle.py).

    const txt = document.createElement("span");
    txt.textContent = p.text;          // textContent: aman XSS

    div.append(time, spk, txt);
    div.addEventListener("click", () => seekTo(p.start));
    box.appendChild(div);
  });
}

function seekTo(sec) {
  const player = $("#player");
  player.currentTime = sec;
  player.play().catch(() => {});
  player.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Sorot segmen transkrip sesuai posisi video.
$("#player").addEventListener("timeupdate", () => {
  const t = $("#player").currentTime;
  document.querySelectorAll(".seg").forEach((seg) => {
    const on = t >= Number(seg.dataset.start) && t < Number(seg.dataset.end);
    seg.classList.toggle("playing", on);
  });
});

loadSystem();
