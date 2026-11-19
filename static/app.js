(function () {
  const $ = (id) => document.getElementById(id);
  const promptEl = $("prompt"), cfgEl = $("cfg"), cfgVal = $("cfg-val");
  const framesEl = $("frames"), resolutionEl = $("resolution"), fpsEl = $("fps"), seedEl = $("seed");
  const generateBtn = $("generate-btn"), progressWrap = $("progress-wrap");
  const progressFill = $("progress-fill"), statusEl = $("status"), errorEl = $("error");
  const videoCard = $("video-card"), videoEl = $("video"), downloadEl = $("download");

  cfgEl.addEventListener("input", () => { cfgVal.textContent = cfgEl.value; });
  let pollTimer = null;

  function setError(msg) {
    if (!msg) { errorEl.hidden = true; errorEl.textContent = ""; return; }
    errorEl.hidden = false; errorEl.textContent = msg;
  }
  function setProgress(pct, msg) {
    progressWrap.hidden = false;
    progressFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (msg) statusEl.textContent = msg;
  }
  function resetUI() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    setError(""); progressWrap.hidden = true;
    progressFill.style.width = "0%"; statusEl.textContent = "";
    videoCard.hidden = true; generateBtn.disabled = false;
  }
  async function poll(jobId) {
    try {
      const res = await fetch(`/progress?job_id=${encodeURIComponent(jobId)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "progress error");
      const pct = Math.round((data.progress || 0) * 100);
      setProgress(pct, `${data.status} — ${data.message || ""}`);
      if (data.status === "completed") {
        clearInterval(pollTimer); pollTimer = null;
        videoEl.src = `${data.video_url}?t=${Date.now()}`;
        videoEl.load(); videoEl.play().catch(() => {});
        downloadEl.href = data.video_url;
        downloadEl.setAttribute("download", "generated_video.mp4");
        videoCard.hidden = false; generateBtn.disabled = false;
        statusEl.textContent = "Completed";
      } else if (data.status === "failed") {
        clearInterval(pollTimer); pollTimer = null;
        setError(data.error || "Generation failed");
        generateBtn.disabled = false;
      }
    } catch (err) {
      clearInterval(pollTimer); pollTimer = null;
      setError(err.message || String(err));
      generateBtn.disabled = false;
    }
  }
  async function generate() {
    resetUI(); generateBtn.disabled = true;
    const parsedSeed = parseInt(seedEl.value, 10);
    const payload = {
      prompt: promptEl.value.trim(),
      cfg_scale: parseFloat(cfgEl.value),
      num_frames: parseInt(framesEl.value, 10),
      image_size: parseInt(resolutionEl.value, 10),
      fps: parseInt(fpsEl.value, 10),
      seed: Number.isFinite(parsedSeed) ? parsedSeed : 42,
    };
    if (!payload.prompt) { setError("Please enter a prompt."); generateBtn.disabled = false; return; }
    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "request failed");
      setProgress(1, "queued");
      pollTimer = setInterval(() => poll(data.job_id), 1200);
      poll(data.job_id);
    } catch (err) {
      setError(err.message || String(err)); generateBtn.disabled = false;
    }
  }
  generateBtn.addEventListener("click", generate);
})();
