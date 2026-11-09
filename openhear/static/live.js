/* M2-alpha research console: local VAD segmentation + whole-window provisional ASR.
   This is not native streaming Whisper or simultaneous interpretation. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(['liveStart','liveStop','liveBadge','liveAutoplay','liveStatus',
    'liveStats','liveStable','livePartial','liveHistory','liveExport','liveClear',
    'direction','consent','record','stop','outputDevice','waveform','level','timer',
    'transcript','translation','sourceLang','targetLang','metrics','playback','ttsState'].map(k => [k, $(k)]));
  const fallbackPolicy = {speech_on_rms:0.018, speech_off_rms:0.012, min_speech_ms:420,
    pre_roll_ms:260,long_utterance_ms:3500,pause_short_ms:750,pause_long_ms:520,
    max_segment_ms:11000,preview_after_ms:1800,preview_every_ms:2500,max_pending_segments:2};
  const state = {listening:false, opening:false, mic:null, context:null, source:null, node:null, silent:null,
    rate:48000, policy:fallbackPolicy, rolling:[], rollingSamples:0, frames:[], samples:0,
    speech:false, hot:0, quietMs:0, voicedMs:0, lastPreviewAt:0, previewPromise:null,
    latestPreview:'', previousPreview:'', sequence:0, queue:[], worker:false, turns:[],
    dropped:0, errors:0, audit:[], auditOverflow:0, currentPlayback:null, playedUrls:[], epoch:0, pttBusy:false, peaks:[], startAt:0, revoked:false};
  let historyPlaceholder = true;
  const setStatus = (value, error=false) => {ui.liveStatus.textContent=value;ui.liveStatus.classList.toggle('live-error',error);};
  function update() {
    ui.liveStart.disabled = !ui.consent.checked || state.listening || state.opening || state.worker || state.queue.length>0 || state.pttBusy;
    ui.liveStop.disabled = !state.listening && !state.opening;
    ui.liveBadge.textContent = state.listening ? (state.speech ? 'SPEECH DETECTED' : 'LISTENING') : state.worker ? 'TRANSLATING' : 'STOPPED';
    ui.liveExport.disabled = state.audit.length === 0;
    ui.liveClear.disabled = state.listening || state.worker;
    ui.liveStats.textContent = `${state.audit.length} recorded · ${state.queue.length} queued · ${state.dropped} dropped · ${state.errors} errors`;
  }
  function broadcast() {document.dispatchEvent(new CustomEvent('openhear:live-change',{detail:{active:state.listening || state.opening || state.worker || state.queue.length>0}}));update();}
  const durationMs = count => count*1000/state.rate;
  function showSignal(frame, rms) {
    const canvas=ui.waveform,ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    let peak=0;for(const x of frame)peak=Math.max(peak,Math.abs(x));
    state.peaks.push(peak);if(state.peaks.length>120)state.peaks.shift();
    ctx.clearRect(0,0,w,h);ctx.strokeStyle='#2b4357';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
    ctx.strokeStyle=state.speech?'#80e5c6':'#648ca4';ctx.lineWidth=3;
    const bar=w/120;
    for(let i=0;i<state.peaks.length;i++){
      const amp=Math.max(1,Math.min(.94,state.peaks[i]*3)*h*.46);
      ctx.beginPath();ctx.moveTo(i*bar+bar/2,h/2-amp);ctx.lineTo(i*bar+bar/2,h/2+amp);ctx.stroke();
    }
    ui.level.textContent=`LIVE RMS ${Math.round(100*rms)}% · PEAK ${Math.round(100*peak)}%`;
    const t=Math.floor((performance.now()-state.startAt)/1000);
    ui.timer.textContent=`${String(Math.floor(t/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`;
    if (state.listening) {
      const bars = ui.liveStart.querySelectorAll('.v-bar');
      bars.forEach((bar, idx) => {
        const mult = state.speech ? Math.min(2.8, Math.max(0.6, rms * 16 * (0.8 + 0.4 * Math.sin(idx)))) : 0.6;
        bar.style.transform = `scaleY(${mult})`;
      });
    }
  }
  function monoRms(frame) {
    let power=0;for(let i=0;i<frame.length;i++)power+=frame[i]*frame[i];
    return Math.sqrt(power/Math.max(1,frame.length));
  }
  function makeWav(frames, sampleCount) {
    const max = Math.round(28*state.rate);
    if(sampleCount<=0 || sampleCount>max)throw new Error('Audio turn outside allowed capture limit');
    const samples=new Float32Array(sampleCount);let at=0;
    for(const f of frames){samples.set(f,at);at+=f.length;}
    const bytes = new ArrayBuffer(44+sampleCount*2), view=new DataView(bytes);
    const fourcc=(i,s)=>{for(let j=0;j<s.length;j++)view.setUint8(i+j,s.charCodeAt(j));};
    fourcc(0,'RIFF');view.setUint32(4,36+sampleCount*2,true);
    fourcc(8,'WAVE');fourcc(12,'fmt ');view.setUint32(16,16,true);
    view.setUint16(20,1,true);view.setUint16(22,1,true);
    view.setUint32(24,state.rate,true);view.setUint32(28,state.rate*2,true);
    view.setUint16(32,2,true);view.setUint16(34,16,true);
    fourcc(36,'data');view.setUint32(40,sampleCount*2,true);
    for(let i=0;i<samples.length;i++)view.setInt16(44+i*2,Math.round(Math.max(-1,Math.min(1,samples[i]))*32767),true);
    return new Blob([bytes],{type:'audio/wav'});
  }
  function resetSpeech() {
    state.frames=[];state.samples=0;state.speech=false;
    state.quietMs=0;state.voicedMs=0;state.latestPreview='';state.previousPreview='';
    state.lastPreviewAt=0;
    ui.liveStable.textContent='Waiting for speech…';ui.livePartial.textContent='';
    update();
  }
  function pushRolling(frame) {
    state.rolling.push(frame);state.rollingSamples+=frame.length;
    const allowed=Math.ceil(state.policy.pre_roll_ms*state.rate/1000);
    while(state.rolling.length>1 && state.rollingSamples-state.rolling[0].length>=allowed){
      state.rollingSamples-=state.rolling.shift().length;
    }
  }
  function stablePrefix(a,b,language) {
    if(language==='zh') {
      let i=0;while(i<a.length && i<b.length && a[i]===b[i])i++;
      return a.slice(0,i).trim();
    }
    const left=a.trim().split(/\s+/), right=b.trim().split(/\s+/);let i=0;
    while(i<left.length && i<right.length && left[i].toLowerCase()===right[i].toLowerCase())i++;
    return left.slice(0,i).join(' ');
  }
  async function previewCurrent() {
    if(!state.listening || !state.speech || state.previewPromise || state.worker || state.queue.length)return;
    if(durationMs(state.samples) > state.policy.max_segment_ms) return;
    const frames=state.frames.slice(), size=state.samples, epoch=state.epoch, sequence=state.sequence;
    const blob=makeWav(frames,size);
    const source=ui.direction.value.split(':')[0];
    const form=new FormData();form.append('source',source);form.append('audio',blob,'preview.wav');
    state.previewPromise=(async()=>{
      try {
        const r=await fetch('/api/preview',{method:'POST',body:form,cache:'no-store'});
        if(!r.ok)return;
        const result=await r.json();
        if(!state.listening || !state.speech || state.epoch!==epoch || sequence!==state.sequence)return;
        const current=(result.transcript||'').trim();
        if(!current)return;
        const stable=stablePrefix(state.latestPreview,current,source);
        state.previousPreview=state.latestPreview;state.latestPreview=current;
        ui.liveStable.textContent=stable || 'Provisional hypothesis (unconfirmed)';
        ui.livePartial.textContent=current;
      } catch(_) { /* provisional inference is optional; final turns remain authoritative */ }
      finally {state.previewPromise=null;}
    })();
  }
  function onPCM(frameInput) {
    if(!state.listening)return;
    const frame=frameInput instanceof Float32Array ? frameInput : new Float32Array(frameInput);
    const rms=monoRms(frame), dt=durationMs(frame.length);
    showSignal(frame,rms);
    pushRolling(frame);
    if(!state.speech) {
      state.hot = rms >= state.policy.speech_on_rms ? state.hot+1 : 0;
      if(state.hot>=2) {
        state.speech=true;state.sequence++;
        if(state.currentPlayback){state.currentPlayback.pause();state.currentPlayback.currentTime=0;state.currentPlayback=null;}
        state.frames=state.rolling.slice();state.samples=state.rollingSamples;
        state.voicedMs=2*dt;state.quietMs=0;state.lastPreviewAt=0;
        setStatus('Speech detected; listening for a natural pause.');update();
      }
      return;
    }
    state.frames.push(frame);state.samples+=frame.length;
    state.voicedMs += rms >= state.policy.speech_off_rms ? dt : 0;
    state.quietMs = rms < state.policy.speech_off_rms ? state.quietMs+dt : 0;
    const elapsed=durationMs(state.samples);
    const pause=elapsed>=state.policy.long_utterance_ms?state.policy.pause_long_ms:state.policy.pause_short_ms;
    if(elapsed>=state.policy.max_segment_ms) {finalizeTurn('max-duration');return;}
    if(state.quietMs>=pause && state.voicedMs>=state.policy.min_speech_ms) {finalizeTurn('pause-commit');return;}
    if(elapsed>=state.policy.preview_after_ms && (state.lastPreviewAt===0 || elapsed-state.lastPreviewAt>=state.policy.preview_every_ms)) {
      state.lastPreviewAt=elapsed;void previewCurrent();
    }
  }
  function finalizeTurn(reason) {
    if(!state.speech)return;
    const valid=state.voicedMs>=state.policy.min_speech_ms;
    const frames=state.frames.slice(),count=state.samples, epoch=state.epoch;
    resetSpeech();state.hot=0;
    if(!valid || count===0)return;
    let blob;
    try {blob=makeWav(frames,count);}catch(e){setStatus(`Turn discarded: ${e.message}`,true);return;}
    if(state.queue.length>=state.policy.max_pending_segments){state.dropped++;setStatus('Inference backlog: a completed turn was dropped. Repeat it.',true);update();return;}
    const id=++state.sequence;
    state.queue.push({id,epoch,blob,committed_at_ms:performance.now(),reason,duration_ms:Math.round(durationMs(count)),
                      source:ui.direction.value.split(':')[0],target:ui.direction.value.split(':')[1],
                      captured_utc:new Date().toISOString()});
    setStatus(`Turn ${id} ready (${reason}); translating locally.`);
    void drainQueue();broadcast();
  }
  function addTurn(item,result,error,requestMs) {
    if(historyPlaceholder){ui.liveHistory.replaceChildren();historyPlaceholder=false;}
    const element=document.createElement('article');element.className='live-turn';
    const header=document.createElement('header');
    const heading=document.createElement('span');heading.textContent=`${item.source.toUpperCase()} → ${item.target.toUpperCase()} · ${item.reason}`;
    const time=document.createElement('span');time.textContent=new Date(item.captured_utc).toLocaleTimeString();
    header.append(heading,time);element.append(header);
    const source=document.createElement('p'),target=document.createElement('p');
    source.textContent=error?`Turn failed: ${error}`:result.transcript;
    if(error)source.className='live-error';
    target.className='translation';target.textContent=error?'Translation unavailable':result.translation;
    element.append(source,target);
    const timings=document.createElement('p');timings.className='evidence-timing';
    timings.textContent=error?`Browser request: ${requestMs} ms; failed`:
      `ASR ${result.timings_ms.asr} ms · MT ${result.timings_ms.mt} ms · TTS ${result.timings_ms.tts} ms · request ${requestMs} ms · pause→result ${Math.round(performance.now()-item.committed_at_ms)} ms`;
    element.append(timings);
    let audioUrl=null;
    if(result && result.audio_base64){
      try {
        const raw=atob(result.audio_base64), bytes=new Uint8Array(raw.length);
        for(let k=0;k<raw.length;k++)bytes[k]=raw.charCodeAt(k);
        audioUrl=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));
        const player=document.createElement('audio');player.controls=true;player.src=audioUrl;
        element.append(player);
        const output=ui.outputDevice.value;
        if(output!=='default' && typeof player.setSinkId==='function')void player.setSinkId(output).catch(()=>{});
        if(ui.liveAutoplay.checked){
          player.play().then(()=>{state.currentPlayback=player;}).catch(()=>{
            setStatus('Auto-play blocked by the browser; press play on the translated audio.');
          });
        }
      }catch(e){setStatus(`Generated audio could not be decoded: ${e.message}`,true);}
    }
    const record={id:item.id,source:item.source,target:item.target,reason:item.reason,duration_ms:item.duration_ms,
      captured_utc:item.captured_utc,status:error?'failed':result.status,request_ms:requestMs,
      pause_to_result_ms:Math.round(performance.now()-item.committed_at_ms),
      transcript:error?null:result.transcript,translation:error?null:result.translation,
      timings_ms:error?null:result.timings_ms,tts_status:error?'unavailable':result.tts_status,
      error:error||null, audio_url:audioUrl};
    state.audit.push({id:record.id,source:record.source,target:record.target,reason:record.reason,
      duration_ms:record.duration_ms,captured_utc:record.captured_utc,status:record.status,
      request_ms:record.request_ms,pause_to_result_ms:record.pause_to_result_ms,
      timings_ms:record.timings_ms,tts_status:record.tts_status,error:record.error});
    while(state.audit.length>500){state.audit.shift();state.auditOverflow++;}
    // Keep the primary translation panels synchronized with the most recent
    // finalized live turn; failed turns do not overwrite previously valid text.
    if(!error){
      ui.transcript.textContent=result.transcript;ui.transcript.classList.remove('placeholder');
      ui.translation.textContent=result.translation;ui.translation.classList.remove('placeholder');
      ui.sourceLang.textContent=item.source==='en'?'ENGLISH':'MANDARIN CHINESE';
      ui.targetLang.textContent=item.target==='en'?'ENGLISH':'MANDARIN CHINESE';
      ui.metrics.textContent=`Live · ASR ${result.timings_ms.asr} ms · MT ${result.timings_ms.mt} ms · TTS ${result.timings_ms.tts} ms · pause→result ${record.pause_to_result_ms} ms`;
      ui.ttsState.textContent=audioUrl?'Translated audio ready: press play in history or below.':
        result.tts_status==='unavailable'?'Speech synthesis unavailable for this turn.':'Text complete; no speech audio generated.';
      if(audioUrl){ui.playback.src=audioUrl;ui.playback.hidden=false;}
      else {ui.playback.pause();ui.playback.removeAttribute('src');ui.playback.hidden=true;}
    }
    state.turns.push(record);ui.liveHistory.prepend(element);
    while(state.turns.length>20){
      const removed=state.turns.shift();
      if(removed.audio_url)URL.revokeObjectURL(removed.audio_url);
      if(ui.liveHistory.lastElementChild)ui.liveHistory.lastElementChild.remove();
    }
    update();
  }
  async function translateOne(item) {
    // A preview may still own the ASR model lock. Wait for it to finish first.
    if(state.previewPromise)await state.previewPromise;
    const form=new FormData();form.append('source',item.source);form.append('target',item.target);
    form.append('audio',item.blob,`turn-${item.id}.wav`);
    const started=performance.now();
    for(let attempt=0;attempt<4;attempt++) {
      try {
        const r=await fetch('/api/translate',{method:'POST',body:form,cache:'no-store'});
        if(r.status===429 && attempt<3){await new Promise(resolve=>setTimeout(resolve,350*(attempt+1)));continue;}
        const payload=await r.json();
        if(!r.ok)throw new Error(typeof payload.detail==='string'?payload.detail:`HTTP ${r.status}`);
        return {result:payload,error:null,ms:Math.round(performance.now()-started)};
      }catch(error){
        if(attempt===3 || error.message!=='Failed to fetch')return {result:null,error:String(error.message),ms:Math.round(performance.now()-started)};
        await new Promise(resolve=>setTimeout(resolve,350*(attempt+1)));
      }
    }
    return {result:null,error:'Retry budget exhausted',ms:Math.round(performance.now()-started)};
  }
  async function drainQueue() {
    if(state.worker)return;
    state.worker=true;broadcast();
    try {
      while(state.queue.length) {
        const item=state.queue.shift();
        if(state.revoked)continue;
        const answer=await translateOne(item);
        if(state.revoked)continue;
        if(answer.error)state.errors++;
        addTurn(item,answer.result,answer.error,answer.ms);
        setStatus(answer.error?`Turn ${item.id} failed: ${answer.error}`:`Turn ${item.id} translated; review the result.`,Boolean(answer.error));
        update();
      }
    } finally {state.worker=false;broadcast();}
  }
  async function startLive() {
    if(!ui.consent.checked || state.listening || state.opening || state.worker || state.pttBusy)return;
    state.opening=true;broadcast();
    try {
      const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:false}});
      state.mic=stream;
      const ctx=new (window.AudioContext||window.webkitAudioContext)();state.context=ctx;state.rate=ctx.sampleRate;
      const source=ctx.createMediaStreamSource(stream);state.source=source;
      const silent=ctx.createGain();silent.gain.value=0;state.silent=silent;
      if(ctx.audioWorklet && window.AudioWorkletNode) {
        try {
          await ctx.audioWorklet.addModule('/static/capture-worklet.js');
          const node=new AudioWorkletNode(ctx,'openhear-capture',{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]});
          node.port.onmessage=e=>onPCM(e.data);state.node=node;
        } catch(_) {state.node=null;}
      }
      if(!state.node){
        const node=ctx.createScriptProcessor(2048,1,1);
        node.onaudioprocess=e=>onPCM(new Float32Array(e.inputBuffer.getChannelData(0)));
        state.node=node;
        setStatus('Using legacy audio capture fallback (AudioWorklet unavailable).');
      }
      source.connect(state.node);state.node.connect(silent);silent.connect(ctx.destination);
      await ctx.resume();
      state.epoch++;state.revoked=false;state.listening=true;state.startAt=performance.now();state.peaks=[];resetSpeech();state.rolling=[];state.rollingSamples=0;
      setStatus('Microphone active. Speak a sentence, then pause to translate.');
    } catch(error) {
      await releaseMic();setStatus(`Could not start continuous microphone: ${error.message}`,true);
    } finally {state.opening=false;broadcast();}
  }
  async function releaseMic() {
    if(state.mic){state.mic.getTracks().forEach(t=>t.stop());state.mic=null;}
    if(state.node){try{state.node.disconnect();}catch(_){}state.node=null;}
    if(state.source){try{state.source.disconnect();}catch(_){}state.source=null;}
    if(state.silent){try{state.silent.disconnect();}catch(_){}state.silent=null;}
    const context=state.context;state.context=null;
    if(context)await context.close().catch(()=>{});
  }
  async function stopLive(discard=false) {
    if(!state.listening && !state.opening && !discard)return;
    if(discard) {
      state.revoked=true;state.queue=[];resetSpeech();
      state.rolling=[];state.rollingSamples=0;
    } else if(state.speech)finalizeTurn('manual-stop');
    state.listening=false;state.epoch++;
    await releaseMic();
    setStatus(discard?'Consent unchecked: audio capture and queued turns discarded. An already-running local request may finish.':
      state.worker||state.queue.length?'Listening stopped; completing queued translation.':'Continuous microphone stopped.');
    broadcast();
  }
  function clearLive() {
    if(state.listening || state.worker || state.queue.length)return;
    for(const turn of state.turns)if(turn.audio_url)URL.revokeObjectURL(turn.audio_url);
    state.turns=[];state.audit=[];state.auditOverflow=0;state.dropped=0;state.errors=0;state.currentPlayback=null;
    ui.liveHistory.replaceChildren();const p=document.createElement('p');p.className='subdued';p.textContent='No recorded turns in this session.';ui.liveHistory.append(p);historyPlaceholder=true;
    resetSpeech();setStatus('Live session cleared from this browser.');broadcast();
  }
  function exportLive() {
    if(!state.audit.length)return;
    const data={schema_version:'0.4-live-client',utc:new Date().toISOString(),
      note:'Contains speech transcripts. Software times only. Provisional ASR is not a verified translation.',
      vad:state.policy,dropped:state.dropped,errors:state.errors,audit_overflow:state.auditOverflow,
      audit_records:state.audit,
      recent_full_text_turns:state.turns.map(({audio_url,...turn})=>turn)};
    const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=`openhear-live-${new Date().toISOString().replaceAll(':','-')}.json`;
    a.click();setTimeout(()=>URL.revokeObjectURL(url),1200);
  }
  document.addEventListener('openhear:ptt-change',e => {state.pttBusy=Boolean(e.detail?.active);update();});
  ui.liveStart.addEventListener('click',startLive);
  ui.liveStop.addEventListener('click',()=>{void stopLive();});
  ui.liveClear.addEventListener('click',clearLive);
  ui.liveExport.addEventListener('click',exportLive);
  ui.consent.addEventListener('change',()=>{if(!ui.consent.checked && (state.listening || state.worker || state.queue.length))void stopLive(true);update();});
  window.addEventListener('beforeunload',()=>{if(state.mic)state.mic.getTracks().forEach(t=>t.stop());for(const turn of state.turns)if(turn.audio_url)URL.revokeObjectURL(turn.audio_url);});
  fetch('/api/stream/config',{cache:'no-store'}).then(r=>r.ok?r.json():null).then(data=>{if(data?.vad)state.policy=data.vad;}).catch(()=>{});
  update();
})();
