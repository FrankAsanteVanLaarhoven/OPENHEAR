(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const controls = ['record','stop','upload','file','consent','direction','saveRecording','clear','saveEvidence'];
  const ui = Object.fromEntries([...controls,'status','waveform','level','timer','captureState','sourcePlayback','playback','transcript','translation','sourceLang','targetLang','ttsState','metrics','outputDevice','refreshOutputs','voiceState'].map(x => [x, $(x)]));
  const state = {active:false, busy:false, mic:null, context:null, processor:null, src:null, analyser:null,
                 mute:null, blocks:[], rate:48000, start:0, raf:0, peaks:[], inputUrl:null,
                 outputUrl:null, lastWav:null, lastEvidence:null, recordStarted:null, liveActive:false};
  const chart = ui.waveform.getContext('2d');
  const setStatus = (message, error=false) => { ui.status.textContent = message; ui.status.classList.toggle('error', error); };
  const clearUrl = key => { if (state[key]) { URL.revokeObjectURL(state[key]); state[key] = null; } };
  function update() {
    const allowed = ui.consent.checked;
    ui.record.disabled = !allowed || state.active || state.busy || state.liveActive;
    ui.stop.disabled = !state.active;
    ui.upload.disabled = !allowed || state.active || state.busy || state.liveActive || !ui.file.files.length;
    ui.file.disabled = !allowed || state.active || state.busy || state.liveActive;
    ui.direction.disabled = state.active || state.busy || state.liveActive;
    ui.saveRecording.disabled = !state.lastWav || state.active;
    ui.saveEvidence.disabled = !state.lastEvidence || state.active;
    ui.captureState.textContent = state.active ? 'RECORDING' : state.busy ? 'TRANSLATING' : state.liveActive ? 'CONTINUOUS MODE' : 'READY';
    document.dispatchEvent(new CustomEvent('openhear:ptt-change',{detail:{active:state.active || state.busy}}));
  }
  function setDirection(){
    const [s,t] = ui.direction.value.split(':');
    ui.sourceLang.textContent = s==='en' ? 'ENGLISH' : 'MANDARIN CHINESE';
    ui.targetLang.textContent = t==='en' ? 'ENGLISH' : 'MANDARIN CHINESE';
  }
  function waveform() {
    const w=chart.canvas.width,h=chart.canvas.height;
    chart.clearRect(0,0,w,h);
    chart.strokeStyle='#253c52';chart.lineWidth=1;
    chart.beginPath();chart.moveTo(0,h/2);chart.lineTo(w,h/2);chart.stroke();
    chart.strokeStyle='#83e7c7';chart.lineWidth=3;
    const bars=state.peaks, n=bars.length, barwidth=w/120;
    for(let i=0;i<n;i++){
      const magnitude=Math.min(.94,bars[i]*2.6), amplitude=Math.max(1,magnitude*h*.48);
      chart.beginPath();chart.moveTo(i*barwidth+barwidth/2,h/2-amplitude);
      chart.lineTo(i*barwidth+barwidth/2,h/2+amplitude);chart.stroke();
    }
    if (state.active) state.raf=requestAnimationFrame(waveform);
  }
  const formatElapsed = ms => {
    const t=Math.floor(ms/1000);return `${String(Math.floor(t/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`;
  };
  function drawLevel(samples){
    let sum=0,peak=0;for(const x of samples){sum+=x*x;peak=Math.max(peak,Math.abs(x));}
    const rms=Math.sqrt(sum/Math.max(1,samples.length));
    state.peaks.push(Math.min(1,peak));if(state.peaks.length>120)state.peaks.shift();
    ui.level.textContent = `RMS ${Math.round(100*rms)}% · PEAK ${Math.round(100*peak)}%`;
    ui.timer.textContent = formatElapsed(Date.now()-state.start);
  }
  async function stopMicrophone() {
    if(!state.active)return null;
    state.active=false;
    if(state.raf)cancelAnimationFrame(state.raf);
    if(state.mic){state.mic.getTracks().forEach(x=>x.stop());state.mic=null;}
    if(state.src)state.src.disconnect();
    if(state.processor){state.processor.disconnect();state.processor.onaudioprocess=null;}
    if(state.analyser)state.analyser.disconnect();
    if(state.mute)state.mute.disconnect();
    const context=state.context;state.context=null;
    if(context)await context.close();
    const blocks=state.blocks;state.blocks=[];
    const count=blocks.reduce((n,b)=>n+b.length,0);
    if(count < state.rate/3) { update();setStatus('Recording too short. Please record at least one second.',true);return null; }
    const samples=new Float32Array(count);let offset=0;
    for(const b of blocks){samples.set(b,offset);offset+=b.length;}
    const blob=makeWav(samples,state.rate);
    clearUrl('inputUrl');state.inputUrl=URL.createObjectURL(blob);
    ui.sourcePlayback.src=state.inputUrl;ui.sourcePlayback.hidden=false;
    state.lastWav=blob;waveform();update();
    return blob;
  }
  function makeWav(samples,rate){
    const bytes=new ArrayBuffer(44+samples.length*2),v=new DataView(bytes);
    const text=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
    text(0,'RIFF');v.setUint32(4,36+samples.length*2,true);text(8,'WAVE');text(12,'fmt ');
    v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
    v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);
    v.setUint16(34,16,true);text(36,'data');v.setUint32(40,samples.length*2,true);
    for(let i=0;i<samples.length;i++)v.setInt16(44+2*i,Math.round(Math.max(-1,Math.min(1,samples[i]))*32767),true);
    return new Blob([bytes],{type:'audio/wav'});
  }
  async function startRecording(){
    if(!ui.consent.checked || state.active || state.busy || state.liveActive)return;
    try {
      const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:false}});
      state.mic=stream;
      const context=new (window.AudioContext||window.webkitAudioContext)();state.context=context;
      state.rate=context.sampleRate;state.blocks=[];state.peaks=[];
      const source=context.createMediaStreamSource(stream),analyser=context.createAnalyser();analyser.fftSize=2048;
      const processor=context.createScriptProcessor(2048,1,1),mute=context.createGain();mute.gain.value=0;
      source.connect(analyser);analyser.connect(processor);processor.connect(mute);mute.connect(context.destination);
      state.src=source;state.analyser=analyser;state.processor=processor;state.mute=mute;
      state.start=Date.now();state.active=true;
      processor.onaudioprocess=e=>{
        if(!state.active)return;
        const data=e.inputBuffer.getChannelData(0);
        state.blocks.push(new Float32Array(data));drawLevel(data);
        if(Date.now()-state.start>=28000){void finishRecording();}
      };
      waveform();update();setStatus('Recording: speak clearly, up to 28 seconds.');
      await refreshOutputs();
    } catch(e) {
      if(state.mic){state.mic.getTracks().forEach(x=>x.stop());state.mic=null;}
      state.active=false;update();setStatus(`Microphone permission or audio initialization failed: ${e.message}`,true);
    }
  }
  async function finishRecording(){
    const blob=await stopMicrophone();if(blob)await translate(blob,'recording');
  }
  function textResult(id,value){const e=ui[id];e.textContent=value;e.classList.remove('placeholder');}
  function metric(name,value){const e=document.createElement('span');e.className='metric';e.textContent=`${name} ${value} ms`;ui.metrics.append(e);}
  function base64Blob(b64,mime='audio/wav') {
    const raw=atob(b64),length=raw.length,arr=new Uint8Array(length);
    for(let i=0;i<length;i++)arr[i]=raw.charCodeAt(i);
    return new Blob([arr],{type:mime});
  }
  async function translate(blob,kind){
    state.busy=true;update();setStatus('Running local inference. First request may load model weights…');
    const [source,target]=ui.direction.value.split(':');
    const form=new FormData();form.append('source',source);form.append('target',target);form.append('audio',blob,kind==='file'?'upload.wav':'speech.wav');
    const t0=performance.now();
    try {
      const response=await fetch('/api/translate',{method:'POST',body:form,cache:'no-store'});
      const result=await response.json();
      if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:`HTTP ${response.status}`);
      textResult('transcript',result.transcript);textResult('translation',result.translation);
      ui.metrics.replaceChildren();
      metric('ASR',result.timings_ms.asr);metric('MT',result.timings_ms.mt);
      metric('TTS',result.timings_ms.tts);metric('Processing',result.timings_ms.total);
      metric('Browser request',Math.round(performance.now()-t0));
      clearUrl('outputUrl');ui.playback.pause();ui.playback.removeAttribute('src');ui.playback.hidden=true;
      if(result.audio_base64){
        state.outputUrl=URL.createObjectURL(base64Blob(result.audio_base64));
        ui.playback.src=state.outputUrl;ui.playback.hidden=false;
        ui.ttsState.textContent='Translated speech generated locally. Press play to listen.';
        await selectOutput();
        setStatus('Translation and audio generated. Review meaning, then press play.');
      } else {
        ui.ttsState.textContent=result.tts_status==='unavailable'?`Speech unavailable: ${result.tts_error||'voice not installed'}`:
          'Text translation completed; speech synthesis is disabled.';
        setStatus('Translation complete without audio. Check the TTS voice status.');
      }
      // An explicit export contains text/metadata, never the audio base64.
      const {audio_base64,audio_mime,...metadata}=result;
      state.lastEvidence={...metadata,client_request_ms:Math.round(performance.now()-t0),
                          audio_source:kind,recorded_seconds:kind==='recording'?Number(((Date.now()-state.start)/1000).toFixed(2)):null,
                          export_warning:'Contains potentially identifiable transcripts. Keep private.'};
    }catch(error){setStatus(`Translation failed: ${error.message}`,true);}
    finally{state.busy=false;update();}
  }
  function downloadFile(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function refreshOutputs(){
    const list=ui.outputDevice;const existing=list.value;
    list.replaceChildren(new Option('System default','default'));
    if(!navigator.mediaDevices?.enumerateDevices){list.disabled=true;return;}
    try {
      const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audiooutput');
      for(const d of devices){if(d.deviceId && d.deviceId!=='default')list.add(new Option(d.label||'Audio output',d.deviceId));}
      if([...list.options].some(o=>o.value===existing))list.value=existing;
      list.disabled=!(typeof ui.playback.setSinkId==='function');
      if(list.disabled)ui.voiceState.textContent+=' · Browser output selection unavailable: use macOS Sound settings.';
    }catch(error){list.disabled=true;}
  }
  async function selectOutput(){
    if(typeof ui.playback.setSinkId!=='function')return;
    try {await ui.playback.setSinkId(ui.outputDevice.value==='default'?'':ui.outputDevice.value);}
    catch(e){setStatus('Could not select that output. Set the Bluetooth earbuds as macOS system output.',true);}
  }
  async function checkVoices(){
    try {
      const data=await (await fetch('/api/voices',{cache:'no-store'})).json();
      if(data.message){ui.voiceState.textContent=data.message;return;}
      const en=data.preferred?.en||'not installed',zh=data.preferred?.zh||'not installed';
      ui.voiceState.textContent=`Voices · English: ${en} · Mandarin: ${zh}`;
    }catch(e){ui.voiceState.textContent='Could not inspect installed voices.';}
  }
  function clearSession(){
    if(state.active || state.busy)return;
    for(const k of ['inputUrl','outputUrl'])clearUrl(k);
    for(const k of ['playback','sourcePlayback']){ui[k].pause();ui[k].removeAttribute('src');ui[k].hidden=true;}
    state.lastWav=null;state.lastEvidence=null;state.peaks=[];
    ui.waveform.getContext('2d').clearRect(0,0,1000,116);ui.level.textContent='LEVEL —';ui.timer.textContent='00:00';
    ui.transcript.textContent='Your speech will appear here.';ui.transcript.classList.add('placeholder');
    ui.translation.textContent='Translated text will appear here.';ui.translation.classList.add('placeholder');
    ui.ttsState.textContent='No translated speech generated yet.';ui.metrics.textContent='No timing evidence yet.';
    ui.file.value='';setStatus('Session cleared from this browser page.');update();
  }
  document.addEventListener('openhear:live-change', e => { state.liveActive=Boolean(e.detail?.active);update(); });
  ui.direction.addEventListener('change',setDirection);
  ui.consent.addEventListener('change',update);
  ui.file.addEventListener('change',update);
  ui.record.addEventListener('click',startRecording);
  ui.stop.addEventListener('click',finishRecording);
  ui.upload.addEventListener('click',async()=>{if(!ui.consent.checked || !ui.file.files.length)return;
    const blob=ui.file.files[0];clearUrl('inputUrl');state.inputUrl=URL.createObjectURL(blob);
    ui.sourcePlayback.src=state.inputUrl;ui.sourcePlayback.hidden=false;state.lastWav=null;
    await translate(blob,'file');
  });
  ui.outputDevice.addEventListener('change',selectOutput);
  ui.refreshOutputs.addEventListener('click',refreshOutputs);
  ui.clear.addEventListener('click',clearSession);
  ui.saveRecording.addEventListener('click',()=>{if(state.lastWav)downloadFile(state.lastWav,`openhear-${ui.direction.value.replace(':','-')}-${new Date().toISOString().replaceAll(':','-')}.wav`);});
  ui.saveEvidence.addEventListener('click',()=>{if(state.lastEvidence)downloadFile(new Blob([JSON.stringify(state.lastEvidence,null,2)],{type:'application/json'}),`openhear-evidence-${new Date().toISOString().replaceAll(':','-')}.json`);});
  window.addEventListener('beforeunload',()=>{if(state.mic)state.mic.getTracks().forEach(t=>t.stop());for(const k of ['inputUrl','outputUrl'])clearUrl(k);});
  setDirection();waveform();update();void checkVoices();void refreshOutputs();
})();
