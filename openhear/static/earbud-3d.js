/**
 * OpenHear 3D Procedural Hearing Earbud Renderer (WebGL SDF Raymarcher)
 * Renders a futuristic Apple-style hearing earbud with full 360° views,
 * glossy ceramic materials, chrome accents, acoustic sound mesh, and voice reactivity.
 */
(() => {
  'use strict';

  let voiceEnergy = 0.0;
  let voicePeak = 0.0;

  window.setEarbudVoiceEnergy = (energy, peak) => {
    voiceEnergy = Math.max(0.0, Math.min(1.0, energy));
    voicePeak = Math.max(0.0, Math.min(1.0, peak));
  };

  function initEarbud3D() {
    const canvas = document.getElementById('earbud3dCanvas');
    if (!canvas) return;

    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) {
      console.warn('WebGL unavailable for 3D earbud');
      return;
    }

    const vsSource = `
      attribute vec2 a_pos;
      varying vec2 v_uv;
      void main() {
        v_uv = a_pos * 0.5 + 0.5;
        gl_Position = vec4(a_pos, 0.0, 1.0);
      }
    `;

    const fsSource = `
      precision highp float;
      varying vec2 v_uv;
      uniform vec2 u_resolution;
      uniform float u_time;
      uniform vec2 u_rot;
      uniform float u_voice;
      uniform float u_peak;

      // 3D Rotation matrices
      mat3 rotX(float a) {
        float c = cos(a), s = sin(a);
        return mat3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
      }
      mat3 rotY(float a) {
        float c = cos(a), s = sin(a);
        return mat3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
      }
      mat3 rotZ(float a) {
        float c = cos(a), s = sin(a);
        return mat3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
      }

      // Smooth minimum for unibody Apple ergonomic curvature
      float smin(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) - k * h * (1.0 - h);
      }

      float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
        vec3 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h) - r;
      }

      float sdSphere(vec3 p, float r) {
        return length(p) - r;
      }

      // Material ID return: vec2(dist, mat_id)
      // mat_id: 1.0 = Glossy Ceramic White, 2.0 = Chrome Metal Base, 3.0 = Black Acoustic Grill, 4.0 = Silicone Tip
      vec2 map(vec3 p) {
        // Base coordinate center
        vec3 q = p;

        // 1. Acoustic Head Bulb (Ellipsoidal Apple shape)
        vec3 pHead = q - vec3(0.0, 0.22, 0.0);
        float dHead = length(pHead / vec3(1.0, 0.94, 1.08)) * 0.96 - 0.44;

        // 2. Ergonomic Contoured Stem
        vec3 stemTop = vec3(0.08, 0.16, 0.08);
        vec3 stemBottom = vec3(0.18, -0.66, 0.22);
        float dStem = sdCapsule(q, stemTop, stemBottom, 0.13);

        // Seamless unibody junction between head and stem
        float dUnibody = smin(dHead, dStem, 0.15);

        // 3. In-Ear Sound Nozzle & Soft Silicone Tip
        vec3 pTip = q - vec3(-0.35, 0.26, -0.14);
        pTip = rotY(0.42) * rotZ(-0.32) * pTip;
        float dTip = sdCapsule(pTip, vec3(0.0, 0.0, -0.1), vec3(0.0, 0.0, 0.08), 0.23);
        float dBodyWithTip = smin(dUnibody, dTip, 0.11);

        // 4. Force-Sensor Capacitive Indent on Stem
        vec3 pSensor = q - vec3(0.14, -0.32, 0.23);
        float dSensor = sdCapsule(pSensor, vec3(0.0, -0.12, 0.0), vec3(0.0, 0.12, 0.0), 0.05);

        // Main Ceramic Body with sensor indentation
        float dCeramic = max(dBodyWithTip, -dSensor);

        // 5. Chrome Charging Contacts at Base of Stem
        float dChromeBase = sdCapsule(q, stemBottom - vec3(0.01, -0.06, 0.01), stemBottom + vec3(0.01, 0.03, 0.01), 0.138);

        // 6. Black Acoustic Speaker Grill (Front Sound Exit)
        float dFrontGrill = sdSphere(pTip - vec3(0.0, 0.0, -0.13), 0.155);

        // 7. Outer Noise-Canceling Microphone Mesh Vent
        vec3 pVent = q - vec3(0.31, 0.44, 0.0);
        float dVent = length(pVent) - 0.085;

        // Combine into materials
        float d = dCeramic;
        float mat = 1.0; // White glossy ceramic

        if (dChromeBase < d) {
          d = dChromeBase;
          mat = 2.0; // Chrome
        }
        if (dFrontGrill < d + 0.01 && length(pTip - vec3(0.0, 0.0, -0.12)) < 0.22) {
          d = dFrontGrill;
          mat = 3.0; // Acoustic Grill
        }
        if (dVent < d + 0.01) {
          d = dVent;
          mat = 3.0; // Acoustic Vent
        }

        return vec2(d, mat);
      }

      vec3 calcNormal(vec3 p) {
        float h = 0.001;
        vec2 k = vec2(1.0, -1.0);
        return normalize(k.xyy * map(p + k.xyy * h).x +
                         k.yyx * map(p + k.yyx * h).x +
                         k.yxy * map(p + k.yxy * h).x +
                         k.xxx * map(p + k.xxx * h).x);
      }

      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);

        // Camera ray setup
        vec3 ro = vec3(0.0, 0.0, 3.2);
        vec3 rd = normalize(vec3(uv, -1.8));

        // 360-degree rotation: combination of smooth turntable spin and mouse rotation
        mat3 rot = rotX(u_rot.y) * rotY(u_rot.x) * rotZ(sin(u_time * 0.5) * 0.08);

        ro = rot * ro;
        rd = rot * rd;

        // Raymarching loop
        float t = 0.0;
        float mat = 0.0;
        for (int i = 0; i < 75; i++) {
          vec3 p = ro + rd * t;
          vec2 res = map(p);
          if (res.x < 0.001) {
            mat = res.y;
            break;
          }
          if (t > 7.0) break;
          t += res.x;
        }

        // Ambient Acoustic Shockwave Energy Rings (Voice-Reactive)
        float waveDist = length(uv);
        float pulseRate = u_time * 2.8;
        float wave1 = sin(waveDist * 22.0 - pulseRate);
        float wave2 = sin(waveDist * 16.0 - pulseRate * 0.7);
        float ringIntensity = smoothstep(0.85, 1.0, wave1) * 0.18 + smoothstep(0.9, 1.0, wave2) * 0.22;
        ringIntensity *= smoothstep(0.7, 0.1, waveDist);
        ringIntensity *= (0.35 + u_voice * 1.8 + u_peak * 0.9);

        vec3 ringCol = mix(vec3(0.0, 0.85, 1.0), vec3(0.2, 0.6, 1.0), waveDist);

        if (t > 7.0 || mat < 0.5) {
          // Transparent background with subtle glowing voice ring ripple
          gl_FragColor = vec4(ringCol, ringIntensity);
          return;
        }

        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 v = -rd;

        // Studio Lighting: Key light, fill light, and visionOS cyan rim light
        vec3 keyLight = normalize(vec3(1.2, 2.0, 1.8));
        vec3 fillLight = normalize(vec3(-1.8, -0.8, 1.0));
        vec3 rimLight = normalize(vec3(0.0, 1.5, -2.0));

        float diffKey = max(0.0, dot(n, keyLight));
        float diffFill = max(0.0, dot(n, fillLight)) * 0.35;
        float diffRim = max(0.0, dot(n, rimLight)) * 0.45;

        // Specular reflections
        vec3 hKey = normalize(keyLight + v);
        float specKey = pow(max(0.0, dot(n, hKey)), 36.0);

        // Fresnel edge glow
        float fresnel = pow(1.0 - max(0.0, dot(n, v)), 3.0);

        vec3 albedo = vec3(0.96, 0.97, 1.0); // Apple Ceramic Pearl
        float roughness = 0.15;
        vec3 specColor = vec3(1.0);

        if (mat > 1.5 && mat < 2.5) {
          // Chrome Charging Contacts
          albedo = vec3(0.82, 0.84, 0.88);
          specKey = pow(max(0.0, dot(n, hKey)), 72.0) * 1.8;
          fresnel *= 1.4;
        } else if (mat > 2.5) {
          // Dark Acoustic Mesh Grill / Microphone Vent
          albedo = vec3(0.12, 0.14, 0.18);
          specKey *= 0.3;
          roughness = 0.8;
        }

        // Composite lighting
        vec3 col = albedo * (0.28 + diffKey * 0.72 + diffFill + diffRim);
        col += specColor * specKey * 0.75;

        // Signature visionOS / CleanMyMac Cyan Fresnel Rim & Voice Luminescence
        vec3 cyanGlow = vec3(0.0, 0.92, 1.0);
        col += cyanGlow * (fresnel * 0.55 + u_voice * 0.45);

        // Inner audio glow when speaking
        if (mat > 2.5 && u_voice > 0.05) {
          col += cyanGlow * u_voice * 1.5;
        }

        // Add acoustic wave rings over top
        col += ringCol * ringIntensity * 0.8;

        gl_FragColor = vec4(col, 1.0);
      }
    `;

    function createShader(gl, type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    }

    const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return;
    }

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]), gl.STATIC_DRAW);

    const aPosLoc = gl.getAttribLocation(program, 'a_pos');
    const uResLoc = gl.getUniformLocation(program, 'u_resolution');
    const uTimeLoc = gl.getUniformLocation(program, 'u_time');
    const uRotLoc = gl.getUniformLocation(program, 'u_rot');
    const uVoiceLoc = gl.getUniformLocation(program, 'u_voice');
    const uPeakLoc = gl.getUniformLocation(program, 'u_peak');

    // 360-degree rotation angles
    let angleY = 0.4;
    let angleX = 0.25;
    let isDragging = false;
    let startX = 0, startY = 0;
    let baseAngleX = 0.25, baseAngleY = 0.4;

    canvas.addEventListener('mousedown', e => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      baseAngleX = angleX;
      baseAngleY = angleY;
    });

    window.addEventListener('mousemove', e => {
      if (!isDragging) return;
      const dx = (e.clientX - startX) * 0.015;
      const dy = (e.clientY - startY) * 0.015;
      angleY = baseAngleY + dx;
      angleX = Math.max(-1.2, Math.min(1.2, baseAngleX + dy));
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // Touch support for mobile 360° rotation
    canvas.addEventListener('touchstart', e => {
      if (e.touches.length === 1) {
        isDragging = true;
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        baseAngleX = angleX;
        baseAngleY = angleY;
      }
    }, { passive: true });

    window.addEventListener('touchmove', e => {
      if (!isDragging || e.touches.length !== 1) return;
      const dx = (e.touches[0].clientX - startX) * 0.015;
      const dy = (e.touches[0].clientY - startY) * 0.015;
      angleY = baseAngleY + dx;
      angleX = Math.max(-1.2, Math.min(1.2, baseAngleX + dy));
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isDragging = false;
    });

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    let startTime = performance.now();

    function render() {
      const now = performance.now();
      const elapsed = (now - startTime) * 0.001;

      // When not dragging, rotate gracefully 360° continuously like a turntable
      if (!isDragging) {
        angleY += 0.014; // Smooth continuous 360-degree rotation
      }

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);
      gl.enableVertexAttribArray(aPosLoc);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
      gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(uResLoc, canvas.width, canvas.height);
      gl.uniform1f(uTimeLoc, elapsed);
      gl.uniform2f(uRotLoc, angleY, angleX);
      gl.uniform1f(uVoiceLoc, voiceEnergy);
      gl.uniform1f(uPeakLoc, voicePeak);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      requestAnimationFrame(render);
    }

    requestAnimationFrame(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEarbud3D);
  } else {
    initEarbud3D();
  }
})();
