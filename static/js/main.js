// ── Clock ─────────────────────────────────────────────────────────────────────
setInterval(() => {
    const clock = document.getElementById('clock');
    if (clock) clock.innerText = new Date().toLocaleTimeString();
}, 1000);

// ── DOM refs ──────────────────────────────────────────────────────────────────
const fireCard = document.getElementById('fire-card');
const fireStatusText = document.getElementById('fire-status-text');
const fireConf = document.getElementById('fire-conf');
const attStatusText = document.getElementById('att-status-text');
const recentFacesList = document.getElementById('recent-faces-list');
const logList = document.getElementById('system-log');
const alarmSound = document.getElementById('alarm-sound');
const muteBtn = document.getElementById('mute-btn');
const alertModal = document.getElementById('fire-alert');
const alertTitle = document.getElementById('alert-title');
const alertDesc = document.getElementById('alert-desc');

const nearbyObjects = document.getElementById('nearby-objects');
const objectStatusText = document.getElementById('object-status-text');
const objectCard = document.getElementById('object-card');

const camModal = document.getElementById('camera-modal');
const camList = document.getElementById('camera-list');
const camMgmtBtn = document.getElementById('cam-mgmt-btn');

let isMuted = false;
let isAlertActive = false;

// Guard against missing audio
if (alarmSound) {
    alarmSound.addEventListener('error', () => {
        console.warn("Alarm sound could not be loaded. Alerts will be visual only.");
    });
}

// ── Log helper ────────────────────────────────────────────────────────────────
function addLog(message, isAlert = false) {
    if (!logList) return;
    const time = new Date().toLocaleTimeString();
    const li = document.createElement('li');
    li.innerHTML = `<span class="time">${time}</span> <span style="${isAlert ? 'color:#ef4444;font-weight:bold' : ''}">${message}</span>`;
    logList.insertBefore(li, logList.firstChild);
    if (logList.children.length > 60) logList.lastChild.remove();
}

// ── Alarm controls ────────────────────────────────────────────────────────────
if (muteBtn) {
    muteBtn.addEventListener('click', () => {
        isMuted = !isMuted;
        muteBtn.innerHTML = isMuted ? '<i class="fas fa-volume-up"></i>' : '<i class="fas fa-volume-mute"></i>';
        if (isMuted && alarmSound) alarmSound.pause();
        else if (isAlertActive && alarmSound) {
            try { alarmSound.play(); } catch(e) {}
        }
    });
}

window.dismissAlert = function () {
    if (alertModal) alertModal.classList.add('hidden');
    if (alarmSound) {
        alarmSound.pause();
        alarmSound.currentTime = 0;
    }
    isAlertActive = false;
};

// ── Camera Management ────────────────────────────────────────────────────────
if (camMgmtBtn) {
    camMgmtBtn.addEventListener('click', () => {
        if (camModal) camModal.classList.remove('hidden');
        refreshCameraList();
    });
}

window.closeCameraModal = () => {
    if (camModal) camModal.classList.add('hidden');
};

async function refreshCameraList() {
    try {
        const resp = await fetch('/api/cameras');
        const data = await resp.json();
        const activeId = data.active_id;
        
        camList.innerHTML = '';
        data.cameras.forEach(cam => {
            const isActive = cam.id === activeId;
            const li = document.createElement('li');
            li.style.cssText = `display:flex; justify-content:space-between; align-items:center; padding:8px; margin-bottom:5px; border-radius:6px; background:${isActive ? 'rgba(255,65,61,0.15)' : 'rgba(255,255,255,0.03)'}; border: 1px solid ${isActive ? 'var(--primary-red)' : 'rgba(255,255,255,0.05)'}`;
            
            li.innerHTML = `
                <div style="flex:1; cursor:pointer;" onclick="switchCamera(${cam.id})">
                    <div style="font-weight:bold; font-size:0.85rem; color:${isActive ? 'var(--primary-red)' : '#fff'}">${cam.name}</div>
                    <div style="font-size:0.7rem; color:var(--text-muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px;">${cam.source === 0 ? 'Local Webcam' : cam.source}</div>
                </div>
                ${cam.id !== 0 ? `<button onclick="deleteCamera(${cam.id})" style="background:transparent; border:none; color:#ef4444; cursor:pointer; padding:5px;"><i class="fas fa-trash"></i></button>` : ''}
            `;
            camList.appendChild(li);
        });
    } catch (e) { console.error(e); }
}

window.addCamera = async () => {
    const name = document.getElementById('cam-name').value.trim();
    const source = document.getElementById('cam-source').value.trim();
    if (!name || !source) return alert("Please enter name and source.");

    try {
        const resp = await fetch('/api/cameras', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, source })
        });
        if (resp.ok) {
            document.getElementById('cam-name').value = '';
            document.getElementById('cam-source').value = '';
            refreshCameraList();
            addLog(`📹 Added camera: ${name}`);
        }
    } catch (e) { console.error(e); }
};

window.switchCamera = async (id) => {
    try {
        const resp = await fetch('/api/cameras/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        if (resp.ok) {
            refreshCameraList();
            addLog(`🔄 Switched to camera source.`);
            // Reload video feed to catch the new source
            const videoFeed = document.getElementById('video-feed');
            const currentSrc = videoFeed.src.split('?')[0];
            videoFeed.src = currentSrc + '?t=' + new Date().getTime();
        }
    } catch (e) { console.error(e); }
};

window.deleteCamera = async (id) => {
    if (!confirm("Remove this camera source?")) return;
    try {
        const resp = await fetch('/api/cameras/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        if (resp.ok) refreshCameraList();
    } catch (e) { console.error(e); }
};

// ── Status polling (both systems at once) ─────────────────────────────────────
async function pollStatus() {
    try {
        const resp = await fetch('/status');
        const data = await resp.json();

        // ── Fire ──
        const fire = data.fire;
        if (fireCard && fireStatusText && fireConf) {
            if (fire.detected) {
                fireStatusText.innerText = 'FIRE DETECTED!';
                fireStatusText.className = 'status-main danger-color';
                fireCard.className = 'status-card glass danger';
                fireConf.innerText = fire.confidence + '%';

                if (!isAlertActive && alertModal && alertTitle && alertDesc) {
                    isAlertActive = true;
                    alertTitle.innerText = 'FIRE DETECTED!';
                    alertDesc.innerText = 'Immediate action required.';
                    alertModal.classList.remove('hidden');
                    if (!isMuted && alarmSound) {
                        try { alarmSound.play(); } catch(e) {}
                    }
                    addLog(`🔥 FIRE DETECTED! Conf: ${fire.confidence}%`, true);
                }
            } else {
                fireStatusText.innerText = 'ENVIRONMENT SAFE';
                fireStatusText.className = 'status-main safe-color';
                fireCard.className = 'status-card glass safe';
                fireConf.innerText = '0%';
                if (isAlertActive && alertTitle && alertTitle.innerText.includes('FIRE')) dismissAlert();
            }
        }

        // ── Object / Weapon ──
        const obj = data.object;
        const hasLighter = obj && obj.nearby.includes('LIGHTER');

        if (objectCard && objectStatusText) {
            if (obj && obj.weapon) {
                objectStatusText.innerText = 'WEAPON DETECTED!';
                objectStatusText.className = 'status-main danger-color';
                objectCard.className = 'status-card glass danger';

                if (!isAlertActive && alertModal && alertTitle && alertDesc) {
                    isAlertActive = true;
                    alertTitle.innerText = 'WEAPON DETECTED!';
                    alertDesc.innerText = 'Threat detected on camera.';
                    alertModal.classList.remove('hidden');
                    if (!isMuted && alarmSound) {
                        try { alarmSound.play(); } catch(e) {}
                    }
                    addLog(`⚔️ THREAT: ${obj.weapon_labels.join(', ')}`, true);
                }
            } else if (hasLighter) {
                objectStatusText.innerText = 'LIGHTER DETECTED';
                objectStatusText.className = 'status-main';
                objectStatusText.style.color = '#ff9f43';
                objectCard.className = 'status-card glass';
                objectCard.style.borderColor = '#ff9f43';
                if (isAlertActive && alertTitle && alertTitle.innerText.includes('WEAPON')) dismissAlert();
            } else if (obj) {
                objectStatusText.innerText = 'AREA CLEAR';
                objectStatusText.className = 'status-main safe-color';
                objectStatusText.style.color = ''; 
                objectCard.className = 'status-card glass safe';
                objectCard.style.borderColor = '';
                if (isAlertActive && alertTitle && alertTitle.innerText.includes('WEAPON')) dismissAlert();
            }
        }

        if (nearbyObjects) {
            nearbyObjects.innerText = (obj && obj.nearby.length > 0) ? obj.nearby.join(', ') : 'None';
        }

        // ── Face attendance ──
        const att = data.attendance;
        const tableBody = document.getElementById('att-table-body');

        if (attStatusText) {
            if (att.recent_faces && att.recent_faces.length > 0) {
                const known = att.recent_faces.filter(f => f !== 'Unknown');
                attStatusText.innerText = known.length > 0 ? 'FACE RECOGNIZED' : 'FACE DETECTED';
            } else {
                attStatusText.innerText = 'SCANNING...';
            }
        }

        if (tableBody) {
            if (att.table && att.table.length > 0) {
                tableBody.innerHTML = '';
                att.table.forEach(row => {
                    const isRecent = att.recent_faces.some(f => f.toLowerCase() === row.Name.toLowerCase());
                    const tr = document.createElement('tr');

                    let checkoutDisplay = row['Check-Out'];
                    let checkoutColor = '#10b981';

                    if (isRecent) {
                        checkoutDisplay = '<span style="background:var(--accent-safe); color:#fff; padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; animation: pulse-safe 1s infinite alternate;">LIVE</span>';
                    } else if (row['Check-Out'] === '—' || !row['Check-Out']) {
                        checkoutDisplay = '—';
                        checkoutColor = '#64748b';
                    }

                    tr.innerHTML = `
                        <td><strong>${row.Name}</strong></td>
                        <td style="color:#10b981">${row['Check-In']}</td>
                        <td style="color:${checkoutColor}">${checkoutDisplay}</td>`;
                    tableBody.appendChild(tr);
                });
            } else {
                tableBody.innerHTML = '<tr><td colspan="3" style="color:#64748b;text-align:center;padding:15px;">No records yet…</td></tr>';
            }
        }

    } catch (e) {
        console.error("Polling error:", e);
    }
}

setInterval(pollStatus, 1000);

// ── Face Registration ─────────────────────────────────────────────────────────
window.registerFace = async function () {
    const nameInput = document.getElementById('new-face-name'); // Syncing with potential dashboard elements
    if (!nameInput) return; // Not present on dashboard currently

    const msgDiv = document.getElementById('register-msg');
    const name = nameInput.value.trim();

    if (!name) { msgDiv.className = 'msg error'; msgDiv.innerText = 'Enter a name first.'; return; }

    try {
        for (let i = 1; i <= 3; i++) {
            msgDiv.className = 'msg';
            msgDiv.innerText = `📸 Capturing Photo ${i}/3... Stay still!`;

            const resp = await fetch('/register_face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, photo_num: i })
            });
            const data = await resp.json();

            if (!resp.ok || data.status !== 'success') {
                throw new Error(data.message || `Photo ${i} failed.`);
            }
            if (i < 3) await new Promise(r => setTimeout(r, 800));
        }

        msgDiv.className = 'msg success';
        msgDiv.innerText = `✅ Registered successfully with 3 photos!`;
        nameInput.value = '';
        addLog(`📸 Registered: ${name}`);
    } catch (err) {
        msgDiv.className = 'msg error';
        msgDiv.innerText = err.message || 'Error';
    }

    setTimeout(() => { if (msgDiv) msgDiv.innerText = ''; }, 5000);
};
