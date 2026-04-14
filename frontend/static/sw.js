/**
 * Service Worker — CHW Stunting Prediction Tool
 * ================================================
 * Provides:
 *   1. Offline-first caching of the app shell
 *   2. Background Sync (Chrome/Android) for queued offline assessments
 *   3. Falls back gracefully on iOS/Firefox via the 'online' event
 *      handled in index.html
 *
 * IndexedDB schema:
 *   Database : chw_offline  (version 1)
 *   Store    : pending_assessments  { keyPath: 'offline_id' }
 */

const CACHE_NAME  = 'chw-stunting-v1';
const APP_SHELL   = ['/'];          // cache just the HTML — it contains all CSS/JS inline
const SYNC_TAG    = 'sync-assessments';
const DB_NAME     = 'chw_offline';
const STORE_NAME  = 'pending_assessments';

// ── INSTALL ──────────────────────────────────────────────────────────────────
// Pre-cache the app shell so it loads instantly on next visit, even offline.
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())   // activate immediately, don't wait for old SW
  );
});

// ── ACTIVATE ─────────────────────────────────────────────────────────────────
// Delete stale caches from previous versions.
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())  // take control of all open tabs immediately
  );
});

// ── FETCH ─────────────────────────────────────────────────────────────────────
// Strategy:
//   / (app shell)   → cache-first  (load instantly, even offline)
//   /predict, /sync → network-only (the page handles offline with IndexedDB)
//   everything else → network, fall back to cache
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Only intercept same-origin requests
  if (url.origin !== self.location.origin) return;

  if (url.pathname === '/') {
    // App shell: serve from cache, fall back to network
    event.respondWith(
      caches.match(request)
        .then(cached => cached || fetch(request).then(response => {
          // Update the cache with the fresh response
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(request, copy));
          return response;
        }))
    );
    return;
  }

  // API endpoints: network-only (never cache predictions)
  if (['/predict', '/sync', '/health'].some(p => url.pathname.startsWith(p))) {
    event.respondWith(fetch(request));
    return;
  }
});

// ── BACKGROUND SYNC ───────────────────────────────────────────────────────────
// Fired by Chrome/Android when connectivity is restored after the page
// registered a sync tag ('sync-assessments').
self.addEventListener('sync', event => {
  if (event.tag === SYNC_TAG) {
    event.waitUntil(flushPendingAssessments());
  }
});

// ── INDEXEDDB HELPERS ─────────────────────────────────────────────────────────
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);

    req.onupgradeneeded = e => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'offline_id' });
      }
    };

    req.onsuccess = e  => resolve(e.target.result);
    req.onerror   = e  => reject(e.target.error);
  });
}

function getAllPending(db) {
  return new Promise((resolve, reject) => {
    const tx    = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req   = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function deletePending(db, ids) {
  return new Promise((resolve, reject) => {
    const tx    = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    ids.forEach(id => store.delete(id));
    tx.oncomplete = resolve;
    tx.onerror    = () => reject(tx.error);
  });
}

// ── FLUSH QUEUE ───────────────────────────────────────────────────────────────
// Called by both Background Sync and the 'online' fallback message from the page.
async function flushPendingAssessments() {
  const db      = await openDB();
  const pending = await getAllPending(db);

  if (pending.length === 0) return;

  let response;
  try {
    response = await fetch('/sync', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ assessments: pending }),
    });
  } catch (networkError) {
    // Still offline — throw so Background Sync retries later
    throw networkError;
  }

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`Sync endpoint returned ${response.status}: ${detail}`);
  }

  const data = await response.json();

  // Remove successfully synced items from IndexedDB
  const syncedIds = pending.map(p => p.offline_id);
  await deletePending(db, syncedIds);

  // Notify all open tabs so the UI can update the pending badge
  const clients = await self.clients.matchAll({ includeUncontrolled: true });
  clients.forEach(client =>
    client.postMessage({
      type:    'SYNC_COMPLETE',
      count:   pending.length,
      results: data.results || [],
    })
  );
}

// ── MESSAGE HANDLER ───────────────────────────────────────────────────────────
// The page sends { type: 'FLUSH_NOW' } when the 'online' event fires
// (fallback for iOS / Firefox where Background Sync is not supported).
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'FLUSH_NOW') {
    event.waitUntil(flushPendingAssessments());
  }
});
