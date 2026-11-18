const oldKey = 'franok.sessions.v1',
  key = 'franok.active.v1',
  writer = crypto.randomUUID();
const completed = (tx) =>
  new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
const request = (r) =>
  new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
let db,
  chain = Promise.resolve();
function readOld() {
  try {
    const rows = JSON.parse(localStorage.getItem(oldKey) || '[]');
    return Array.isArray(rows) ? rows : [];
  } catch {
    return [];
  }
}
function queue(operation) {
  chain = chain.catch(() => {}).then(operation);
  return chain;
}
export async function loadSessions() {
  try {
    const open = indexedDB.open('franok-workspace', 2);
    open.onupgradeneeded = () => {
      for (const store of ['sessions', 'drafts'])
        if (!open.result.objectStoreNames.contains(store))
          open.result.createObjectStore(store, { keyPath: 'id' });
    };
    db = await request(open);
    db.onversionchange = () => db.close();
    if (!localStorage.getItem('franok.idb.v1')) {
      const tx = db.transaction('sessions', 'readwrite'),
        done = completed(tx);
      for (const item of readOld())
        if (item?.id && Array.isArray(item.turns))
          tx.objectStore('sessions').put({ ...item, rev: 0, updated: item.updated || 0 });
      await done;
      localStorage.setItem('franok.idb.v1', '1');
    }
    const tx = db.transaction(['sessions', 'drafts']);
    const [rows, drafts] = await Promise.all(
      ['sessions', 'drafts'].map((s) => request(tx.objectStore(s).getAll())),
    );
    const byId = new Map(drafts.map((d) => [d.id, d]));
    for (const row of rows) {
      const draft = byId.get(row.id);
      if (draft && draft.updated >= row.updated) row.draft = draft.text;
    }
    return rows.sort((a, b) => (b.updated || 0) - (a.updated || 0));
  } catch {
    db = null;
    return readOld();
  }
}
export const activeId = () => sessionStorage.getItem(key) || localStorage.getItem(key);
export function setActive(id) {
  sessionStorage.setItem(key, id);
  localStorage.setItem(key, id);
}
export function saveDraft(item) {
  if (!db) return saveSession(item);
  const text = item.draft,
    updated = Date.now();
  return queue(async () => {
    const tx = db.transaction('drafts', 'readwrite'),
      done = completed(tx);
    tx.objectStore('drafts').put({ id: item.id, text, updated });
    await done;
  });
}
export function saveSession(item) {
  const snapshot = structuredClone(item);
  snapshot.turns = snapshot.turns.filter((t) => t.role !== 'assistant' || t.text);
  snapshot.updated = Date.now();
  return queue(async () => {
    snapshot.id = item.id;
    if (!db) {
      const list = readOld().filter((r) => r.id !== item.id);
      localStorage.setItem(oldKey, JSON.stringify([snapshot, ...list]));
      item.updated = snapshot.updated;
      return;
    }
    const tx = db.transaction(['sessions', 'drafts'], 'readwrite'),
      done = completed(tx),
      store = tx.objectStore('sessions');
    const prior = await request(store.get(item.id));
    if (prior && prior.writer !== writer && (prior.rev || 0) !== (item.rev || 0)) {
      snapshot.id = crypto.randomUUID();
      item.id = snapshot.id;
    }
    snapshot.rev = (prior?.rev || 0) + 1;
    snapshot.writer = writer;
    store.put(snapshot);
    tx.objectStore('drafts').delete(snapshot.id);
    await done;
    item.rev = snapshot.rev;
    item.updated = snapshot.updated;
  });
}
export function removeSession(id) {
  return queue(async () => {
    if (!db) {
      localStorage.setItem(oldKey, JSON.stringify(readOld().filter((x) => x.id !== id)));
      return;
    }
    const tx = db.transaction(['sessions', 'drafts'], 'readwrite'),
      done = completed(tx);
    tx.objectStore('sessions').delete(id);
    tx.objectStore('drafts').delete(id);
    await done;
    const legacy = readOld();
    if (legacy.some((row) => row.id === id))
      localStorage.setItem(oldKey, JSON.stringify(legacy.filter((row) => row.id !== id)));
  });
}
