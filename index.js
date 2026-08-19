import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import crypto from 'crypto';
import { GoogleGenAI, Type } from '@google/genai';
import admin from 'firebase-admin';
import rateLimit from 'express-rate-limit';

dotenv.config();

const app = express();
// Render (et la plupart des PaaS) place le serveur derrière un proxy inverse :
// sans ce réglage, req.ip renvoie l'IP du proxy pour toutes les requêtes, pas
// celle du client, ce qui casserait tout rate limiting basé sur l'IP.
app.set('trust proxy', 1);
// Limite relevée par rapport au défaut (100kb) : les photos de repas compressées
// en base64 (voir prepareImageForAI côté frontend) dépassent largement 100kb.
app.use(express.json({ limit: '10mb' }));
app.use(cors({ origin: process.env.CLIENT_URL || '*' }));

// Sans ce middleware, une erreur de body-parser (JSON malformé, ou payload qui
// dépasse la limite de 10mb ci-dessus) était renvoyée par Express sous forme de
// page HTML par défaut, pas de JSON — le frontend (window.api.post) qui appelle
// `response.json()` échouait alors silencieusement et retombait sur un message
// générique peu clair. On intercepte ici pour toujours répondre en JSON.
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.too.large') {
    return res.status(413).json({ error: "Cette image est trop volumineuse pour être envoyée. Réessaie avec une photo plus légère." });
  }
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: "Requête invalide." });
  }
  next(err);
});

// Timeout applicatif générique pour tout appel externe (Gemini) : évite qu'une
// requête reste bloquée indéfiniment si le modèle ou le réseau ne répond pas,
// et laisse un message clair plutôt qu'un crash ou un hang côté client.
function withTimeout(promise, ms, timeoutMessage) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const err = new Error(timeoutMessage);
      err.isTimeout = true;
      reject(err);
    }, ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Gemini renvoie parfois une erreur transitoire 503 "UNAVAILABLE" (pic de charge
// sur le modèle) — ce n'est pas une panne de notre serveur, juste un pic
// temporaire côté Google. On retente automatiquement quelques fois avec un
// court délai avant d'abandonner, pour que l'utilisateur n'ait pas à relancer
// l'action manuellement pour un souci qui se résout tout seul en quelques secondes.
async function generateContentWithRetry(params, { retries = 2, delayMs = 1500 } = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await ai.models.generateContent(params);
    } catch (err) {
      const status = err?.error?.status || err?.status;
      const isOverloaded = status === 'UNAVAILABLE' || err?.error?.code === 503;
      if (!isOverloaded || attempt >= retries) throw err;
      console.warn(`Gemini surchargé (503), nouvelle tentative ${attempt + 1}/${retries}...`);
      await new Promise(r => setTimeout(r, delayMs * (attempt + 1)));
    }
  }
}

// ============================================================================
// Firebase Admin — nécessaire pour vérifier l'identité de l'utilisateur côté
// serveur (routes Garmin ci-dessous) et pour lire/écrire dans Firestore avec
// des droits élevés, HORS des règles de sécurité client (indispensable pour
// stocker les tokens Garmin dans un endroit que le frontend ne peut jamais lire).
// Le reste de l'app (Goals, Todos, Finance, etc.) continue de passer par le
// SDK client Firebase (dbGet/dbSet) — on ne touche à rien de ça ici.
//
// ⚠️ MANQUANT : ce backend n'a pas encore les credentials d'un compte de
// service Firebase. À créer dans Firebase Console → Paramètres du projet →
// Comptes de service → "Générer une nouvelle clé privée", puis coller le JSON
// obtenu (en une ligne) dans la variable d'environnement Render
// FIREBASE_SERVICE_ACCOUNT_JSON. Tant que cette variable n'est pas définie,
// toutes les routes Garmin ci-dessous répondent en mode développement (501)
// plutôt que de planter ou de simuler des données.
let firebaseAdminReady = false;
try {
  if (process.env.FIREBASE_SERVICE_ACCOUNT_JSON) {
    const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    firebaseAdminReady = true;
  } else {
    console.warn('FIREBASE_SERVICE_ACCOUNT_JSON manquant : routes /api/garmin/* en mode développement (non fonctionnelles).');
  }
} catch (e) {
  console.error("Échec d'initialisation de Firebase Admin", e);
}
const fsDb = () => admin.firestore();

// ============================================================================
// Rate limiting des routes IA — chaque appel consomme du quota Gemini payant ;
// sans limite, un compte compromis (ou un abus involontaire côté frontend, ex.
// double-clic non bloqué) pourrait générer une facture importante en quelques
// minutes. Comptabilisé PAR UTILISATEUR (req.uid, posé par requireFirebaseAuth
// qui s'exécute avant ce middleware sur toutes les routes IA) plutôt que par IP :
// plus juste sur un réseau partagé, et cohérent puisque ces routes exigent déjà
// une connexion. Fallback sur l'IP uniquement si req.uid est absent (ne devrait
// pas arriver derrière requireFirebaseAuth, gardé par sécurité).
//
// EXEMPTIONS : certains utilisateurs (toi, un testeur de confiance...) peuvent
// être dispensés de cette limite via la variable d'environnement Render
// AI_RATE_LIMIT_EXEMPT_UIDS — une liste d'UID Firebase séparés par des virgules
// (ex. "abc123,def456"). Récupère ton UID dans Firebase Console → Authentication,
// ou via `console.log(window.currentUser.uid)` dans la console du navigateur une
// fois connecté à l'app. Aucun redéploiement de code n'est nécessaire pour
// ajouter/retirer un utilisateur : juste modifier la variable sur Render.
const AI_RATE_LIMIT_EXEMPT_UIDS = new Set(
  (process.env.AI_RATE_LIMIT_EXEMPT_UIDS || '')
    .split(',')
    .map(uid => uid.trim())
    .filter(Boolean)
);

const aiRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 30, // 30 appels IA / 15 min / utilisateur — large pour un usage normal, bas pour un abus
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => req.uid || req.ip,
  skip: (req) => Boolean(req.uid && AI_RATE_LIMIT_EXEMPT_UIDS.has(req.uid)),
  handler: (req, res) => {
    res.status(429).json({ error: "Trop de requêtes IA en peu de temps. Réessaie dans quelques minutes." });
  }
});

// Vérifie le token d'authentification Firebase envoyé par le frontend
// (Authorization: Bearer <idToken>, voir window.currentUser.getIdToken() côté
// client). Sans ça, n'importe qui pourrait appeler ces routes en se faisant
// passer pour un autre utilisateur.
async function requireFirebaseAuth(req, res, next) {
  if (!firebaseAdminReady) {
    return res.status(501).json({ error: "Intégration Garmin non configurée sur ce serveur (Firebase Admin manquant).", devMode: true });
  }
  const authHeader = req.headers.authorization || '';
  const match = authHeader.match(/^Bearer (.+)$/);
  if (!match) return res.status(401).json({ error: 'Authentification requise.' });
  try {
    const decoded = await admin.auth().verifyIdToken(match[1]);
    req.uid = decoded.uid;
    next();
  } catch (e) {
    res.status(401).json({ error: 'Session expirée, reconnecte-toi et réessaie.' });
  }
}

// ============================================================================
// Garmin Connect — configuration OAuth 2.0 + PKCE.
// ============================================================================
// ⚠️ MANQUANT : ce backend n'a pas encore d'accès au Garmin Connect Developer
// Program. Pour rendre cette intégration réellement fonctionnelle, il faut :
//   1. Une demande d'accès approuvée auprès de Garmin (programme actuellement
//      en pause pour les nouvelles inscriptions au moment de l'écriture de ce
//      code — vérifie l'état actuel sur developer.garmin.com ; le programme
//      est en outre réservé à un usage "business", pas personnel).
//   2. Une fois approuvé : un Client ID et un Client Secret (Developer
//      Portal), à placer dans les variables d'environnement Render
//      GARMIN_CLIENT_ID et GARMIN_CLIENT_SECRET (jamais dans le frontend).
//   3. Une Redirect URI enregistrée côté Garmin, pointant vers
//      GARMIN_REDIRECT_URI ci-dessous (ex. https://routinepov-backend.onrender.com/api/garmin/callback).
//   4. Une URL de webhook publique (celle de ce backend, /api/garmin/webhook)
//      à déclarer dans le Developer Portal : Garmin y POSTe les données après
//      chaque synchronisation de l'appareil (fonctionnement en "push", pas en
//      requête à la demande — voir la route /api/garmin/sync plus bas).
// Tant que ces 3 variables ne sont pas toutes définies, l'intégration reste
// en mode développement : aucune fausse connexion n'est simulée.
const GARMIN_CLIENT_ID = process.env.GARMIN_CLIENT_ID || '';
const GARMIN_CLIENT_SECRET = process.env.GARMIN_CLIENT_SECRET || '';
const GARMIN_REDIRECT_URI = process.env.GARMIN_REDIRECT_URI || '';
const GARMIN_CONFIGURED = Boolean(GARMIN_CLIENT_ID && GARMIN_CLIENT_SECRET && GARMIN_REDIRECT_URI);
// URL de l'app frontend vers laquelle rediriger l'utilisateur une fois le
// callback OAuth traité (retour sur l'onglet Réglages).
const FRONTEND_URL = process.env.CLIENT_URL_FOR_REDIRECT || 'https://myroutinepov.web.app';

// Endpoints officiels du Developer Portal Garmin (OAuth2 + PKCE). À reconfirmer
// dans la documentation une fois l'accès au Developer Portal obtenu : Garmin a
// fait évoluer son authentification (migration OAuth1 → OAuth2 en cours) et ces
// URLs peuvent différer selon la version exacte de programme accordée.
const GARMIN_AUTH_URL = 'https://connect.garmin.com/oauth2Confirm';
const GARMIN_TOKEN_URL = 'https://diauth.garmin.com/di-oauth2-service/oauth/token';
const GARMIN_API_BASE = 'https://apis.garmin.com';

function base64url(buffer) {
  return buffer.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// Génère une entrée temporaire liant un "state" OAuth à l'utilisateur Firebase
// qui a initié la connexion, avec le code_verifier PKCE — nécessaire pour
// retrouver l'utilisateur au retour du callback Garmin (qui ne transporte pas
// le token Firebase, seulement notre "state"). Expire après 10 minutes.
async function createGarminOAuthState(uid) {
  const state = base64url(crypto.randomBytes(24));
  const codeVerifier = base64url(crypto.randomBytes(48));
  const codeChallenge = base64url(crypto.createHash('sha256').update(codeVerifier).digest());
  await fsDb().collection('garminOAuthState').doc(state).set({
    uid, codeVerifier, createdAt: admin.firestore.FieldValue.serverTimestamp(),
    expiresAt: Date.now() + 10 * 60 * 1000
  });
  return { state, codeChallenge };
}

// ============================================================================
// GET /api/garmin/status — état de connexion Garmin de l'utilisateur courant.
// Ne renvoie JAMAIS de token, uniquement des métadonnées d'affichage.
// ============================================================================
app.get('/api/garmin/status', requireFirebaseAuth, async (req, res) => {
  try {
    if (!GARMIN_CONFIGURED) {
      return res.json({ connected: false, devMode: true, lastSyncAt: null });
    }
    const snap = await fsDb().collection('garminConnections').doc(req.uid).get();
    if (!snap.exists) return res.json({ connected: false, devMode: false, lastSyncAt: null });
    const data = snap.data();
    res.json({ connected: true, devMode: false, lastSyncAt: data.lastSyncAt || null });
  } catch (e) {
    console.error('garmin/status error', e);
    res.status(500).json({ error: "Impossible de vérifier l'état de la connexion Garmin." });
  }
});

// ============================================================================
// POST /api/garmin/connect — démarre le flux OAuth2 + PKCE, renvoie l'URL
// d'autorisation Garmin vers laquelle le frontend doit rediriger l'utilisateur.
// ============================================================================
app.post('/api/garmin/connect', requireFirebaseAuth, async (req, res) => {
  if (!GARMIN_CONFIGURED) {
    return res.status(501).json({
      error: "L'intégration Garmin n'est pas encore configurée sur le serveur (accès Developer Program et/ou clés manquants).",
      devMode: true
    });
  }
  try {
    const { state, codeChallenge } = await createGarminOAuthState(req.uid);
    const url = new URL(GARMIN_AUTH_URL);
    url.searchParams.set('client_id', GARMIN_CLIENT_ID);
    url.searchParams.set('response_type', 'code');
    url.searchParams.set('redirect_uri', GARMIN_REDIRECT_URI);
    url.searchParams.set('state', state);
    url.searchParams.set('code_challenge', codeChallenge);
    url.searchParams.set('code_challenge_method', 'S256');
    res.json({ authUrl: url.toString() });
  } catch (e) {
    console.error('garmin/connect error', e);
    res.status(500).json({ error: 'Impossible de démarrer la connexion Garmin. Réessaie plus tard.' });
  }
});

// ============================================================================
// GET /api/garmin/callback — Garmin redirige l'utilisateur ici après
// consentement. Pas de header Authorization disponible à ce stade (c'est une
// redirection de navigateur) : on retrouve l'utilisateur via le "state".
// ============================================================================
app.get('/api/garmin/callback', async (req, res) => {
  if (!GARMIN_CONFIGURED) return res.redirect(`${FRONTEND_URL}?garmin=error`);
  const { code, state, error: garminError } = req.query;
  try {
    if (garminError) return res.redirect(`${FRONTEND_URL}?garmin=denied`);
    if (!code || !state) return res.redirect(`${FRONTEND_URL}?garmin=error`);

    const stateRef = fsDb().collection('garminOAuthState').doc(String(state));
    const stateSnap = await stateRef.get();
    if (!stateSnap.exists || stateSnap.data().expiresAt < Date.now()) {
      return res.redirect(`${FRONTEND_URL}?garmin=expired`);
    }
    const { uid, codeVerifier } = stateSnap.data();
    await stateRef.delete();

    // Échange code → tokens. Adapter le format exact du corps de requête si
    // besoin une fois testé contre le vrai endpoint (spec PKCE Garmin).
    const tokenRes = await fetch(GARMIN_TOKEN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: GARMIN_CLIENT_ID,
        client_secret: GARMIN_CLIENT_SECRET,
        redirect_uri: GARMIN_REDIRECT_URI,
        code: String(code),
        code_verifier: codeVerifier
      })
    });
    if (!tokenRes.ok) {
      console.error('garmin/callback: échange de token échoué', tokenRes.status, await tokenRes.text());
      return res.redirect(`${FRONTEND_URL}?garmin=error`);
    }
    const tokens = await tokenRes.json();
    // { access_token, refresh_token, expires_in, refresh_token_expires_in, ... }

    // Récupère le Garmin User ID (identifiant stable utilisé par les webhooks
    // pour désigner cet utilisateur — voir /api/garmin/webhook).
    const userIdRes = await fetch(`${GARMIN_API_BASE}/wellness-api/rest/user/id`, {
      headers: { Authorization: `Bearer ${tokens.access_token}` }
    });
    const garminUserId = userIdRes.ok ? (await userIdRes.json()).userId : null;

    const now = Date.now();
    await fsDb().collection('garminConnections').doc(uid).set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      accessTokenExpiresAt: now + (tokens.expires_in || 0) * 1000,
      garminUserId,
      connectedAt: now,
      lastSyncAt: null
    });
    // Index inverse : nécessaire pour retrouver l'utilisateur de l'app à
    // partir du garminUserId reçu dans les webhooks (voir /api/garmin/webhook).
    if (garminUserId) {
      await fsDb().collection('garminUserIndex').doc(garminUserId).set({ uid });
    }

    res.redirect(`${FRONTEND_URL}?garmin=connected`);
  } catch (e) {
    console.error('garmin/callback error', e);
    res.redirect(`${FRONTEND_URL}?garmin=error`);
  }
});

// ============================================================================
// POST /api/garmin/sync — NE récupère PAS de données de façon synchrone :
// les APIs Health/Activity de Garmin fonctionnent en push (webhook), pas en
// pull. Cette route déclenche une demande de "backfill" (réémission des
// données récentes vers notre webhook) ; les données arrivent ensuite de façon
// asynchrone via /api/garmin/webhook, généralement en quelques secondes à
// quelques minutes. Le frontend doit donc afficher "Synchronisation demandée"
// plutôt que d'attendre une réponse contenant déjà les données.
// ============================================================================
const garminSyncInFlight = new Set(); // anti double-clic par utilisateur
app.post('/api/garmin/sync', requireFirebaseAuth, async (req, res) => {
  if (!GARMIN_CONFIGURED) {
    return res.status(501).json({ error: "L'intégration Garmin n'est pas encore configurée sur le serveur.", devMode: true });
  }
  if (garminSyncInFlight.has(req.uid)) {
    return res.status(409).json({ error: 'Une synchronisation est déjà en cours.' });
  }
  garminSyncInFlight.add(req.uid);
  try {
    const connSnap = await fsDb().collection('garminConnections').doc(req.uid).get();
    if (!connSnap.exists) return res.status(404).json({ error: "Garmin n'est pas connecté." });
    const conn = connSnap.data();

    // Fenêtre de backfill : dernières 24h (à ajuster une fois testé contre
    // l'API réelle — endpoint et paramètres exacts à confirmer dans la doc
    // Health API "Backfill" une fois l'accès au Developer Portal obtenu).
    const end = Math.floor(Date.now() / 1000);
    const start = end - 24 * 60 * 60;
    const backfillRes = await fetch(
      `${GARMIN_API_BASE}/wellness-api/rest/backfill/dailies?summaryStartTimeInSeconds=${start}&summaryEndTimeInSeconds=${end}`,
      { headers: { Authorization: `Bearer ${conn.accessToken}` } }
    );
    if (!backfillRes.ok) {
      console.error('garmin/sync: demande de backfill échouée', backfillRes.status, await backfillRes.text());
      return res.status(502).json({ error: 'Impossible de synchroniser Garmin. Réessaie plus tard.' });
    }
    await fsDb().collection('garminConnections').doc(req.uid).update({ lastSyncRequestedAt: Date.now() });
    res.json({ requested: true });
  } catch (e) {
    console.error('garmin/sync error', e);
    res.status(500).json({ error: 'Impossible de synchroniser Garmin. Réessaie plus tard.' });
  } finally {
    garminSyncInFlight.delete(req.uid);
  }
});

// ============================================================================
// POST /api/garmin/disconnect — supprime les tokens stockés côté serveur.
// ============================================================================
app.post('/api/garmin/disconnect', requireFirebaseAuth, async (req, res) => {
  if (!GARMIN_CONFIGURED) {
    return res.status(501).json({ error: "L'intégration Garmin n'est pas encore configurée sur le serveur.", devMode: true });
  }
  try {
    const connSnap = await fsDb().collection('garminConnections').doc(req.uid).get();
    if (connSnap.exists && connSnap.data().garminUserId) {
      await fsDb().collection('garminUserIndex').doc(connSnap.data().garminUserId).delete().catch(() => {});
    }
    await fsDb().collection('garminConnections').doc(req.uid).delete();
    res.json({ disconnected: true });
  } catch (e) {
    console.error('garmin/disconnect error', e);
    res.status(500).json({ error: 'Impossible de déconnecter Garmin pour le moment.' });
  }
});

// ============================================================================
// POST /api/garmin/webhook — endpoint PUBLIC appelé par les serveurs Garmin
// (pas par le frontend), déclaré dans le Developer Portal. Reçoit les
// résumés journaliers ("dailies") après chaque synchronisation d'un appareil.
// Pas d'authentification Firebase ici (Garmin ne connaît pas nos tokens
// Firebase) : l'utilisateur est retrouvé via garminUserIndex.
//
// ⚠️ Champs basés sur la spécification publique du "Daily Summary" de la
// Health REST API (calendarDate, steps, activeKilocalories, distanceInMeters,
// heart rate min/max/moyenne/repos...) — à revérifier contre un vrai payload
// de test une fois l'environnement d'évaluation Garmin accessible : Garmin
// peut envelopper ces objets différemment selon la version d'API accordée.
// ============================================================================
app.post('/api/garmin/webhook', async (req, res) => {
  if (!GARMIN_CONFIGURED) return res.status(501).json({ error: 'Garmin non configuré.' });
  try {
    const dailies = Array.isArray(req.body?.dailies) ? req.body.dailies : [];
    for (const daily of dailies) {
      const garminUserId = daily.userId;
      if (!garminUserId) continue;
      const indexSnap = await fsDb().collection('garminUserIndex').doc(garminUserId).get();
      if (!indexSnap.exists) continue; // utilisateur inconnu de notre app
      const uid = indexSnap.data().uid;
      const dateIso = daily.calendarDate; // format YYYY-MM-DD attendu par mrp-steps

      // Fusionne avec les pas existants (voir js/05-meals-weight-steps-settings.js
      // upsertGarminStepsEntry) sans jamais écraser une correction manuelle en
      // cours (override=true) — voir point 6/8 de la demande.
      await upsertGarminDailyServerSide(uid, dateIso, {
        steps: typeof daily.steps === 'number' ? daily.steps : null,
        caloriesBurned: typeof daily.activeKilocalories === 'number' ? daily.activeKilocalories : null,
        distanceMeters: typeof daily.distanceInMeters === 'number' ? daily.distanceInMeters : null,
        avgHeartRate: typeof daily.averageHeartRateInBeatsPerMinute === 'number' ? daily.averageHeartRateInBeatsPerMinute : null,
        restingHeartRate: typeof daily.restingHeartRateInBeatsPerMinute === 'number' ? daily.restingHeartRateInBeatsPerMinute : null
      });

      await fsDb().collection('garminConnections').doc(uid).update({ lastSyncAt: Date.now() }).catch(() => {});
    }
    res.json({ received: true });
  } catch (e) {
    console.error('garmin/webhook error', e);
    // Toujours répondre 200 côté webhook pour éviter que Garmin ne retente en
    // boucle sur une erreur de notre côté ; l'erreur reste loguée pour nous.
    res.status(200).json({ received: false });
  }
});

// Écrit les données Garmin du jour dans le même document Firestore que le
// frontend (users/{uid}/data/mrp-steps), en respectant le modèle
// garminValue/manualValue/effectiveValue/override déjà utilisé côté client
// (voir upsertGarminStepsEntry dans js/05-meals-weight-steps-settings.js) —
// on ne crée pas une deuxième structure de données pour les pas.
async function upsertGarminDailyServerSide(uid, dateIso, values) {
  if (!dateIso || values.steps == null) return;
  const ref = fsDb().collection('users').doc(uid).collection('data').doc('mrp-steps');
  const snap = await ref.get();
  const current = snap.exists ? snap.data().value : { goal: 10000, entries: [] };
  if (!Array.isArray(current.entries)) current.entries = [];

  const existing = current.entries.find(e => e.date === dateIso);
  if (existing) {
    existing.garminValue = values.steps;
    existing.source = 'garmin';
    if (!existing.override) existing.steps = values.steps; // pas de correction manuelle active : on suit Garmin
  } else {
    current.entries.push({
      id: crypto.randomBytes(6).toString('hex'), date: dateIso, steps: values.steps,
      garminValue: values.steps, manualValue: null, override: false, source: 'garmin'
    });
  }
  current.entries.sort((a, b) => a.date.localeCompare(b.date));
  await ref.set({ value: current });
}

app.post('/api/generate-recipe', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { ingredients } = req.body || {};
    if (!ingredients || typeof ingredients !== 'string' || !ingredients.trim()) {
      return res.status(400).json({ error: "Ingrédients manquants" });
    }
    // Garde-fou raisonnable : évite d'envoyer un prompt démesuré à Gemini
    // (coût, latence) si le champ contient un texte anormalement long.
    if (ingredients.length > 500) {
      return res.status(400).json({ error: "La liste d'ingrédients est trop longue (500 caractères max)." });
    }

    const response = await withTimeout(
      generateContentWithRetry({
        model: 'gemini-3.5-flash',
        contents: `Propose une recette simple avec ces ingrédients : ${ingredients}`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              prepTimeMinutes: { type: Type.NUMBER },
              ingredientsList: { type: Type.ARRAY, items: { type: Type.STRING } },
              instructions: { type: Type.ARRAY, items: { type: Type.STRING } }
            },
            required: ["title", "prepTimeMinutes", "ingredientsList", "instructions"]
          }
        }
      }),
      20000,
      "La génération de recette prend trop de temps. Réessaie."
    );

    // Comme pour les autres routes IA : Gemini peut bloquer la réponse (filtre
    // de sécurité) ou ne renvoyer aucun candidat exploitable, auquel cas
    // response.text est vide/undefined plutôt qu'une exception JS.
    if (!response.text) {
      const blockReason = response.promptFeedback?.blockReason;
      const finishReason = response.candidates?.[0]?.finishReason;
      console.error('generate-recipe: réponse vide de Gemini', { blockReason, finishReason });
      const msg = blockReason || finishReason === 'SAFETY'
        ? "Cette demande n'a pas pu être traitée (filtre de sécurité). Reformule tes ingrédients."
        : "L'IA n'a pas pu générer de recette. Réessaie.";
      return res.status(422).json({ error: msg });
    }

    let recipe;
    try {
      recipe = JSON.parse(response.text);
    } catch (parseErr) {
      console.error('generate-recipe: réponse Gemini non-JSON', response.text, parseErr);
      return res.status(502).json({ error: "L'IA a renvoyé une réponse inattendue. Réessaie." });
    }

    res.json({ success: true, recipe });
  } catch (error) {
    if (error?.isTimeout) {
      console.error('generate-recipe: timeout dépassé', error.message);
      return res.status(504).json({ error: error.message });
    }
    const status = error?.error?.status || error?.status;
    if (status === 'UNAVAILABLE' || error?.error?.code === 503) {
      console.error('generate-recipe: Gemini indisponible après plusieurs tentatives', error);
      return res.status(503).json({ error: "L'IA est temporairement surchargée. Réessaie dans quelques instants." });
    }
    if (status === 'RESOURCE_EXHAUSTED' || error?.error?.code === 429) {
      console.error('generate-recipe: quota Gemini dépassé', error);
      return res.status(429).json({ error: "Le quota d'analyses IA est temporairement dépassé. Réessaie plus tard." });
    }
    console.error('generate-recipe error', error);
    res.status(500).json({ error: "La génération de recette a échoué. Réessaie plus tard." });
  }
});

// ============================================================================
// Meals Tracker — analyse IA d'une photo de repas.
// Reprend le contrat déjà documenté côté frontend (js/05-meals-weight-steps-
// settings.js) et dans l'exemple de Cloud Function fourni (analyzeMealPhoto.js) :
// body { image: "data:image/...;base64,..." } -> { success, items: [...] }.
// ============================================================================
// Formats acceptés côté Gemini pour cette route. HEIC/HEIF est volontairement
// exclu : le frontend (prepareImageForAI) est censé l'avoir déjà rejeté ou
// converti avant l'envoi, mais on revalide ici côté serveur — ne jamais faire
// confiance uniquement au frontend pour la validation de format.
const MEAL_PHOTO_ALLOWED_MIME = ['image/jpeg', 'image/png', 'image/webp'];
// Le frontend cible ~1 Mo après compression ; on laisse une marge confortable
// avant la limite express de 10mb (qui inclut l'overhead JSON + le reste du body).
const MEAL_PHOTO_MAX_BYTES = 8 * 1024 * 1024;

app.post('/api/analyze-meal-photo', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { image } = req.body;
    if (!image || typeof image !== 'string') {
      return res.status(400).json({ error: "Aucune image fournie" });
    }
    const match = image.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
    if (!match) {
      return res.status(400).json({ error: "Format d'image invalide (data URL attendue)" });
    }
    const [, mimeType, base64Data] = match;

    if (!MEAL_PHOTO_ALLOWED_MIME.includes(mimeType.toLowerCase())) {
      return res.status(400).json({ error: "Ce format d'image n'est pas compatible. Essaie avec une photo JPG, PNG ou WEBP." });
    }
    // Taille réelle approximative des octets bruts derrière le base64 (base64
    // gonfle la taille d'environ 4/3) : on rejette tôt plutôt que de laisser
    // Gemini échouer plus loin avec une erreur moins compréhensible.
    const approxBytes = Math.ceil(base64Data.length * 3 / 4);
    if (approxBytes > MEAL_PHOTO_MAX_BYTES) {
      return res.status(413).json({ error: "Cette image est trop volumineuse. Réessaie avec une photo plus légère." });
    }

    const response = await withTimeout(
      generateContentWithRetry({
      // gemini-2.5-flash (utilisé jusqu'ici dans une des deux copies du fichier
      // backend) est en cours de dépréciation (arrêt prévu le 16/20 octobre 2026).
      // gemini-3.5-flash-lite est le modèle actuel recommandé pour ce type de
      // tâche : multimodal (accepte l'image en entrée), rapide et nettement
      // moins coûteux que gemini-3.5-flash pour une simple extraction structurée.
      model: 'gemini-3.5-flash-lite',
      contents: [
        { inlineData: { mimeType, data: base64Data } },
        { text:
            "Identifie chaque aliment visible sur cette photo de repas. Pour chacun, " +
            "fournis son nom, une estimation de sa quantité/portion, et une ESTIMATION " +
            "OBLIGATOIRE de ses calories (kcal), protéines, glucides et lipides (en " +
            "grammes) — base-toi sur des valeurs nutritionnelles typiques pour ce type " +
            "d'aliment et la quantité estimée ; ne laisse jamais ces champs vides ou à " +
            "0 sauf si l'aliment est réellement négligeable en apport (ex. une feuille " +
            "de menthe). Indique 'low' comme confidence si tu n'es pas sûr d'un aliment " +
            "ou d'une quantité, sinon 'medium' ou 'high'."
        }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            items: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  quantity: { type: Type.STRING },
                  calories: { type: Type.NUMBER },
                  protein: { type: Type.NUMBER },
                  carbs: { type: Type.NUMBER },
                  fat: { type: Type.NUMBER },
                  confidence: { type: Type.STRING }
                },
                // "name" seul était obligatoire ici auparavant : Gemini identifiait
                // bien l'aliment mais omettait souvent les champs numériques
                // (optionnels dans le schema), qui devenaient alors 0 côté frontend
                // (Number(undefined) || 0). En rendant les macros obligatoires, on
                // force le modèle à toujours fournir une estimation.
                required: ["name", "quantity", "calories", "protein", "carbs", "fat", "confidence"]
              }
            }
          },
          required: ["items"]
        }
      }
      }),
      25000,
      "L'analyse prend trop de temps. Vérifie ta connexion et réessaie."
    );

    // Certaines photos font que Gemini bloque la réponse (filtre de sécurité) ou
    // ne renvoie aucun candidat exploitable : response.text est alors vide/undefined
    // plutôt qu'une exception JS. Sans ce contrôle, JSON.parse plante silencieusement
    // et l'erreur générique masque la vraie cause — d'où ce log détaillé pour
    // diagnostiquer facilement les échecs "aléatoires" liés à une photo précise.
    const blockReason = response.promptFeedback?.blockReason;
    const finishReason = response.candidates?.[0]?.finishReason;
    if (!response.text) {
      console.error('analyze-meal-photo: réponse vide de Gemini', { blockReason, finishReason });
      const msg = blockReason || finishReason === 'SAFETY'
        ? "Cette photo n'a pas pu être analysée (filtre de sécurité). Essaie une autre photo."
        : "L'IA n'a pas pu analyser cette photo. Réessaie avec une photo plus nette.";
      return res.status(422).json({ error: msg });
    }

    let parsed;
    try {
      // Avec responseMimeType: "application/json", Gemini ne devrait pas
      // entourer sa réponse de blocs Markdown ```json ... ``` — mais on le
      // tolère quand même ici en défense, au cas où le modèle s'en écarte.
      const cleanText = response.text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
      parsed = JSON.parse(cleanText);
    } catch (parseErr) {
      console.error('analyze-meal-photo: réponse Gemini non-JSON', response.text, parseErr);
      return res.status(502).json({ error: "L'IA a renvoyé une réponse inattendue. Réessaie." });
    }

    // Validation de structure : un item sans nom exploitable est ignoré plutôt
    // que transmis tel quel au frontend, qui suppose déjà que `name` est fiable.
    const items = Array.isArray(parsed?.items)
      ? parsed.items.filter(it => it && typeof it === 'object' && typeof it.name === 'string' && it.name.trim())
      : [];

    res.json({ success: true, items });
  } catch (error) {
    if (error?.isTimeout) {
      console.error('analyze-meal-photo: timeout dépassé', error.message);
      return res.status(504).json({ error: error.message });
    }
    const status = error?.error?.status || error?.status;
    if (status === 'UNAVAILABLE' || error?.error?.code === 503) {
      console.error('analyze-meal-photo: Gemini indisponible après plusieurs tentatives', error);
      return res.status(503).json({ error: "L'IA est temporairement surchargée. Réessaie dans quelques instants." });
    }
    if (status === 'RESOURCE_EXHAUSTED' || error?.error?.code === 429) {
      console.error('analyze-meal-photo: quota Gemini dépassé', error);
      return res.status(429).json({ error: "Le quota d'analyses IA est temporairement dépassé. Réessaie plus tard." });
    }
    console.error('analyze-meal-photo error', error);
    res.status(500).json({ error: "L'analyse a échoué. Réessaie plus tard." });
  }
});

// ============================================================================
// Life Goals — évaluation IA de la progression d'un objectif.
// Reprend le contrat déjà documenté dans js/06-lifegoals-ai.js (GoalAiProviders)
// et dans l'exemple de Cloud Function fourni (evaluateLifeGoal.js) :
// body { goal, userData, answers, history } -> { success, score, reasoning, recommendations }.
// ============================================================================
app.post('/api/evaluate-life-goal', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { goal, userData, answers, history } = req.body || {};
    if (!goal || !goal.title || !Array.isArray(answers)) {
      return res.status(400).json({ error: "Données d'objectif ou réponses manquantes" });
    }

    const prompt =
      `Objectif de l'utilisateur : "${goal.title}". Description : "${goal.notes || ""}". ` +
      `Progression actuelle affichée : ${userData && userData.currentProgress != null ? userData.currentProgress : "inconnue"}%. ` +
      `Réponses au questionnaire : ${JSON.stringify(answers)}. ` +
      `Historique des évaluations précédentes : ${JSON.stringify(history || [])}. ` +
      `À partir de ces éléments, évalue le niveau d'avancement réel de l'utilisateur ` +
      `vers son objectif. Ne reprends pas simplement le pourcentage affiché : déduis ` +
      `un score réaliste à partir des réponses. Explique ton raisonnement en français, ` +
      `en 2-3 phrases, et propose 1 à 3 recommandations concrètes.`;

    const response = await generateContentWithRetry({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: { type: Type.NUMBER },
            reasoning: { type: Type.STRING },
            recommendations: { type: Type.ARRAY, items: { type: Type.STRING } }
          },
          required: ["score", "reasoning"]
        }
      }
    });

    const parsed = JSON.parse(response.text);
    const score = Math.max(0, Math.min(100, Number(parsed.score) || 0));
    res.json({
      success: true,
      score,
      reasoning: typeof parsed.reasoning === 'string' ? parsed.reasoning : "Évaluation indisponible.",
      recommendations: Array.isArray(parsed.recommendations) ? parsed.recommendations : []
    });
  } catch (error) {
    console.error('evaluate-life-goal error', error);
    res.status(500).json({ error: "L'évaluation a échoué. Réessaie plus tard." });
  }
});

// ============================================================================
// Gym — analyse IA des calories brûlées après une séance.
// Reprend le contrat côté frontend (js/04-gym.js) : body { session, answers,
// standardEstimate } -> { success, calories, reasoning }. "session" décrit les
// exercices réellement effectués (séries faites, poids, durée) ; "answers" les
// réponses de l'utilisateur au petit questionnaire sur le ressenti de l'effort.
// L'estimation IA vient COMPLÉTER l'estimation MET déjà calculée localement
// (standardEstimate), pas la remplacer par défaut — le frontend laisse le choix
// à l'utilisateur.
// ============================================================================
app.post('/api/analyze-workout', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { session, answers, standardEstimate } = req.body || {};
    if (!session || !Array.isArray(session.exercises) || !session.exercises.length) {
      return res.status(400).json({ error: "Données de séance manquantes" });
    }

    const exercisesDesc = session.exercises.map(e =>
      `- ${e.name} (${e.type || 'inconnu'}) : ${e.setsDone ?? '?'} série(s) faite(s)` +
      (e.type === 'cardio' ? `, ${e.totalMinutes ?? '?'} min au total` : `, ${e.reps ?? '?'} répétitions par série`) +
      (e.weight ? `, ${e.weight} kg` : '')
    ).join('\n');

    const answersDesc = Array.isArray(answers) && answers.length
      ? answers.map(a => `- ${a.question} : ${a.answer}`).join('\n')
      : "Aucune réponse fournie.";

    const prompt =
      `Tu es coach sportif. Un utilisateur de ${session.bodyWeightKg || 70} kg vient de terminer la séance ` +
      `"${session.name}", d'une durée totale de ${session.durationMinutes} minutes. Voici les exercices réellement ` +
      `effectués :\n${exercisesDesc}\n\nRéponses de l'utilisateur sur son ressenti pendant la séance :\n${answersDesc}\n\n` +
      `Une estimation standard basée sur les valeurs MET donne ${standardEstimate ?? '?'} kcal. À partir de tous ces ` +
      `éléments (exercices, durée, poids de corps, ressenti d'effort réel), donne une estimation plus précise et ` +
      `réaliste des calories réellement brûlées pendant cette séance. Explique ton raisonnement en français, en 2-3 ` +
      `phrases, en mentionnant ce qui t'a fait ajuster (ou non) l'estimation standard.`;

    const response = await generateContentWithRetry({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            calories: { type: Type.NUMBER },
            reasoning: { type: Type.STRING }
          },
          required: ["calories", "reasoning"]
        }
      }
    });

    if (!response.text) {
      console.error('analyze-workout: réponse vide de Gemini');
      return res.status(422).json({ error: "L'IA n'a pas pu analyser cette séance. Réessaie." });
    }

    let parsed;
    try {
      parsed = JSON.parse(response.text);
    } catch (parseErr) {
      console.error('analyze-workout: réponse Gemini non-JSON', response.text, parseErr);
      return res.status(502).json({ error: "L'IA a renvoyé une réponse inattendue. Réessaie." });
    }

    const calories = Math.max(0, Math.round(Number(parsed.calories) || 0));
    res.json({
      success: true,
      calories,
      reasoning: typeof parsed.reasoning === 'string' ? parsed.reasoning : "Analyse indisponible."
    });
  } catch (error) {
    const status = error?.error?.status || error?.status;
    if (status === 'UNAVAILABLE' || error?.error?.code === 503) {
      console.error('analyze-workout: Gemini indisponible après plusieurs tentatives', error);
      return res.status(503).json({ error: "L'IA est temporairement surchargée. Réessaie dans quelques instants." });
    }
    console.error('analyze-workout error', error);
    res.status(500).json({ error: "L'analyse a échoué. Réessaie plus tard." });
  }
});

// ============================================================================
// Assistant — conversation libre avec le modèle IA.
// body { message, context } -> { success, reply }.
// Pas d'historique conservé côté serveur : le frontend renvoie le message
// courant à chaque appel (voir assistantConversation côté client) ; on reste
// donc sans état ici, simple et facile à faire évoluer plus tard.
// ============================================================================
app.post('/api/assistant-chat', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { message, context } = req.body || {};
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: "Message manquant" });
    }

    const contextNote = context && context.activeTab
      ? `L'utilisateur se trouve actuellement sur l'onglet "${context.activeTab}" de l'application. `
      : '';

    const response = await generateContentWithRetry({
      model: 'gemini-3.5-flash',
      contents:
        `Tu es l'assistant intégré de l'application "My RoutinePov" (suivi personnel : ` +
        `objectifs, tâches, finances, recettes, voyages, sport, repas, poids, pas). ` +
        `${contextNote}Réponds de façon utile, concise et en français au message suivant : "${message}"`
    });

    res.json({ success: true, reply: response.text });
  } catch (error) {
    console.error('assistant-chat error', error);
    res.status(500).json({ error: "L'assistant n'a pas pu répondre. Réessaie plus tard." });
  }
});

// ============================================================================
// Assistant vocal — transcription audio -> texte.
// Cette route NE fait QUE transcrire : elle ne répond pas à la place de
// l'assistant et ne prend aucune décision. Le frontend récupère le texte
// renvoyé ici et le fait ensuite transiter par le MÊME pipeline que les
// messages tapés au clavier (sendMessageToAI -> /api/assistant-chat, voir
// js/01-core-home-goals-todos.js). Aucun deuxième système IA : on réutilise
// le même client Gemini (ai / generateContentWithRetry) que tout le reste de
// ce fichier, et aucune clé d'API n'est exposée côté frontend — l'audio brut
// est envoyé au backend, qui seul contacte Gemini.
// body { audio: <data URL base64, ex. "data:audio/webm;base64,..."> }
//   -> { success, text }
// ============================================================================
const VOICE_AUDIO_ALLOWED_MIME = ['audio/webm', 'audio/ogg', 'audio/mp4', 'audio/mpeg', 'audio/wav', 'audio/x-m4a', 'audio/aac'];
// Un enregistrement vocal de commande reste court (quelques secondes à ~60s,
// la limite appliquée côté frontend) : 8 Mo laisse une marge très confortable
// même en qualité non compressée, tout en restant sous la limite express (10mb).
const VOICE_AUDIO_MAX_BYTES = 8 * 1024 * 1024;

app.post('/api/transcribe-audio', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { audio } = req.body || {};
    if (!audio || typeof audio !== 'string') {
      return res.status(400).json({ error: "Aucun enregistrement audio fourni" });
    }
    const match = audio.match(/^data:(audio\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
    if (!match) {
      return res.status(400).json({ error: "Format audio invalide (data URL attendue)" });
    }
    const [, mimeTypeRaw, base64Data] = match;
    // Certains navigateurs ajoutent des paramètres au MIME type du MediaRecorder
    // (ex. "audio/webm;codecs=opus") : on ne garde que la partie avant le ';'
    // pour la validation, mais on transmet le type complet à Gemini tel quel.
    const mimeType = mimeTypeRaw.split(';')[0].trim();

    if (!VOICE_AUDIO_ALLOWED_MIME.includes(mimeType.toLowerCase())) {
      return res.status(400).json({ error: "Ce format audio n'est pas pris en charge." });
    }
    const approxBytes = Math.ceil(base64Data.length * 3 / 4);
    if (approxBytes > VOICE_AUDIO_MAX_BYTES) {
      return res.status(413).json({ error: "Cet enregistrement est trop long. Réessaie avec un message plus court." });
    }

    const response = await withTimeout(
      generateContentWithRetry({
        model: 'gemini-3.5-flash',
        contents: [
          { inlineData: { mimeType: mimeTypeRaw, data: base64Data } },
          { text:
              "Transcris fidèlement, mot pour mot, la parole contenue dans cet enregistrement " +
              "audio, en français. Ne réponds pas à la demande formulée dans l'audio, ne " +
              "l'interprète pas, ne la reformule pas : renvoie uniquement le texte exact " +
              "prononcé. Si l'audio est silencieux, inaudible, ou ne contient aucune parole " +
              "exploitable, renvoie une chaîne vide pour \"text\"."
          }
        ],
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              text: { type: Type.STRING }
            },
            required: ["text"]
          }
        }
      }),
      25000,
      "La transcription prend trop de temps. Vérifie ta connexion et réessaie."
    );

    const blockReason = response.promptFeedback?.blockReason;
    const finishReason = response.candidates?.[0]?.finishReason;
    if (!response.text) {
      console.error('transcribe-audio: réponse vide de Gemini', { blockReason, finishReason });
      const msg = blockReason || finishReason === 'SAFETY'
        ? "Cet enregistrement n'a pas pu être transcrit (filtre de sécurité). Réessaie ou écris ton message."
        : "La transcription a échoué. Réessaie ou écris ton message.";
      return res.status(422).json({ error: msg });
    }

    let parsed;
    try {
      const cleanText = response.text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
      parsed = JSON.parse(cleanText);
    } catch (parseErr) {
      console.error('transcribe-audio: réponse Gemini non-JSON', response.text, parseErr);
      return res.status(502).json({ error: "La transcription a renvoyé une réponse inattendue. Réessaie." });
    }

    const text = typeof parsed.text === 'string' ? parsed.text.trim() : '';
    res.json({ success: true, text });
  } catch (error) {
    if (error?.isTimeout) {
      console.error('transcribe-audio: timeout dépassé', error.message);
      return res.status(504).json({ error: error.message });
    }
    const status = error?.error?.status || error?.status;
    if (status === 'UNAVAILABLE' || error?.error?.code === 503) {
      console.error('transcribe-audio: Gemini indisponible après plusieurs tentatives', error);
      return res.status(503).json({ error: "La transcription est temporairement surchargée. Réessaie dans quelques instants." });
    }
    if (status === 'RESOURCE_EXHAUSTED' || error?.error?.code === 429) {
      console.error('transcribe-audio: quota Gemini dépassé', error);
      return res.status(429).json({ error: "Le quota d'analyses IA est temporairement dépassé. Réessaie plus tard." });
    }
    console.error('transcribe-audio error', error);
    res.status(500).json({ error: "La transcription a échoué. Réessaie plus tard." });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Serveur prêt sur le port ${PORT}`));