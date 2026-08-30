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
// Subscription — statut FREE / TRIAL / PREMIUM de l'utilisateur.
// ----------------------------------------------------------------------------
// Document stocké dans users/{uid}/subscription/current, dans une collection
// "subscription" séparée de "data" (users/{uid}/data/{key} — voir dbGet/dbSet
// côté frontend). Cette séparation est volontaire : contrairement aux clés de
// "data", ce document ne doit JAMAIS pouvoir être écrit par le frontend (voir
// Firestore Security Rules) — seul ce backend (plus tard : webhook Stripe ou
// action admin) pourra le modifier. À ce stade, AUCUNE écriture n'existe
// encore : uniquement la lecture et la valeur par défaut.
//
// Tous les utilisateurs (existants ou nouveaux) sans document sont considérés
// FREE/none par défaut, sans qu'il soit nécessaire de créer ce document pour
// chacun d'eux.
// ============================================================================
const DEFAULT_SUBSCRIPTION = {
  plan: 'free',
  status: 'none',
  // "source" et "expiresAt" ajoutés à cette étape (gestion des accès Premium
  // offerts par un admin) — fusionnés avec les champs déjà posés à l'étape 2
  // (trialStart..stripeSubscriptionId), pas de remplacement de la structure.
  source: 'none',
  trialStart: null,
  trialEnd: null,
  currentPeriodStart: null,
  currentPeriodEnd: null,
  billingInterval: null,
  expiresAt: null,
  stripeCustomerId: null,
  stripeSubscriptionId: null,
  updatedAt: null
};

// Point d'entrée UNIQUE pour lire le statut d'abonnement d'un utilisateur —
// à réutiliser partout où ce statut est nécessaire côté serveur, pour éviter
// que plusieurs bouts de code n'interprètent différemment un document
// manquant ou partiel. Renvoie toujours une structure complète.
async function getSubscription(uid) {
  try {
    const snap = await fsDb().collection('users').doc(uid).collection('subscription').doc('current').get();
    if (!snap.exists) {
      console.log('[SUBSCRIPTION] no subscription document → default FREE');
      return { ...DEFAULT_SUBSCRIPTION };
    }
    // Fusionné avec les valeurs par défaut : un document existant mais
    // partiel (créé manuellement, ou futur champ ajouté plus tard) reste
    // toujours complet côté appelant.
    return { ...DEFAULT_SUBSCRIPTION, ...snap.data() };
  } catch (e) {
    console.error('[SUBSCRIPTION] Firestore read failed', e);
    return { ...DEFAULT_SUBSCRIPTION };
  }
}

// Convertit une valeur trialEnd potentiellement hétérogène (Firestore Timestamp,
// ISO string, ou millis) en objet Date. Rien n'écrit encore ce champ (il reste
// toujours null pour l'instant), mais getEffectivePlan doit rester robuste peu
// importe la représentation choisie plus tard (Stripe, admin, etc.).
function toDateSafe(value) {
  if (!value) return null;
  if (typeof value.toDate === 'function') return value.toDate(); // Firestore Timestamp
  if (typeof value === 'number') return new Date(value);
  const d = new Date(value);
  return isNaN(d.getTime()) ? null : d;
}

// ============================================================================
// hasPremiumAccess — fonction centrale unique décidant si un utilisateur a
// accès aux fonctionnalités Premium, MAINTENANT. Ne modifie JAMAIS le
// document Firestore (pas de downgrade automatique écrit ici) : un Premium
// expiré (trial ou cadeau) est simplement traité comme sans accès pour cette
// requête — le document reste tel quel, contrôlé en temps réel à chaque
// appel, sans avoir besoin d'un cron pour "corriger" la valeur stockée.
//
//   - plan "admin"           → accès Premium toujours accordé (jamais soumis
//                               à expiresAt : un rôle admin ne "périme" pas).
//   - plan "premium", actif  → accès accordé si pas d'expiration dépassée.
//     L'expiration pertinente dépend du statut : trialEnd pendant un essai,
//     sinon expiresAt (cadeau admin, ou futur renouvellement payant).
//   - tout le reste (dont "free")            → pas d'accès.
// ============================================================================
function hasPremiumAccess(subscription) {
  const sub = subscription || DEFAULT_SUBSCRIPTION;
  if (sub.plan === 'admin') return true;
  if (sub.plan !== 'premium') return false;
  if (sub.status !== 'active' && sub.status !== 'trialing') return false;

  const expiry = sub.status === 'trialing' ? sub.trialEnd : sub.expiresAt;
  if (expiry) {
    const expiryDate = toDateSafe(expiry);
    if (expiryDate && expiryDate.getTime() < Date.now()) return false; // expiré
  }
  return true; // pas d'expiration = permanent
}

// Rôle admin — distinct de l'accès Premium (un admin A un accès Premium via
// hasPremiumAccess, mais isAdmin sert à protéger les routes /api/admin/*
// elles-mêmes, où "a accès aux fonctionnalités Premium" ne suffit pas).
function isAdmin(subscription) {
  const sub = subscription || DEFAULT_SUBSCRIPTION;
  return sub.plan === 'admin';
}

// Plan effectif "free"/"premium" utilisé par le système de limites de
// l'étape 3 (FEATURE_LIMITS n'a que ces deux clés) — admin s'y range du côté
// premium via hasPremiumAccess, sans dupliquer la logique d'expiration.
function getEffectivePlan(subscription) {
  return hasPremiumAccess(subscription) ? 'premium' : 'free';
}

// ============================================================================
// Table centrale des limites par plan.
// ----------------------------------------------------------------------------
// - Un nombre = limite de QUANTITÉ (compte des éléments existants).
// - { count, window } = limite TEMPORELLE (compteur d'utilisation par
//   semaine/mois, voir users/{uid}/usage/{feature-periodKey} plus bas).
// - Infinity = illimité. Volontairement gardé en Infinity (pas en null) dans
//   cette constante JS : Infinity se sérialise nativement en `null` via
//   JSON.stringify/res.json, donc le frontend reçoit bien `null = unlimited`
//   sans conversion manuelle à faire à chaque endroit — un seul sentinel dans
//   tout le fichier.
// ============================================================================
const FEATURE_LIMITS = {
  free: {
    trips: 1,
    lifePlannerActive: 5,
    gymSessions: 7,
    transactions: 10,
    recipes: 20,
    customFoods: 20,
    lifeGoals: 3,
    todoLists: 2,
    tasks: 50,
    assistantChat: { count: 3, window: 'week' },
    assistantVoice: { count: 3, window: 'week' },
    mealPhotoAnalysis: { count: 3, window: 'month' },
    statsHistoryDays: 7,
    generalHistoryDays: 30
  },
  premium: {
    trips: Infinity,
    lifePlannerActive: Infinity,
    gymSessions: Infinity,
    transactions: Infinity,
    recipes: Infinity,
    customFoods: Infinity,
    lifeGoals: Infinity,
    todoLists: Infinity,
    tasks: Infinity,
    assistantChat: { count: Infinity, window: 'week' },
    assistantVoice: { count: Infinity, window: 'week' },
    mealPhotoAnalysis: { count: Infinity, window: 'month' },
    statsHistoryDays: Infinity,
    generalHistoryDays: Infinity
  }
};

// "Features" purement informatives (nombre de jours d'historique affiché) :
// il n'y a rien à "autoriser/refuser" ici, donc canUseFeature() les traite à
// part plutôt que d'inventer une notion de "used" qui n'aurait pas de sens.
const DISPLAY_ONLY_FEATURES = new Set(['statsHistoryDays', 'generalHistoryDays']);

// ============================================================================
// Sources de comptage pour les limites de QUANTITÉ — une entrée par feature,
// pointant vers la même clé Firestore (users/{uid}/data/{key}) que celle déjà
// utilisée par dbGet/dbSet côté frontend (voir js/01-core-home-goals-todos.js
// et consorts). Vérifié dans le code actuel de chaque onglet avant d'écrire
// ceci — aucune structure supposée :
//   - trips            -> mrp-voyages-advanced : { trips: [...] }
//   - gymSessions       -> mrp-gym-history      : [...] (tableau direct)
//   - transactions      -> mrp-budget           : [...] (tableau direct)
//   - recipes           -> mrp-recipes          : [...] (tableau direct)
//   - lifeGoals         -> mrp-goals            : [...] (tableau direct)
//   - todoLists         -> mrp-todo-lists       : { lists: [...], activeListId }
//   - tasks             -> mrp-todos            : [...] (tableau direct)
//   - lifePlannerActive -> mrp-lifeplanner      : [...] de { start, end, ... } ;
//     "actives" = période dont l'intervalle [start,end] couvre la date du jour
//     (pas juste le total de périodes créées, vu le nom "Active").
//
// customFoods : AUCUNE clé de données correspondante n'existe aujourd'hui dans
// l'app (pas de "custom foods" séparé trouvé dans js/05-meals-weight-steps-
// settings.js ni ailleurs — seules mrp-meals { entries } et weight-entries
// existent). Plutôt que d'inventer une structure, cette feature reste dans
// FEATURE_LIMITS (demandé), mais SANS source de comptage : canUseFeature() le
// signale explicitement via `unsupported: true` au lieu de faire semblant de
// la compter. Voir rapport.
// ============================================================================
const COUNT_SOURCES = {
  trips: { dataKey: 'mrp-voyages-advanced', extract: v => Array.isArray(v?.trips) ? v.trips.length : 0 },
  gymSessions: { dataKey: 'mrp-gym-history', extract: v => Array.isArray(v) ? v.length : 0 },
  transactions: { dataKey: 'mrp-budget', extract: v => Array.isArray(v) ? v.length : 0 },
  recipes: { dataKey: 'mrp-recipes', extract: v => Array.isArray(v) ? v.length : 0 },
  lifeGoals: { dataKey: 'mrp-goals', extract: v => Array.isArray(v) ? v.length : 0 },
  todoLists: { dataKey: 'mrp-todo-lists', extract: v => Array.isArray(v?.lists) ? v.lists.length : 0 },
  tasks: { dataKey: 'mrp-todos', extract: v => Array.isArray(v) ? v.length : 0 },
  lifePlannerActive: {
    dataKey: 'mrp-lifeplanner',
    extract: v => {
      if (!Array.isArray(v)) return 0;
      const todayIso = new Date().toISOString().slice(0, 10);
      return v.filter(p => p && p.start && p.end && p.start <= todayIso && p.end >= todayIso).length;
    }
  }
};

async function countFeatureUsage(uid, source) {
  try {
    const snap = await fsDb().collection('users').doc(uid).collection('data').doc(source.dataKey).get();
    const value = snap.exists ? snap.data().value : undefined;
    return source.extract(value);
  } catch (e) {
    console.error(`[FEATURES] Lecture Firestore échouée pour ${source.dataKey}`, e);
    return 0;
  }
}

// ============================================================================
// Limites TEMPORELLES — users/{uid}/usage/{feature-periodKey}, ex.
// "assistantChat-2026-W34" ou "mealPhotoAnalysis-2026-08" (structure proposée
// telle quelle dans la demande). Cette étape prépare uniquement la LECTURE :
// aucune route existante n'incrémente encore ces compteurs (voir rapport),
// donc used vaudra 0 partout tant que rien n'écrit ce document.
// ============================================================================
function getIsoWeekNumber(date) {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
}

function getUsagePeriodKey(window) {
  const now = new Date();
  if (window === 'week') return `${now.getUTCFullYear()}-W${String(getIsoWeekNumber(now)).padStart(2, '0')}`;
  if (window === 'month') return `${now.getUTCFullYear()}-${String(now.getUTCMonth() + 1).padStart(2, '0')}`;
  return null;
}

async function getUsageCount(uid, feature, window) {
  const periodKey = getUsagePeriodKey(window);
  if (!periodKey) return 0;
  try {
    const docId = `${feature}-${periodKey}`;
    const snap = await fsDb().collection('users').doc(uid).collection('usage').doc(docId).get();
    const count = snap.exists ? snap.data().count : 0;
    return typeof count === 'number' ? count : 0;
  } catch (e) {
    console.error(`[FEATURES] Lecture usage échouée pour ${feature}`, e);
    return 0;
  }
}

// Incrémente le compteur d'usage temporel d'une feature — à appeler APRÈS un
// appel IA réussi (pas avant, pour ne jamais compter un appel qui a échoué).
// Utilise FieldValue.increment pour rester correct même en cas d'appels
// concurrents, sans avoir besoin de lire puis d'écrire séparément.
async function incrementUsageCount(uid, feature, window) {
  const periodKey = getUsagePeriodKey(window);
  if (!periodKey) return;
  try {
    const docId = `${feature}-${periodKey}`;
    const ref = fsDb().collection('users').doc(uid).collection('usage').doc(docId);
    await ref.set({
      count: admin.firestore.FieldValue.increment(1),
      feature,
      periodKey,
      updatedAt: Date.now()
    }, { merge: true });
  } catch (e) {
    console.error(`[FEATURES] Incrément usage échoué pour ${feature}`, e);
  }
}

// ============================================================================
// canUseFeature — fonction centrale d'autorisation, réutilisable partout.
// N'écrit rien, ne bloque rien : renvoie juste un résultat structuré. Accepte
// en option une subscription déjà récupérée pour éviter une lecture Firestore
// redondante quand plusieurs features sont vérifiées d'affilée (voir
// /api/subscription/limits ci-dessous).
// ============================================================================
async function evaluateFeatureLimit(uid, effectivePlan, feature) {
  const planLimits = FEATURE_LIMITS[effectivePlan];
  if (!planLimits || !(feature in planLimits)) {
    return { allowed: false, reason: 'UNKNOWN_FEATURE', plan: effectivePlan, feature, used: null, limit: null, remaining: null };
  }

  if (DISPLAY_ONLY_FEATURES.has(feature)) {
    const limit = planLimits[feature];
    return {
      allowed: true, plan: effectivePlan, feature,
      used: null, limit: limit === Infinity ? null : limit, remaining: null,
      notApplicable: true // valeur d'affichage (jours d'historique), pas une action à autoriser/refuser
    };
  }

  const limitConfig = planLimits[feature];
  const isTemporal = limitConfig && typeof limitConfig === 'object';
  const limit = isTemporal ? limitConfig.count : limitConfig;

  let used;
  if (isTemporal) {
    used = await getUsageCount(uid, feature, limitConfig.window);
  } else {
    const source = COUNT_SOURCES[feature];
    if (!source) {
      console.warn(`[FEATURES] Pas de source de comptage pour "${feature}" — vérification ignorée.`);
      return {
        allowed: true, plan: effectivePlan, feature,
        used: null, limit: limit === Infinity ? null : limit, remaining: null,
        unsupported: true
      };
    }
    used = await countFeatureUsage(uid, source);
  }

  const unlimited = limit === Infinity;
  const remaining = unlimited ? null : Math.max(0, limit - used);
  const allowed = unlimited || used < limit;

  const result = { allowed, plan: effectivePlan, feature, used, limit: unlimited ? null : limit, remaining };
  if (!allowed) result.reason = 'LIMIT_REACHED';
  return result;
}

async function canUseFeature(uid, feature, { subscription } = {}) {
  const sub = subscription || await getSubscription(uid);
  const effectivePlan = getEffectivePlan(sub);
  return evaluateFeatureLimit(uid, effectivePlan, feature);
}

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
    // Ce middleware protège désormais aussi les routes subscription/admin, pas
    // seulement Garmin — message généralisé (comportement inchangé : toujours
    // un 501 devMode tant que FIREBASE_SERVICE_ACCOUNT_JSON n'est pas défini).
    return res.status(501).json({ error: "Cette fonctionnalité n'est pas encore configurée sur ce serveur (Firebase Admin manquant).", devMode: true });
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
// requireAdmin — à chaîner APRÈS requireFirebaseAuth sur toute route
// /api/admin/*. req.uid est déjà posé à ce stade.
// ----------------------------------------------------------------------------
// Deux façons d'être reconnu admin (l'une bootstrap, l'autre définitive) :
//   1. ADMIN_UIDS (variable d'environnement, même mécanisme que
//      AI_RATE_LIMIT_EXEMPT_UIDS ci-dessus) — sert à démarrer : le tout
//      premier admin n'a pas de document subscription avec plan:"admin" tant
//      que personne ne peut encore le lui donner.
//   2. subscription.plan === "admin" dans Firestore — la voie normale une
//      fois qu'un premier admin existe (édité manuellement dans Firestore
//      pour l'instant : aucune route de cette étape ne permet de PROMOUVOIR
//      quelqu'un admin depuis l'app, volontairement, voir rapport).
// Un utilisateur ne peut jamais se rendre admin lui-même via ces routes.
// ============================================================================
const ADMIN_UIDS = new Set(
  (process.env.ADMIN_UIDS || '')
    .split(',')
    .map(uid => uid.trim())
    .filter(Boolean)
);

async function requireAdmin(req, res, next) {
  try {
    if (ADMIN_UIDS.has(req.uid)) return next();
    const subscription = await getSubscription(req.uid);
    if (isAdmin(subscription)) return next();
    return res.status(403).json({ error: "Accès administrateur requis." });
  } catch (e) {
    console.error('[ADMIN] Vérification des droits admin échouée', e);
    res.status(500).json({ error: "Vérification des droits administrateur impossible pour le moment." });
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
const FRONTEND_URL = process.env.CLIENT_URL_FOR_REDIRECT || 'https://my-routinepov.web.app';

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

// ============================================================================
// GET /api/subscription/status — statut d'abonnement (FREE / PREMIUM / ADMIN)
// de l'utilisateur connecté.
// ----------------------------------------------------------------------------
// Information d'AFFICHAGE / UX uniquement : premiumAccess et isAdmin sont
// calculés ici pour que le frontend puisse adapter son interface (ex.
// révéler l'onglet Admin), mais ne sont JAMAIS la protection réelle — chaque
// route sensible revérifie elle-même côté backend (hasPremiumAccess /
// requireAdmin). La sécurité du document source reste assurée par les
// Firestore Security Rules (le frontend ne peut jamais l'écrire directement).
//
// Ne renvoie jamais les identifiants Stripe (même null) : uniquement les
// champs utiles à l'affichage.
// ============================================================================
app.get('/api/subscription/status', requireFirebaseAuth, async (req, res) => {
  console.log('[SUBSCRIPTION] status requested');
  console.log('[SUBSCRIPTION] user:', req.uid);
  try {
    const subscription = await getSubscription(req.uid);
    res.json({
      plan: subscription.plan,
      status: subscription.status,
      source: subscription.source,
      expiresAt: subscription.expiresAt,
      trialStart: subscription.trialStart,
      trialEnd: subscription.trialEnd,
      currentPeriodStart: subscription.currentPeriodStart,
      currentPeriodEnd: subscription.currentPeriodEnd,
      billingInterval: subscription.billingInterval,
      premiumAccess: hasPremiumAccess(subscription),
      isAdmin: ADMIN_UIDS.has(req.uid) || isAdmin(subscription)
    });
  } catch (e) {
    console.error('[SUBSCRIPTION] status route error', e);
    res.status(500).json({ error: "Impossible de récupérer le statut d'abonnement pour le moment." });
  }
});

// ============================================================================
// GET /api/subscription/limits — used/limit/remaining pour chaque feature du
// plan effectif de l'utilisateur.
// ----------------------------------------------------------------------------
// Destinée UNIQUEMENT à l'affichage / UX (ex. barre "3/20 recettes utilisées").
// Ce n'est jamais la protection réelle : chaque action sensible devra appeler
// canUseFeature() elle-même côté backend au moment de l'action (branchement
// prévu dans une étape ultérieure — voir rapport).
//
// Une seule lecture de la subscription, puis les lectures par feature en
// parallèle (Promise.all) plutôt qu'en série, pour éviter les allers-retours
// Firestore inutiles (voir §14 performance de la demande).
// ============================================================================
app.get('/api/subscription/limits', requireFirebaseAuth, async (req, res) => {
  try {
    const subscription = await getSubscription(req.uid);
    const effectivePlan = getEffectivePlan(subscription);
    const featureNames = Object.keys(FEATURE_LIMITS[effectivePlan]);

    const results = await Promise.all(
      featureNames.map(name => evaluateFeatureLimit(req.uid, effectivePlan, name))
    );

    const features = {};
    featureNames.forEach((name, i) => {
      const r = results[i];
      features[name] = { used: r.used, limit: r.limit, remaining: r.remaining };
    });

    res.json({ plan: effectivePlan, features });
  } catch (e) {
    console.error('[SUBSCRIPTION] limits route error', e);
    res.status(500).json({ error: "Impossible de récupérer les limites pour le moment." });
  }
});

// ============================================================================
// PROMPT 23 — GET /api/widget-data : couche de données pour les futurs
// widgets (iOS/Android natifs). Suit la convention déjà en place dans ce
// fichier (routes /api/... sans préfixe de version dans l'URL) plutôt que
// d'introduire /api/v1/... isolément — le champ "v": 1 dans la réponse
// JSON permet déjà de faire évoluer le format plus tard sans casser un
// widget déjà installé (il peut vérifier ce champ avant de parser).
// Authentification IDENTIQUE à toutes les autres routes : requireFirebaseAuth
// vérifie le token Firebase et fournit req.uid — un widget ne peut donc
// jamais demander les données d'un autre utilisateur en changeant un
// paramètre, exactement comme le reste de l'API (voir §17 du prompt).
// ============================================================================
app.get('/api/widget-data', requireFirebaseAuth, async (req, res) => {
  try {
    const data = await buildWidgetData(req.uid);
    res.json(data);
  } catch (e) {
    console.error('[WIDGET-DATA] route error', e);
    res.status(500).json({ error: "Impossible de récupérer les données pour le moment." });
  }
});

// ============================================================================
// Administration — offrir/retirer Premium à un utilisateur.
// ----------------------------------------------------------------------------
// Toutes les routes /api/admin/* sont protégées par requireFirebaseAuth PUIS
// requireAdmin : un utilisateur normal qui appelle ces routes (même en
// construisant la requête à la main) reçoit systématiquement 403, jamais un
// succès. Rien côté frontend ne peut jamais accorder Premium à qui que ce
// soit — seul ce backend écrit le document subscription pour cette action.
// ============================================================================
const GIFT_DURATIONS = {
  '7d': 7 * 24 * 60 * 60 * 1000,
  '30d': 30 * 24 * 60 * 60 * 1000,
  '3m': 90 * 24 * 60 * 60 * 1000, // ~3 mois (pas de calendrier mensuel exact ici, cohérent avec "30d")
  '1y': 365 * 24 * 60 * 60 * 1000,
  permanent: null
};

// GET /api/admin/users — liste des comptes (email via Firebase Auth, déjà le
// système d'auth existant — pas de deuxième annuaire créé) + leur
// subscription actuelle. Usage personnel/petite échelle : une seule page de
// jusqu'à 1000 comptes Firebase Auth, largement suffisant ici.
app.get('/api/admin/users', requireFirebaseAuth, requireAdmin, async (req, res) => {
  try {
    const listResult = await admin.auth().listUsers(1000);
    const authUsers = listResult.users;
    const subscriptions = await Promise.all(authUsers.map(u => getSubscription(u.uid)));
    const users = authUsers.map((u, i) => {
      const s = subscriptions[i];
      return {
        uid: u.uid,
        email: u.email || null,
        plan: s.plan,
        status: s.status,
        source: s.source,
        expiresAt: s.expiresAt
      };
    });
    res.json({ users });
  } catch (e) {
    console.error('[ADMIN] Liste des utilisateurs échouée', e);
    res.status(500).json({ error: "Impossible de récupérer la liste des utilisateurs." });
  }
});

// POST /api/admin/users/:userId/grant-premium  body: { duration }
app.post('/api/admin/users/:userId/grant-premium', requireFirebaseAuth, requireAdmin, async (req, res) => {
  const { userId } = req.params;
  const { duration } = req.body || {};
  if (!Object.prototype.hasOwnProperty.call(GIFT_DURATIONS, duration)) {
    return res.status(400).json({ error: "Durée invalide. Attendu : 7d, 30d, 3m, 1y ou permanent." });
  }
  try {
    const ms = GIFT_DURATIONS[duration];
    const expiresAt = ms == null ? null : new Date(Date.now() + ms).toISOString();
    const update = { plan: 'premium', status: 'active', source: 'gift', expiresAt, updatedAt: Date.now() };
    // merge:true : ne touche pas trialStart/trialEnd/stripeCustomerId/etc. déjà
    // posés à l'étape 2, évite de "recréer" tout le document à chaque don.
    await fsDb().collection('users').doc(userId).collection('subscription').doc('current').set(update, { merge: true });
    console.log(`[SUBSCRIPTION] ADMIN ${req.uid} granted PREMIUM to ${userId} (${duration})`);
    res.json({ success: true, subscription: { ...DEFAULT_SUBSCRIPTION, ...update } });
  } catch (e) {
    console.error('[SUBSCRIPTION] grant-premium échoué', e);
    res.status(500).json({ error: "Impossible d'offrir Premium pour le moment." });
  }
});

// POST /api/admin/users/:userId/revoke-premium
app.post('/api/admin/users/:userId/revoke-premium', requireFirebaseAuth, requireAdmin, async (req, res) => {
  const { userId } = req.params;
  try {
    const update = { plan: 'free', status: 'active', source: 'none', expiresAt: null, updatedAt: Date.now() };
    await fsDb().collection('users').doc(userId).collection('subscription').doc('current').set(update, { merge: true });
    console.log(`[SUBSCRIPTION] ADMIN ${req.uid} revoked PREMIUM from ${userId}`);
    res.json({ success: true, subscription: { ...DEFAULT_SUBSCRIPTION, ...update } });
  } catch (e) {
    console.error('[SUBSCRIPTION] revoke-premium échoué', e);
    res.status(500).json({ error: "Impossible de retirer Premium pour le moment." });
  }
});

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
    // Limite FREE/PREMIUM/ADMIN — même mécanisme que /api/assistant-chat :
    // FEATURE_LIMITS.mealPhotoAnalysis existait déjà mais n'était pas encore
    // branché à cette route (aucune vérification ne s'y opposait auparavant).
    const access = await canUseFeature(req.uid, 'mealPhotoAnalysis');
    if (!access.allowed) {
      console.log(`[SUBSCRIPTION] mealPhotoAnalysis limit reached for ${req.uid} (plan=${access.plan})`);
      return res.status(403).json({
        error: "Tu as atteint la limite d'analyses de photos de repas pour ton plan actuel.",
        reason: access.reason,
        plan: access.plan,
        limit: access.limit,
        remaining: access.remaining
      });
    }

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

    // Incrémenté seulement APRÈS une analyse réussie, comme pour assistantChat :
    // un appel qui a échoué (503, timeout, photo illisible...) n'est pas compté
    // contre le quota mensuel de l'utilisateur.
    await incrementUsageCount(req.uid, 'mealPhotoAnalysis', FEATURE_LIMITS.free.mealPhotoAnalysis.window);

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
// Meals Tracker — analyse IA d'une description texte de repas.
// ----------------------------------------------------------------------------
// Alternative à /api/analyze-meal-photo ci-dessus, pour les cas où l'utilisateur
// préfère décrire son repas plutôt que d'en prendre une photo. Réutilise
// VOLONTAIREMENT :
//   - le même quota que l'analyse photo (FEATURE_LIMITS.mealPhotoAnalysis) —
//     ce n'est pas une feature séparée du point de vue du plan/abonnement,
//     juste une autre façon d'alimenter la même fonctionnalité "Analyse IA
//     Meals Tracker" ; pas de nouveau compteur, pas de nouvelle clé Firestore.
//   - le même format de sortie `items` que /api/analyze-meal-photo (name,
//     quantity, calories, protein, carbs, fat, confidence), pour que le
//     frontend puisse réutiliser telle quelle toute la UI de vérification/
//     correction déjà écrite pour l'analyse photo (renderMealAiResults,
//     confirmAddMealAiResults, etc. — voir js/05-meals-weight-steps-settings.js).
// Deux champs en plus de ce format commun, propres à l'analyse texte :
//   - needsClarification / clarificationQuestion : la description peut être
//     trop vague pour être exploitée (ex. "j'ai mangé quelque chose") ; dans
//     ce cas l'IA ne doit PAS inventer un repas, elle doit demander une
//     précision. Le frontend renvoie alors une description enrichie
//     (originale + question + réponse de l'utilisateur) sur un nouvel appel.
//   - totalFiber/totalSugar/totalSaturatedFat/totalSodium/estimationNote :
//     informations supplémentaires demandées, PUREMENT informatives — elles ne
//     sont pas persistées dans mealsData.entries (qui ne connaît que calories/
//     protein/carbs/fat), volontairement, pour ne pas créer une structure de
//     sauvegarde parallèle à celle déjà utilisée par tout le reste du Meals
//     Tracker (voir rapport).
// ============================================================================
app.post('/api/analyze-meal-text', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    // Même feature ("mealPhotoAnalysis") et donc même quota FREE/PREMIUM/ADMIN
    // que l'analyse photo — voir commentaire ci-dessus.
    const access = await canUseFeature(req.uid, 'mealPhotoAnalysis');
    if (!access.allowed) {
      console.log(`[SUBSCRIPTION] mealPhotoAnalysis (texte) limit reached for ${req.uid} (plan=${access.plan})`);
      return res.status(403).json({
        error: "Tu as atteint la limite d'analyses de repas pour ton plan actuel.",
        reason: access.reason,
        plan: access.plan,
        limit: access.limit,
        remaining: access.remaining
      });
    }

    const { description } = req.body || {};
    if (!description || typeof description !== 'string' || !description.trim()) {
      return res.status(400).json({ error: "Décris ce que tu as mangé avant de lancer l'analyse." });
    }
    const trimmed = description.trim();
    if (trimmed.length < 3) {
      return res.status(400).json({ error: "Ta description est trop courte pour être analysée. Ajoute un peu plus de détails." });
    }
    // Même logique de garde-fou que /api/generate-recipe (coût/latence Gemini),
    // avec une marge plus large car une description de repas + une éventuelle
    // précision de clarification peuvent être plus longues qu'une liste d'ingrédients.
    if (trimmed.length > 1500) {
      return res.status(400).json({ error: "Ta description est trop longue (1500 caractères max)." });
    }

    const response = await withTimeout(
      generateContentWithRetry({
        model: 'gemini-3.5-flash-lite',
        contents:
          `Description du repas rédigée par l'utilisateur : "${trimmed}"\n\n` +
          "Analyse cette description et identifie chaque aliment mentionné (nom, quantité/portion, " +
          "méthode de préparation si indiquée). Pour chaque aliment, fournis une ESTIMATION " +
          "OBLIGATOIRE de ses calories (kcal), protéines, glucides et lipides (en grammes), basée " +
          "sur des valeurs nutritionnelles typiques. Si l'utilisateur n'a pas donné de quantité " +
          "précise, estime une portion standard raisonnable et indique 'low' comme confidence pour " +
          "cet aliment (ne prétends jamais connaître une quantité non fournie). Fournis aussi, si " +
          "elles sont raisonnablement estimables à partir des aliments identifiés, les fibres, sucres, " +
          "graisses saturées et sodium TOTAUX du repas (sinon laisse ces champs à 0). " +
          "Si la description est vraiment trop vague pour identifier ne serait-ce qu'un aliment " +
          "(ex. \"j'ai mangé quelque chose\", \"un truc rapide\"), NE FOURNIS AUCUN item : mets " +
          "needsClarification à true et pose UNE question courte et précise pour obtenir " +
          "l'information manquante. Si tu peux estimer même approximativement à partir de ce qui " +
          "est écrit, ne demande PAS de clarification — fournis un résultat avec confidence 'low' " +
          "plutôt que de bloquer l'utilisateur.",
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              needsClarification: { type: Type.BOOLEAN },
              clarificationQuestion: { type: Type.STRING },
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
                  required: ["name", "quantity", "calories", "protein", "carbs", "fat", "confidence"]
                }
              },
              totalFiber: { type: Type.NUMBER },
              totalSugar: { type: Type.NUMBER },
              totalSaturatedFat: { type: Type.NUMBER },
              totalSodium: { type: Type.NUMBER },
              estimationNote: { type: Type.STRING }
            },
            required: ["needsClarification", "items"]
          }
        }
      }),
      25000,
      "L'analyse prend trop de temps. Vérifie ta connexion et réessaie."
    );

    // Même contrôle que /api/analyze-meal-photo : réponse bloquée par le filtre
    // de sécurité ou aucun candidat exploitable -> response.text vide/undefined.
    const blockReason = response.promptFeedback?.blockReason;
    const finishReason = response.candidates?.[0]?.finishReason;
    if (!response.text) {
      console.error('analyze-meal-text: réponse vide de Gemini', { blockReason, finishReason });
      const msg = blockReason || finishReason === 'SAFETY'
        ? "Cette description n'a pas pu être analysée (filtre de sécurité). Reformule-la."
        : "L'IA n'a pas pu analyser cette description. Réessaie en la reformulant.";
      return res.status(422).json({ error: msg });
    }

    let parsed;
    try {
      const cleanText = response.text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
      parsed = JSON.parse(cleanText);
    } catch (parseErr) {
      console.error('analyze-meal-text: réponse Gemini non-JSON', response.text, parseErr);
      return res.status(502).json({ error: "L'IA a renvoyé une réponse inattendue. Réessaie." });
    }

    const needsClarification = parsed?.needsClarification === true;
    // Un item sans nom exploitable est ignoré, comme pour l'analyse photo.
    const items = Array.isArray(parsed?.items)
      ? parsed.items.filter(it => it && typeof it === 'object' && typeof it.name === 'string' && it.name.trim())
      : [];

    if (!needsClarification && !items.length) {
      // Ni clarification demandée, ni aliment identifié : réponse inexploitable,
      // à traiter comme une absence de données plutôt que comme un repas vide.
      return res.status(422).json({ error: "Aucun aliment n'a pu être identifié dans cette description. Essaie d'être plus précis (aliments, quantités)." });
    }

    // Incrémenté après toute réponse exploitable de Gemini (y compris une
    // demande de clarification), comme /api/analyze-meal-photo : seul un appel
    // qui a réellement échoué (503, timeout, JSON invalide...) n'est pas compté.
    await incrementUsageCount(req.uid, 'mealPhotoAnalysis', FEATURE_LIMITS.free.mealPhotoAnalysis.window);

    res.json({
      success: true,
      needsClarification,
      clarificationQuestion: needsClarification ? (parsed.clarificationQuestion || "Peux-tu préciser ce que tu as mangé ?") : null,
      items,
      totalFiber: Number(parsed?.totalFiber) || 0,
      totalSugar: Number(parsed?.totalSugar) || 0,
      totalSaturatedFat: Number(parsed?.totalSaturatedFat) || 0,
      totalSodium: Number(parsed?.totalSodium) || 0,
      estimationNote: typeof parsed?.estimationNote === 'string' ? parsed.estimationNote : ''
    });
  } catch (error) {
    if (error?.isTimeout) {
      console.error('analyze-meal-text: timeout dépassé', error.message);
      return res.status(504).json({ error: error.message });
    }
    const status = error?.error?.status || error?.status;
    if (status === 'UNAVAILABLE' || error?.error?.code === 503) {
      console.error('analyze-meal-text: Gemini indisponible après plusieurs tentatives', error);
      return res.status(503).json({ error: "L'IA est temporairement surchargée. Réessaie dans quelques instants." });
    }
    if (status === 'RESOURCE_EXHAUSTED' || error?.error?.code === 429) {
      console.error('analyze-meal-text: quota Gemini dépassé', error);
      return res.status(429).json({ error: "Le quota d'analyses IA est temporairement dépassé. Réessaie plus tard." });
    }
    console.error('analyze-meal-text error', error);
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
//
// Limite FREE/PREMIUM/ADMIN branchée à cette étape (seule cette route IA est
// concernée pour l'instant — voir rapport) : réutilise le système de limites
// déjà posé à l'étape 3 (FEATURE_LIMITS.assistantChat + canUseFeature), lui-
// même dérivé de hasPremiumAccess()/getEffectivePlan(). L'aiRateLimiter
// existant (30 appels IA / 15 min, tous endpoints confondus) reste EN PLUS,
// inchangé : c'est un garde-fou anti-abus général, orthogonal au quota par
// plan ci-dessous, qui protège spécifiquement contre un coût IA excessif
// pour un compte Premium offert.
// ============================================================================
app.post('/api/assistant-chat', requireFirebaseAuth, aiRateLimiter, async (req, res) => {
  try {
    const { message, context } = req.body || {};
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: "Message manquant" });
    }

    const access = await canUseFeature(req.uid, 'assistantChat');
    if (!access.allowed) {
      console.log(`[SUBSCRIPTION] assistantChat limit reached for ${req.uid} (plan=${access.plan})`);
      return res.status(403).json({
        error: "Tu as atteint la limite de demandes à l'Assistant pour ton plan actuel.",
        reason: access.reason,
        plan: access.plan,
        limit: access.limit,
        remaining: access.remaining
      });
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

    // Incrémenté seulement APRÈS un appel Gemini réussi : un appel qui a
    // échoué (503, timeout...) n'est pas compté contre le quota de l'utilisateur.
    await incrementUsageCount(req.uid, 'assistantChat', FEATURE_LIMITS.free.assistantChat.window);

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
    // Limite FREE/PREMIUM/ADMIN — FEATURE_LIMITS.assistantVoice existait déjà
    // mais n'était pas encore branché à cette route, seule assistantChat
    // l'était. La commande vocale complète (transcription + réponse) consomme
    // ensuite aussi le quota assistantChat au moment de l'appel à
    // /api/assistant-chat qui suit côté frontend — les deux quotas restent
    // indépendants, comme prévu par FEATURE_LIMITS.
    const access = await canUseFeature(req.uid, 'assistantVoice');
    if (!access.allowed) {
      console.log(`[SUBSCRIPTION] assistantVoice limit reached for ${req.uid} (plan=${access.plan})`);
      return res.status(403).json({
        error: "Tu as atteint la limite de commandes vocales pour ton plan actuel.",
        reason: access.reason,
        plan: access.plan,
        limit: access.limit,
        remaining: access.remaining
      });
    }

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

    // Incrémenté seulement après une transcription réussie (même logique que
    // mealPhotoAnalysis/assistantChat) — un texte vide (audio silencieux) est
    // un résultat "réussi" du point de vue de l'appel IA et reste compté,
    // comme pour toutes les autres routes de ce fichier.
    await incrementUsageCount(req.uid, 'assistantVoice', FEATURE_LIMITS.free.assistantVoice.window);

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

// ============================================================================
// Complément Prompt 13 — Notifications programmées à heure fixe (FCM)
// ----------------------------------------------------------------------------
// Réutilise scrupuleusement ce qui existe déjà : mêmes 5 catégories et mêmes
// conditions métier ("SI") que NOTIFICATION_CATEGORIES côté frontend (voir
// js/05-meals-weight-steps-settings.js) — seul le "QUAND" change ici (heures
// fixes au lieu d'une fenêtre). Ce backend ne fait que LIRE les mêmes clés
// Firestore que dbGet/dbSet écrit déjà (users/{uid}/data/{clé}) — il n'écrit
// jamais ces documents, sauf mrp-fcm-devices pour retirer un token invalide
// (section 11 du Prompt 13) via la Firestore Security Rule déjà en place pour
// ce chemin (accès Admin SDK, hors règles client de toute façon).
//
// Horaires fixes (heure LOCALE de chaque utilisateur, voir mrp-timezone) :
//   Life Goals    : 08:00
//   To Do         : 10:30, 14:30
//   Meals Tracker : 07:30, 12:30, 19:00
//   Gym / Muscu   : 18:30, 21:30
//   Steps         : 20:00
// ============================================================================
const DEFAULT_TODO_LIST_ID = 'default';
const DASHBOARD_MEAL_SLOTS = ['Petit-déjeuner', 'Déjeuner', 'Dîner'];
// Fuseau de secours UNIQUEMENT pour un utilisateur qui ne s'est pas encore
// reconnecté depuis la mise à jour qui détecte son fuseau (voir mrp-timezone,
// js/05-...) — jamais imposé si un fuseau réel est déjà enregistré (section 17).
const FALLBACK_TIMEZONE = 'Europe/Zurich';

const NOTIFICATION_SCHEDULE = [
  {
    id: 'meals', tab: 'meals', settingsFlag: 'mealsReminder', title: '🍽️ Meals',
    times: ['07:30', '12:30', '19:00'],
    isRelevant: (d) => (d.meals.total - d.meals.loggedTypes.size) > 0,
    build: (d) => {
      const missing = d.meals.total - d.meals.loggedTypes.size;
      return missing === 1
        ? `Tu n'as pas encore enregistré ton ${(d.meals.nextSlot || '').toLowerCase()}.`
        : `Tu as ${missing} repas à enregistrer aujourd'hui.`;
    }
  },
  {
    id: 'workout', tab: 'gym', settingsFlag: 'gymReminder', title: '🏋️ Workout',
    times: ['18:30', '21:30'],
    isRelevant: (d) => d.gym.total > 0 && d.gym.done < d.gym.total,
    build: (d) => {
      const pending = d.gym.sessions.filter(s => s && !s.checked);
      const names = pending.map(s => s.name).filter(Boolean);
      return names.length > 1
        ? `Tu as ${names.length} séances encore prévues aujourd'hui.`
        : `${names[0] || 'Ta séance'} t'attend aujourd'hui 💪`;
    }
  },
  {
    id: 'todos', tab: 'todo', settingsFlag: 'todosReminder', title: '✅ Tâches',
    times: ['10:30', '14:30'],
    isRelevant: (d) => d.todos.pending.length > 0,
    build: (d) => {
      const pending = d.todos.pending;
      if (pending.length === 1) return `Il te reste une tâche à terminer : ${pending[0].text}`;
      const top = d.todos.top;
      const suffix = (top && top.priority === 'Haute') ? ' Une tâche importante t\'attend en priorité.' : '';
      return `Tu as ${pending.length} tâches à terminer aujourd'hui.${suffix}`;
    }
  },
  {
    id: 'goals', tab: 'objectifs', settingsFlag: 'goalsReminder', title: '🎯 Life Goal',
    times: ['08:00'],
    isRelevant: (d) => !!d.goals.focus && (d.goals.focus.progress || 0) < 100,
    build: (d) => `N'oublie pas d'avancer sur ton objectif : ${d.goals.focus.title}`
  },
  {
    id: 'steps', tab: 'steps', settingsFlag: 'stepsReminder', title: '🚶 Steps',
    times: ['20:00'],
    isRelevant: (d) => d.steps.goal > 0 && d.steps.today < d.steps.goal,
    build: (d) => {
      const remaining = Math.max(0, d.steps.goal - d.steps.today);
      return `Plus que ${remaining.toLocaleString('fr-FR')} pas pour atteindre ton objectif.`;
    }
  }
];

// Heure locale + date locale (pour la déduplication) + jour de la semaine en
// français (pour indexer gymPlan, voir getTodayGymDay() dans js/04-gym.js) —
// tout calculé dans le fuseau IANA de l'utilisateur, jamais celui du serveur.
const WEEKDAY_EN_TO_FR = { sunday: 'dimanche', monday: 'lundi', tuesday: 'mardi', wednesday: 'mercredi', thursday: 'jeudi', friday: 'vendredi', saturday: 'samedi' };
function getLocalContext(timezone) {
  const tz = timezone || FALLBACK_TIMEZONE;
  const now = new Date();
  const formatterOptions = {
    hour12: false, year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', weekday: 'long'
  };
  let parts;
  try {
    parts = new Intl.DateTimeFormat('en-CA', { ...formatterOptions, timeZone: tz }).formatToParts(now);
  } catch (e) {
    // Fuseau invalide/corrompu pour cet utilisateur : repli sur le fuseau par
    // défaut plutôt que de faire échouer tout le cycle pour les autres utilisateurs.
    parts = new Intl.DateTimeFormat('en-CA', { ...formatterOptions, timeZone: FALLBACK_TIMEZONE }).formatToParts(now);
  }
  const get = (type) => parts.find(p => p.type === type)?.value;
  return {
    dateIso: `${get('year')}-${get('month')}-${get('day')}`,
    hhmm: `${get('hour')}:${get('minute')}`,
    weekdayFr: WEEKDAY_EN_TO_FR[(get('weekday') || '').toLowerCase()] || 'lundi'
  };
}

// Reconstruit, à partir des documents Firestore bruts (users/{uid}/data/{clé}),
// EXACTEMENT le même instantané que DashboardDataService() côté frontend (voir
// 01-core-home-goals-todos.js) — pour que "SI" une notification est pertinente
// reste identique, que la décision soit prise par le navigateur (onglet ouvert)
// ou par ce backend (app fermée).
async function buildServerDashboardData(uid, dateIso, weekdayFr) {
  const dataCol = fsDb().collection('users').doc(uid).collection('data');
  const [gymSnap, todosSnap, goalsSnap, activeGoalSnap, mealsSnap, stepsSnap] = await Promise.all([
    dataCol.doc('mrp-gym').get(),
    dataCol.doc('mrp-todos').get(),
    dataCol.doc('mrp-goals').get(),
    dataCol.doc('mrp-active-goal').get(),
    dataCol.doc('mrp-meals').get(),
    dataCol.doc('mrp-steps').get()
  ]);

  const gymPlan = gymSnap.exists ? (gymSnap.data().value || {}) : {};
  const gymToday = Array.isArray(gymPlan[weekdayFr]) ? gymPlan[weekdayFr] : [];
  const gymDone = gymToday.filter(s => s && s.checked).length;

  const todos = todosSnap.exists ? (todosSnap.data().value || []) : [];
  const todayListTodos = (Array.isArray(todos) ? todos : []).filter(t => t && t.listId === DEFAULT_TODO_LIST_ID);
  const pendingTodos = todayListTodos.filter(t => !t.done);
  const priorityOrder = { 'Haute': 0, 'Normale': 1, 'Basse': 2 };
  const topTodo = [...pendingTodos].sort((a, b) => (priorityOrder[a.priority] ?? 1) - (priorityOrder[b.priority] ?? 1))[0] || null;

  const goals = goalsSnap.exists ? (goalsSnap.data().value || []) : [];
  const activeGoalId = activeGoalSnap.exists ? activeGoalSnap.data().value : null;
  const focusGoal = activeGoalId ? ((Array.isArray(goals) ? goals : []).find(g => g && g.id === activeGoalId) || null) : null;

  const mealsData = mealsSnap.exists ? (mealsSnap.data().value || { entries: [] }) : { entries: [] };
  const todayMealEntries = (mealsData.entries || []).filter(e => e && e.date === dateIso);
  const loggedMealTypes = new Set(todayMealEntries.map(e => e.type));
  const nextMealSlot = DASHBOARD_MEAL_SLOTS.find(slot => !loggedMealTypes.has(slot)) || null;

  const stepsData = stepsSnap.exists ? (stepsSnap.data().value || { entries: [], goal: 0 }) : { entries: [], goal: 0 };
  const todaySteps = ((stepsData.entries || []).find(e => e && e.date === dateIso) || {}).steps || 0;

  return {
    gym: { sessions: gymToday, done: gymDone, total: gymToday.length },
    todos: { pending: pendingTodos, total: todayListTodos.length, top: topTodo },
    goals: { focus: focusGoal },
    meals: { loggedTypes: loggedMealTypes, total: DASHBOARD_MEAL_SLOTS.length, nextSlot: nextMealSlot },
    steps: { today: todaySteps, goal: stepsData.goal || 0 }
  };
}

// ============================================================================
// PROMPT 23 — Couche de données pour les futurs widgets (iOS/Android natifs).
// ----------------------------------------------------------------------------
// buildWidgetData() ne RECALCULE JAMAIS une donnée qui existe déjà ailleurs :
// elle appelle buildServerDashboardData() (la même fonction déjà utilisée par
// le moteur de notifications, voir runScheduledNotificationsTick ci-dessous)
// et se contente d'AJOUTER les quelques champs qui manquent pour un widget
// (score du jour, macros repas, prochain voyage) — jamais une deuxième
// source de vérité. computeDailyScoreServer() ci-dessous est un PORT
// EXACT, ligne à ligne, de computeDailyScore() côté frontend
// (js/01-core-home-goals-todos.js) : même formule, mêmes données en entrée
// (todos.done est dérivé de total - pending.length, exactement comme le fait
// déjà DashboardDataService() côté frontend) → même résultat que
// l'application, jamais un score différent.
//
// Volontairement HORS PÉRIMÈTRE de cette première version (voir rapport) :
// le "prochain événement" du calendrier (Life Planner). Le calculer
// correctement nécessite de porter ici le moteur de récurrence du calendrier
// (séries, exceptions, décalages) déjà présent côté frontend
// (js/08-calendar.js) — une vraie duplication de logique, pas un simple ajout
// de champ. Conformément à la consigne ("ne recrée pas un deuxième calcul
// différent"), je ne l'ai pas approximé : mieux vaut l'omettre que risquer un
// résultat différent de l'application. À faire dans une étape dédiée si vous
// le souhaitez.
// ============================================================================
// PROMPT 26 — Port du moteur de récurrence du calendrier (js/08-calendar.js)
// pour permettre à planner.nextEvent d'exister enfin dans /api/widget-data.
// ----------------------------------------------------------------------------
// Ce qui suit, jusqu'à getNextEventServer(), est un PORT LIGNE À LIGNE des
// fonctions calParseISO / calAddDays / calDiffDays / calAddMonths /
// calMondayOf / calWeekdayMonday0 / calStepsToReach / calExpandRecurrenceDates
// / calRecurrenceSpanDays / calOccurrenceEndDate de js/08-calendar.js — MÊMES
// NOMS, MÊME CORPS, volontairement, pour que toute correction future du moteur
// de récurrence côté frontend puisse être reportée ici par simple copier-coller
// plutôt que réécrite. Ne PAS diverger de ces fonctions sans reporter le même
// changement des deux côtés (même avertissement que pour
// computeDailyScoreServer, voir plus bas dans ce fichier).
// ============================================================================
function calParseISO(iso){
  const [y,m,d] = iso.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}
function calFormatISO(date){
  return date.toISOString().slice(0,10);
}
function calAddDays(iso, n){
  const d = calParseISO(iso);
  d.setUTCDate(d.getUTCDate() + n);
  return calFormatISO(d);
}
function calDiffDays(aIso, bIso){
  return Math.round((calParseISO(bIso) - calParseISO(aIso)) / 86400000);
}
function calAddMonths(iso, n){
  const d = calParseISO(iso);
  const day = d.getUTCDate();
  const target = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + n, 1));
  const lastDayOfTargetMonth = new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0)).getUTCDate();
  target.setUTCDate(Math.min(day, lastDayOfTargetMonth));
  return calFormatISO(target);
}
function calWeekdayMonday0(iso){
  const jsDay = calParseISO(iso).getUTCDay();
  return (jsDay + 6) % 7;
}
function calMondayOf(iso){
  return calAddDays(iso, -calWeekdayMonday0(iso));
}
function calRecurrenceSpanDays(ev){
  return calDiffDays(ev.date, ev.endDate || ev.date);
}
function calOccurrenceEndDate(startIso, spanDays){
  return spanDays > 0 ? calAddDays(startIso, spanDays) : startIso;
}
function calStepsToReach(freq, interval, fromIso, targetIso){
  if(targetIso <= fromIso) return 0;
  if(freq === 'daily'){
    return Math.floor(calDiffDays(fromIso, targetIso) / interval);
  }
  if(freq === 'weekly'){
    return Math.floor(calDiffDays(fromIso, targetIso) / (7 * interval));
  }
  if(freq === 'monthly' || freq === 'yearly'){
    const fD = calParseISO(fromIso), tD = calParseISO(targetIso);
    const monthsDiff = (tD.getUTCFullYear() - fD.getUTCFullYear()) * 12 + (tD.getUTCMonth() - fD.getUTCMonth());
    const unitMonths = freq === 'monthly' ? interval : interval * 12;
    return Math.max(0, Math.floor(monthsDiff / unitMonths) - 1);
  }
  return 0;
}
function calExpandRecurrenceDates(ev, rangeStart, rangeEnd){
  if(!ev.recurrence || !ev.recurrence.freq || ev.recurrence.freq === 'none'){
    const end = ev.endDate || ev.date;
    return (ev.date <= rangeEnd && end >= rangeStart) ? [ev.date] : [];
  }
  const { freq, interval = 1, byWeekday, until } = ev.recurrence;
  const exceptions = new Set(ev.recurrenceExceptions || []);
  const hardEnd = until && until < rangeEnd ? until : rangeEnd;
  const out = [];

  if(freq === 'weekly' && Array.isArray(byWeekday) && byWeekday.length){
    let blockMonday = calMondayOf(ev.date);
    const jump = calStepsToReach('weekly', interval, blockMonday, calMondayOf(rangeStart));
    if(jump > 0) blockMonday = calAddDays(blockMonday, 7 * interval * jump);
    let iterations = 0;
    while(blockMonday <= hardEnd && iterations < 730){
      for(const wd of byWeekday){
        const occ = calAddDays(blockMonday, wd);
        iterations++;
        if(occ < ev.date) continue;
        if(until && occ > until) continue;
        if(occ < rangeStart || occ > rangeEnd) continue;
        if(exceptions.has(occ)) continue;
        out.push(occ);
      }
      blockMonday = calAddDays(blockMonday, 7 * interval);
    }
    return out.sort();
  }

  const jump = calStepsToReach(freq, interval, ev.date, rangeStart);
  let cursor = ev.date;
  if(jump > 0){
    if(freq === 'daily') cursor = calAddDays(cursor, interval * jump);
    else if(freq === 'weekly') cursor = calAddDays(cursor, 7 * interval * jump);
    else if(freq === 'monthly') cursor = calAddMonths(cursor, interval * jump);
    else if(freq === 'yearly') cursor = calAddMonths(cursor, 12 * interval * jump);
  }

  let iterations = 0;
  while(cursor <= hardEnd && iterations < 730){
    iterations++;
    if(until && cursor > until) break;
    if(cursor > rangeEnd) break;
    if(cursor >= rangeStart && !exceptions.has(cursor)) out.push(cursor);
    if(freq === 'daily') cursor = calAddDays(cursor, interval);
    else if(freq === 'weekly') cursor = calAddDays(cursor, 7 * interval);
    else if(freq === 'monthly') cursor = calAddMonths(cursor, interval);
    else if(freq === 'yearly') cursor = calAddMonths(cursor, 12 * interval);
    else break;
  }
  return out;
}

// getNextEventServer() N'EST PAS un port direct d'une fonction frontend — le
// frontend n'a jamais eu besoin de "la toute prochaine occurrence, tous types
// confondus" (Calendar affiche un jour/une semaine/un mois entier, jamais une
// seule prochaine occurrence isolée). Cette fonction est donc nouvelle, mais
// elle n'introduit AUCUN nouveau calcul de date d'occurrence : elle réutilise
// exclusivement calExpandRecurrenceDates()/calRecurrenceSpanDays() ci-dessus
// (les fonctions, elles, sont bien portées à l'identique) pour générer les
// mêmes dates que Calendar afficherait, puis choisit la plus proche dans le
// futur. Inclut les Events autonomes (récurrents ou non) ET les Tasks avec
// horaire (startTime+endTime) — exactement le même périmètre que
// calGetDayOccurrences() côté frontend (voir le commentaire au-dessus de
// cette fonction dans js/08-calendar.js).
function getNextEventServer(calendarEvents, todos, dateIso, hhmm){
  const HORIZON_DAYS = 60; // au-delà, aucun widget n'a besoin d'annoncer un événement si lointain
  const rangeEnd = calAddDays(dateIso, HORIZON_DAYS);
  const candidates = [];

  (calendarEvents || []).forEach(ev => {
    if(!ev || !ev.date) return;
    const span = calRecurrenceSpanDays(ev);
    calExpandRecurrenceDates(ev, dateIso, rangeEnd).forEach(occDate => {
      candidates.push({
        date: occDate, endDate: calOccurrenceEndDate(occDate, span),
        title: ev.title, startTime: ev.startTime || '', allDay: !!ev.allDay
      });
    });
  });

  (todos || []).forEach(t => {
    if(t && t.dueDate && t.dueDate >= dateIso && t.dueDate <= rangeEnd && t.startTime && t.endTime){
      candidates.push({ date: t.dueDate, endDate: t.dueDate, title: t.text, startTime: t.startTime, allDay: false });
    }
  });

  // Ne garde que ce qui n'a pas encore commencé : tout jour strictement futur,
  // ou aujourd'hui si "toute la journée" ou si l'heure de début n'est pas
  // encore passée (comparaison directe de chaînes "HH:MM", valide car format
  // fixe zero-paddé, déjà utilisé ailleurs dans le projet de la même façon).
  const upcoming = candidates.filter(c => {
    if(c.date > dateIso) return true;
    if(c.date < dateIso) return false;
    if(c.allDay) return true;
    return !c.startTime || c.startTime >= hhmm;
  });

  upcoming.sort((a, b) => {
    if(a.date !== b.date) return a.date < b.date ? -1 : 1;
    const at = a.allDay ? '' : (a.startTime || '');
    const bt = b.allDay ? '' : (b.startTime || '');
    return at.localeCompare(bt);
  });

  const next = upcoming[0];
  if(!next) return null;
  return {
    title: next.title, start: next.date, startTime: next.allDay ? null : (next.startTime || null), allDay: next.allDay
  };
}

async function buildWidgetData(uid) {
  const dataCol = fsDb().collection('users').doc(uid).collection('data');

  // Fuseau horaire de l'utilisateur — mêmes clé et repli que les notifications
  // (mrp-timezone / FALLBACK_TIMEZONE via getLocalContext), pas une nouvelle logique.
  let timezone = null;
  try {
    const tzSnap = await dataCol.doc('mrp-timezone').get();
    timezone = tzSnap.exists ? tzSnap.data().value : null;
  } catch (e) {
    console.error('[WIDGET-DATA] lecture du fuseau horaire échouée pour', uid, e);
  }
  const { dateIso, hhmm, weekdayFr } = getLocalContext(timezone);

  // PROMPT 24 : chaque section est récupérée indépendamment et protégée par son
  // propre try/catch (§18-19 du prompt — "une erreur de récupération d'une
  // donnée ne doit pas casser tout le système"). Si une section échoue, elle
  // est simplement absente de la réponse (valeur null / non incluse) et son nom
  // apparaît dans `errors` — le reste de la réponse reste utilisable. Aucune
  // des 4 lectures ci-dessous ne dépend d'une autre : une erreur Garmin, par
  // exemple, ne peut jamais empêcher tasks/workout/goal/meals/score de revenir.
  const errors = [];

  // 1) Source principale — RÉUTILISE buildServerDashboardData() telle quelle
  //    (déjà utilisée par les notifications, voir runScheduledNotificationsTick
  //    plus haut) : gym, todos (Today's List uniquement), objectif actif,
  //    statut des repas loggés, pas. Aucune deuxième lecture, aucun deuxième
  //    calcul pour ces domaines.
  let d = null;
  try {
    d = await buildServerDashboardData(uid, dateIso, weekdayFr);
  } catch (e) {
    console.error('[WIDGET-DATA] buildServerDashboardData a échoué pour', uid, e);
    errors.push('core');
  }

  // 2) Macros du jour — même calcul que renderMeals() côté frontend
  //    (js/05-meals-weight-steps-settings.js) : somme des entrées du jour,
  //    objectifs Meals Tracker (mêmes valeurs par défaut MEALS_DEFAULT_GOALS
  //    que le frontend si l'utilisateur n'a jamais ouvert Settings — avant ce
  //    correctif, un repli différent [{}] pouvait afficher un objectif "0" au
  //    lieu de la vraie valeur par défaut de l'app).
  let mealTotals = null, mealGoals = null;
  try {
    const mealsSnap = await dataCol.doc('mrp-meals').get();
    const mealsData = mealsSnap.exists ? (mealsSnap.data().value || {}) : {};
    const todayMealEntries = (mealsData.entries || []).filter(e => e && e.date === dateIso);
    mealTotals = todayMealEntries.reduce((acc, e) => {
      acc.calories += e.calories || 0; acc.protein += e.protein || 0;
      acc.carbs += e.carbs || 0; acc.fat += e.fat || 0;
      return acc;
    }, { calories: 0, protein: 0, carbs: 0, fat: 0 });
    mealGoals = { ...MEALS_DEFAULT_GOALS, ...(mealsData.goals || {}) };
  } catch (e) {
    console.error('[WIDGET-DATA] lecture des repas échouée pour', uid, e);
    errors.push('meals');
  }

  // 3) Prochain voyage — logique simple (pas de récurrence, contrairement au
  //    calendrier) : premier voyage dont la date de fin n'est pas déjà passée,
  //    trié par date de début. Même donnée brute que l'onglet Voyage
  //    (mrp-voyages-advanced), juste triée/filtrée — aucun risque de diverger.
  let nextTrip = null;
  try {
    const tripsSnap = await dataCol.doc('mrp-voyages-advanced').get();
    const tripsState = tripsSnap.exists ? (tripsSnap.data().value || { trips: [] }) : { trips: [] };
    const upcomingTrips = (tripsState.trips || [])
      .filter(t => t && (t.endDate || t.startDate) && (t.endDate || t.startDate) >= dateIso)
      .sort((a, b) => (a.startDate || '').localeCompare(b.startDate || ''));
    nextTrip = upcomingTrips[0] || null;
  } catch (e) {
    console.error('[WIDGET-DATA] lecture des voyages échouée pour', uid, e);
    errors.push('travel');
  }

  // 4) Statut Garmin — RÉUTILISE la même collection que GET /api/garmin/status
  //    (jamais le token, uniquement connected/lastSync, déjà jugé non sensible
  //    lors de l'audit sécurité). RoutinePOV ne synchronise réellement QUE les
  //    pas depuis Garmin (voir webhook /api/garmin/webhook, values.steps) — pas
  //    de calories ni de fréquence cardiaque dans le modèle de données actuel,
  //    donc pas de champ inventé pour ces valeurs (§12 : "ne crée aucune donnée
  //    simulée"). Les pas eux-mêmes restent dans `activity.steps` (déjà fournis
  //    par buildServerDashboardData, qu'ils viennent d'une saisie manuelle ou
  //    d'un sync Garmin — la donnée brute mrp-steps ne fait pas la différence
  //    au niveau du total du jour).
  let garmin = { connected: false, lastSync: null };
  try {
    const garminSnap = await fsDb().collection('garminConnections').doc(uid).get();
    garmin = garminSnap.exists
      ? { connected: true, lastSync: garminSnap.data().lastSyncAt || null }
      : { connected: false, lastSync: null };
  } catch (e) {
    console.error('[WIDGET-DATA] lecture du statut Garmin échouée pour', uid, e);
    errors.push('garmin');
  }

  // 5) Prochain événement (Life Planner) — PROMPT 26. Lecture indépendante de
  //    mrp-calendar-events + mrp-todos (todos complet, PAS le sous-ensemble
  //    Today's List déjà renvoyé par buildServerDashboardData — le calendrier
  //    doit voir aussi les tâches Long Term List avec horaire, exactement
  //    comme calGetDayOccurrences() côté frontend). Isolée comme les 4 autres
  //    sections : une panne ici n'affecte jamais tasks/workout/goal/meals/score.
  let nextEvent = null;
  try {
    const [calSnap, todosSnap] = await Promise.all([
      dataCol.doc('mrp-calendar-events').get(),
      dataCol.doc('mrp-todos').get()
    ]);
    const calendarEventsList = calSnap.exists ? (calSnap.data().value || []) : [];
    const todosList = todosSnap.exists ? (todosSnap.data().value || []) : [];
    nextEvent = getNextEventServer(calendarEventsList, todosList, dateIso, hhmm);
  } catch (e) {
    console.error('[WIDGET-DATA] calcul du prochain événement échoué pour', uid, e);
    errors.push('planner');
  }

  // Score du jour — port exact de computeDailyScore() (js/01-core-home-goals-todos.js),
  // voir computeDailyScoreServer ci-dessous. Uniquement si la source principale (d)
  // a pu être lue — sinon impossible à calculer, reste `null` (jamais un faux 0).
  let dailyScore = null;
  let todosDone = 0;
  if (d) {
    todosDone = d.todos.total - d.todos.pending.length;
    dailyScore = computeDailyScoreServer({
      gym: d.gym, todos: { ...d.todos, done: todosDone },
      goals: d.goals, meals: d.meals, steps: d.steps
    });
  }

  // Données MINIMALES uniquement (§7/§16/§24 du prompt) : jamais le profil
  // complet, jamais l'historique complet, jamais de token/secret/donnée Garmin
  // au-delà de connected/lastSync. Convention de nommage stable et groupée par
  // domaine (§13/§25/§26) — structure destinée à ne plus changer de noms de
  // champs une fois consommée par un vrai widget (seul `version` évoluera).
  //
  // IMPORTANT — null vs 0 (§16) : `null` signifie explicitement "aucune donnée
  // / non calculable" (ex. score si la source principale a échoué, `workout`/
  // `goal`/`travel.nextTrip` si rien n'est programmé). Un `0` numérique
  // (tasks.total, meals.calories, activity.steps.today...) est TOUJOURS une
  // vraie valeur observée (ex. "aucun repas loggé aujourd'hui" = 0 réel), au
  // même titre que l'application elle-même ne distingue pas ces cas pour ces
  // champs précis — donc le widget ne les invente pas non plus.
  return {
    version: 1,
    updatedAt: new Date().toISOString(),
    errors, // vide si tout a pu être lu ; sinon liste des sections indisponibles ce cycle-ci

    dailyScore, // number 0-100, identique à Home ; null seulement si `d` n'a pas pu être lu

    tasks: d ? {
      total: d.todos.total,
      completed: todosDone,
      remaining: d.todos.pending.length,
      // Sous-ensemble réel de Today's List (jamais Long Term List, même filtre
      // que DEFAULT_TODO_LIST_ID) — pas une nouvelle notion inventée : ce sont
      // les tâches en attente déjà marquées priorité "Haute" par l'utilisateur.
      priorityTasks: d.todos.pending.filter(t => t.priority === 'Haute').map(t => ({ title: t.text }))
    } : null,

    workout: d ? {
      // null si aucune séance n'est programmée aujourd'hui — jamais une séance
      // inventée (§7). `done`/`total` ne sont inclus qu'avec une vraie séance.
      next: d.gym.sessions[0] ? { title: d.gym.sessions[0].name, done: d.gym.done, total: d.gym.total } : null
    } : null,

    goal: d ? {
      current: d.goals.focus ? { id: d.goals.focus.id, title: d.goals.focus.title, progress: d.goals.focus.progress || 0 } : null
    } : null,

    meals: mealTotals ? {
      calories: Math.round(mealTotals.calories), caloriesGoal: mealGoals.calories,
      protein: Math.round(mealTotals.protein), proteinGoal: mealGoals.protein,
      carbs: Math.round(mealTotals.carbs), carbsGoal: mealGoals.carbs,
      fat: Math.round(mealTotals.fat), fatGoal: mealGoals.fat
    } : null,

    // Life Planner ("prochain événement") — PROMPT 26 : implémenté via
    // getNextEventServer() ci-dessus (port du moteur de récurrence). null si
    // aucun événement/tâche avec horaire dans les 60 prochains jours, ou si la
    // lecture a échoué (voir errors).
    planner: { nextEvent },

    travel: {
      nextTrip: nextTrip ? {
        title: nextTrip.name || null, destination: nextTrip.destination || null,
        start: nextTrip.startDate || null, end: nextTrip.endDate || null
      } : null
    },

    activity: {
      steps: d ? { today: d.steps.today, goal: d.steps.goal || null } : null,
      garmin
    }
  };
}

// Port exact de computeDailyScore() — js/01-core-home-goals-todos.js. Si la
// formule change un jour côté frontend, reporter EXACTEMENT le même
// changement ici (voir avertissement en tête de buildWidgetData ci-dessus).
// Comportement intentionnel hérité du frontend : renvoie 0 (pas null) quand
// aucune catégorie n'a de donnée du tout — c'est déjà ainsi que l'app se
// comporte, le widget doit rester identique plutôt que de "corriger" ce choix.
function computeDailyScoreServer(d) {
  const parts = [];
  if (d.gym.total > 0) parts.push(d.gym.done / d.gym.total);
  if (d.todos.total > 0) parts.push(d.todos.done / d.todos.total);
  if (d.goals.focus) parts.push((d.goals.focus.progress || 0) / 100);
  parts.push(d.meals.loggedTypes.size / d.meals.total);
  if (d.steps.goal > 0) parts.push(Math.min(1, d.steps.today / d.steps.goal));
  if (!parts.length) return 0;
  return Math.round((parts.reduce((s, v) => s + v, 0) / parts.length) * 100);
}

// Idempotence (section 7 du complément) : un document est CRÉÉ (jamais mis à
// jour) pour chaque occurrence exacte (utilisateur + catégorie + date locale +
// heure programmée). .create() échoue si le document existe déjà — la garantie
// est donc atomique même si ce cycle tourne deux fois ou qu'un ancien process
// se chevauche avec le nouveau après un redéploiement.
async function claimNotificationOccurrence(uid, catId, dateIso, time) {
  const id = `${uid}_${catId}_${dateIso}_${time}`;
  try {
    await fsDb().collection('notificationOccurrences').doc(id).create({
      uid, catId, dateIso, time, sentAt: admin.firestore.FieldValue.serverTimestamp()
    });
    return true;
  } catch (e) {
    if (e && (e.code === 6 || e.code === 'already-exists')) return false; // déjà traitée
    console.error('[NOTIF] claimNotificationOccurrence a échoué', uid, catId, e);
    return false; // par prudence : on n'envoie pas si l'idempotence n'a pas pu être vérifiée
  }
}

// Envoie le Push à TOUS les appareils enregistrés de l'utilisateur (section 10 :
// multi-appareils) et retire uniquement le(s) token(s) devenu(s) invalide(s)
// (section 11) — jamais les tokens des autres appareils.
async function sendPushToUserDevices(uid, title, body, catId, tab) {
  const ref = fsDb().collection('users').doc(uid).collection('data').doc('mrp-fcm-devices');
  const snap = await ref.get();
  const devices = snap.exists ? (snap.data().value || {}) : {};
  const deviceIds = Object.keys(devices);
  if (!deviceIds.length) return;

  let devicesChanged = false;
  await Promise.all(deviceIds.map(async (deviceId) => {
    const device = devices[deviceId];
    if (!device || !device.token) return;
    try {
      await admin.messaging().send({
        token: device.token,
        notification: { title, body },
        data: { tab: tab || '', catId }
      });
    } catch (e) {
      const code = e && e.errorInfo && e.errorInfo.code;
      if (code === 'messaging/registration-token-not-registered' || code === 'messaging/invalid-registration-token') {
        delete devices[deviceId];
        devicesChanged = true;
      } else {
        console.error('[NOTIF] Envoi FCM échoué pour', uid, deviceId, e);
      }
    }
  }));
  if (devicesChanged) await ref.set({ value: devices });
}

// Point d'entrée du cycle, exécuté chaque minute (voir setInterval plus bas) —
// PAS un setTimeout/setInterval frontend (interdit section 4 du complément :
// un onglet fermé arrête tout JS navigateur) : ceci tourne dans le process
// Node du backend Render, qui reste actif indépendamment de RoutinePOV. C'est
// le même principe que cron lui-même (une vérification par minute), simplement
// implémenté ici plutôt que délégué à Cloud Scheduler (voir choix d'architecture).
let notificationTickRunning = false;
async function runScheduledNotificationsTick() {
  if (!firebaseAdminReady) return; // FIREBASE_SERVICE_ACCOUNT_JSON non configuré
  if (notificationTickRunning) return; // évite un chevauchement si un cycle précédent traîne
  notificationTickRunning = true;
  try {
    const listResult = await admin.auth().listUsers(1000);
    await Promise.all(listResult.users.map(async (userRecord) => {
      const uid = userRecord.uid;
      try {
        const dataCol = fsDb().collection('users').doc(uid).collection('data');
        const [settingsSnap, tzSnap] = await Promise.all([
          dataCol.doc('mrp-notif-settings').get(),
          dataCol.doc('mrp-timezone').get()
        ]);
        const settings = settingsSnap.exists ? (settingsSnap.data().value || null) : null;
        if (!settings || !settings.enabled) return; // notifications désactivées pour cet utilisateur

        const timezone = tzSnap.exists ? tzSnap.data().value : null;
        const { dateIso, hhmm, weekdayFr } = getLocalContext(timezone);

        const dueCategories = NOTIFICATION_SCHEDULE.filter(cat => cat.times.includes(hhmm) && settings[cat.settingsFlag]);
        if (!dueCategories.length) return;

        const d = await buildServerDashboardData(uid, dateIso, weekdayFr);
        for (const cat of dueCategories) {
          if (!cat.isRelevant(d)) continue;
          const canSend = await claimNotificationOccurrence(uid, cat.id, dateIso, hhmm);
          if (!canSend) continue;
          await sendPushToUserDevices(uid, cat.title, cat.build(d), cat.id, cat.tab);
        }
      } catch (e) {
        console.error('[NOTIF] Cycle échoué pour', uid, e);
      }
    }));
  } catch (e) {
    console.error('[NOTIF] Cycle de notifications programmées échoué', e);
  } finally {
    notificationTickRunning = false;
  }
}

setInterval(runScheduledNotificationsTick, 60 * 1000);

// ============================================================================
// ⚠️ LIMITATION RÉELLE — Plan gratuit Render (section 8/23) : Render met ce
// serveur en veille après une période d'inactivité (voir le commentaire déjà
// présent dans index.html, window.api). Pendant que le serveur dort, le
// setInterval ci-dessus ne tourne PAS : aucune notification n'est déclenchée à
// l'heure programmée tant que personne n'a fait de requête récente. C'est une
// vraie limitation d'infrastructure, pas un détail — voir le rapport final pour
// les options (plan payant "always-on", ou déclencheur externe ci-dessous).
//
// Route de secours : un service de "cron externe" GRATUIT (cron-job.org,
// UptimeRobot, GitHub Actions planifiée...) peut appeler cette URL chaque
// minute pour réveiller le serveur ET déclencher le cycle immédiatement, sans
// attendre le setInterval. Protégée par un secret partagé lu depuis une
// variable d'environnement (JAMAIS exposée au frontend) — sans le bon secret,
// répond 404 (ne révèle même pas que la route existe).
// ============================================================================
app.get('/api/notifications/tick', async (req, res) => {
  if (!process.env.NOTIFICATIONS_TICK_SECRET || req.query.key !== process.env.NOTIFICATIONS_TICK_SECRET) {
    return res.status(404).end();
  }
  await runScheduledNotificationsTick();
  res.json({ ok: true });
});

// ============================================================================
// MODE TEST TEMPORAIRE (section 9 du complément) — À SUPPRIMER une fois la
// vérification terminée : retire uniquement cette route (et le commentaire),
// rien d'autre n'en dépend. Envoie un vrai Push FCM à l'UTILISATEUR CONNECTÉ
// (jamais à un autre uid), immédiatement ou dans quelques minutes, en
// contournant les conditions métier — sert uniquement à vérifier que la chaîne
// Firestore → Backend → FCM → Service Worker → téléphone fonctionne de bout en
// bout, sans attendre un horaire réel.
// ============================================================================
app.post('/api/notifications/test', requireFirebaseAuth, async (req, res) => {
  const delayMinutes = Math.max(0, Math.min(30, Number(req.body?.delayMinutes) || 0));
  const run = async () => {
    try {
      await sendPushToUserDevices(req.uid, '🔔 Test RoutinePOV', 'Cette notification confirme que la chaîne Push fonctionne.', 'test', null);
    } catch (e) {
      console.error('[NOTIF][TEST] échec', req.uid, e);
    }
  };
  if (delayMinutes === 0) {
    await run();
    return res.json({ success: true, sentAt: 'now' });
  }
  setTimeout(run, delayMinutes * 60 * 1000); // usage ponctuel de test, process serveur persistant — voir avertissement ci-dessus
  res.json({ success: true, scheduledInMinutes: delayMinutes });
});

app.listen(PORT, () => console.log(`Serveur prêt sur le port ${PORT}`));