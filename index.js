import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const app = express();
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

app.post('/api/generate-recipe', async (req, res) => {
  try {
    const { ingredients } = req.body;
    if (!ingredients) return res.status(400).json({ error: "Ingrédients manquants" });

    const response = await generateContentWithRetry({
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
    });

    res.json({ success: true, recipe: JSON.parse(response.text) });
  } catch (error) {
    res.status(500).json({ error: "Erreur serveur" });
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

app.post('/api/analyze-meal-photo', async (req, res) => {
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
            "estime son nom, sa quantité approximative, ses calories (kcal), ses " +
            "protéines, glucides et lipides (en grammes). Indique 'low' comme " +
            "confidence si tu n'es pas sûr d'un aliment ou d'une quantité, sinon " +
            "'medium' ou 'high'."
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
                required: ["name"]
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
app.post('/api/evaluate-life-goal', async (req, res) => {
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
app.post('/api/analyze-workout', async (req, res) => {
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
app.post('/api/assistant-chat', async (req, res) => {
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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Serveur prêt sur le port ${PORT}`));