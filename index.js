import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const app = express();
// Limite relevée par rapport au défaut (100kb) : les photos de repas compressées
// en base64 (voir resizeImageFile côté frontend) dépassent largement 100kb.
app.use(express.json({ limit: '10mb' }));
app.use(cors({ origin: process.env.CLIENT_URL || '*' }));

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post('/api/generate-recipe', async (req, res) => {
  try {
    const { ingredients } = req.body;
    if (!ingredients) return res.status(400).json({ error: "Ingrédients manquants" });

    const response = await ai.models.generateContent({
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

    const response = await ai.models.generateContent({
      model: 'gemini-3.5-flash',
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
    });

    const parsed = JSON.parse(response.text);
    res.json({ success: true, items: Array.isArray(parsed.items) ? parsed.items : [] });
  } catch (error) {
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

    const response = await ai.models.generateContent({
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

    const response = await ai.models.generateContent({
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