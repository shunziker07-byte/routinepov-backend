import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors({ origin: process.env.CLIENT_URL || '*' }));

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.post('/api/generate-recipe', async (req, res) => {
  try {
    const { ingredients } = req.body;
    if (!ingredients) return res.status(400).json({ error: "Ingrédients manquants" });

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Serveur prêt sur le port ${PORT}`));