# AI-Powered Property Price Predictor - Pune

A full-stack web application that predicts residential property prices in Pune, India, focusing on two major corridors. Built with React.js, Node.js/Express, PostgreSQL, and an XGBoost-trained ML model.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How the ML Model Works](#how-the-ml-model-works)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Deployment](#deployment)

---

## Project Overview

This application predicts residential property prices (in ₹ Lakhs) across two key Pune corridors:

1. **Dehu Road to Solapur Road** — Covers Dehu Road, Pimpri-Chinchwad, Hinjewadi, Baner, Kothrud, Shivajinagar, Koregaon Park, Hadapsar, Solapur Road
2. **Kolhapur Road to Nashik Road** — Covers Katraj, Dhankawadi, Sinhagad Road, Warje, Khadki, Aundh, Vishrantwadi, Moshi, Nashik Road

Users fill in property details and get an instant AI-powered price prediction with a detailed breakdown of factors that influence the price.

---

## Architecture

```
┌─────────────────────┐     HTTP      ┌─────────────────────┐     Prisma     ┌──────────────┐
│                     │   requests    │                     │    queries     │              │
│   Frontend          │ ──────────>   │   Backend API       │ ──────────>    │  PostgreSQL  │
│   (Next.js 16)      │              │   (Express.js)      │               │  (Neon)      │
│   Tailwind CSS v4   │   <──────    │   Port 5000         │   <──────     │              │
│   Port 3000         │   JSON       │                     │   results     │              │
│                     │              │   ┌───────────────┐ │               └──────────────┘
└─────────────────────┘              │   │ Price Model   │ │
                                     │   │ (XGBoost      │ │
                                     │   │  calibration) │ │
                                     │   └───────────────┘ │
                                     └─────────────────────┘
                                               ▲
                                               │ trained weights
                                     ┌─────────────────────┐
                                     │   ML Training        │
                                     │   (Python)           │
                                     │   XGBoost +          │
                                     │   scikit-learn       │
                                     │   (offline, one-time)│
                                     └─────────────────────┘
```

### How data flows:

1. User fills the property form on the frontend (corridor, BHK, sq.ft, etc.)
2. Frontend sends a POST request to the Express API
3. The API's price model loads pre-trained XGBoost calibration data and interpolates the base price
4. Additional adjustments (floor, age, bathrooms, parking, amenities) are applied
5. The prediction is saved to PostgreSQL and returned to the frontend
6. Frontend displays the price with an interactive breakdown

---

## Features

### 1. Predict Price Tab
- Select one of the two Pune corridors
- Fill property details: BHK (1-5), area in sq.ft, bathrooms, floor, total floors, property age, furnishing type, parking spots
- Toggle amenities: Gym, Swimming Pool, Clubhouse, Garden, Playground, 24/7 Security, Power Backup, Lift, Covered Parking, CCTV
- Get instant prediction in ₹ Lakhs with confidence range (±5%)
- View detailed price breakdown showing impact of each factor

### 2. Compare Corridors Tab
- Enter property specs once
- See side-by-side price comparison for both corridors
- "Better Value" badge on the cheaper corridor
- Price difference in Lakhs and percentage
- Visual factor comparison bars (base price, demand, floor premium, furnishing)

### 3. Explore Tab
- Switch between corridors to view detailed stats
- See base price per sq.ft, demand factor, and number of key locations
- Sample price estimates for 1-4 BHK configurations (bar chart)
- Location-wise price multiplier grid (green = premium, grey = affordable)
- Historical prediction data from the database

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Next.js 16, React 19, TypeScript | UI framework with App Router |
| Styling | Tailwind CSS v4 | Utility-first CSS |
| Backend | Express.js 5 | REST API server |
| Database | PostgreSQL (Neon) | Stores property predictions |
| ORM | Prisma | Database queries and schema management |
| ML Training | Python, XGBoost, scikit-learn, pandas | Model training (offline) |
| ML Inference | Node.js (calibration interpolation) | Runtime predictions using trained weights |

---

## How the ML Model Works

### Training (Python — one-time, offline)

The file `ml/train_model.py` handles model training:

1. **Data Generation**: Creates 10,000 synthetic but realistic Pune property records with proper distributions:
   - BHK distribution: 10% 1BHK, 35% 2BHK, 30% 3BHK, 18% 4BHK, 7% 5BHK
   - Realistic sq.ft ranges per BHK (e.g., 2BHK = 650-1100 sq.ft)
   - 9 sub-locations per corridor with individual price multipliers
   - Noise factor (±8%) for realistic variance

2. **Feature Engineering**: 21 features including:
   - Corridor (label encoded), BHK, sq.ft, bathrooms, floor, total floors
   - Property age, furnishing (label encoded), parking spots
   - 10 binary amenity flags + amenity count
   - Floor ratio (floor / total floors)

3. **Model Training**: XGBoost Regressor with:
   - 200 estimators, max depth 6, learning rate 0.1
   - 80/20 train-test split
   - **Results: R² = 0.9321 (93.2% accuracy), MAE = ₹16.3 Lakhs**

4. **Export**: Three files are generated:
   - `xgboost_model.json` — Full XGBoost model dump (200 trees)
   - `model_weights.json` — Model metadata, feature importance, label encodings
   - `calibration_data.json` — 240 pre-computed XGBoost predictions for key property configurations

### Inference (Node.js — runtime)

The file `services/priceModel.js` handles predictions at runtime:

1. Loads the 240 calibration points from `calibration_data.json` into a lookup index
2. For a given input, finds the matching corridor + BHK + furnishing calibration curve
3. **Interpolates** the XGBoost-trained base price for the exact sq.ft value
4. Applies **adjustment ratios** for factors not in the calibration baseline:
   - Floor position premium (ground floor: -5%, mid-high: +3-6%)
   - Property age depreciation (new: +10%, 10yr: -8%, 20yr+: -18%)
   - Bathroom-to-BHK ratio adjustment
   - Parking bonus (+2% per spot)
   - Amenity bonus (each amenity adds 1-3.5%)

This approach gives us ML-quality predictions without needing Python at runtime.

### Feature Importance (from XGBoost)

| Feature | Importance | Impact |
|---------|-----------|--------|
| BHK | 48.6% | Strongest predictor |
| Corridor | 16.8% | Location premium |
| Sq.ft | 15.2% | Size-price correlation |
| Bathrooms | 7.7% | Utility/luxury signal |
| Property Age | 1.9% | Depreciation |
| Furnishing | 1.3% | Fit-out premium |
| Swimming Pool | 0.8% | Top amenity |
| Others | ~7% | Remaining factors |

---

## API Endpoints

| Method | Endpoint | Description | Body |
|--------|----------|-------------|------|
| `GET` | `/` | API health check | — |
| `GET` | `/api/properties` | List all saved predictions | — |
| `POST` | `/api/properties` | Create a property record | `{ corridor, bhk, sqft, ... }` |
| `POST` | `/api/properties/predict` | **Predict price** | `{ corridor, bhk, sqft, bathrooms, floor, ... }` |
| `POST` | `/api/properties/compare` | **Compare both corridors** | `{ bhk, sqft, bathrooms, floor, ... }` |
| `GET` | `/api/properties/corridors` | List all corridors | — |
| `GET` | `/api/properties/corridors/:key` | Corridor stats + sample prices | — |

### Example: Predict Price

**Request:**
```json
POST /api/properties/predict
{
  "corridor": "dehu-solapur",
  "bhk": 3,
  "sqft": 1200,
  "bathrooms": 3,
  "floor": 8,
  "totalFloors": 15,
  "furnishing": "furnished",
  "amenities": ["gym", "swimming_pool", "security"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 3,
    "predictedPrice": 146.85,
    "priceRange": { "min": 139.51, "max": 154.19 },
    "currency": "INR",
    "unit": "Lakhs",
    "corridor": "Dehu Road to Solapur Road",
    "model": {
      "algorithm": "XGBoost (XGBRegressor)",
      "r2Score": 0.9321,
      "mae": 16.3052,
      "trainingSamples": 8000
    },
    "breakdown": {
      "basePricePerSqft": 5800,
      "sqft": 1200,
      "xgboostBasePrice": 134.79,
      "floorAdjustment": 1.03,
      "ageAdjustment": 1.1,
      "bathroomAdjustment": 1.04,
      "furnishing": "furnished",
      "parkingAdjustment": 1,
      "amenityBonus": 1.075,
      "demandFactor": 1.05
    }
  }
}
```

---

## Project Structure

```
property-predictor-api/          # Backend
├── server.js                    # Express entry point (port 5000)
├── package.json                 # Node.js dependencies
├── .env                         # DATABASE_URL (PostgreSQL)
├── .env.example                 # Template for env vars
├── .gitignore
│
├── controllers/
│   └── property.controller.js   # Route handlers (predict, compare, CRUD)
│
├── routes/
│   └── property.routes.js       # API route definitions
│
├── services/
│   └── priceModel.js            # XGBoost-calibrated prediction engine
│
├── prisma/
│   ├── schema.prisma            # Database schema (Property model)
│   └── prisma.js                # Prisma client instance
│
└── ml/                          # Machine Learning
    ├── train_model.py           # XGBoost training script (Python)
    ├── requirements.txt         # Python dependencies
    ├── xgboost_model.json       # Trained XGBoost model (200 trees)
    ├── model_weights.json       # Feature importance + encodings
    └── calibration_data.json    # 240 pre-computed predictions

my-app/                          # Frontend
├── app/
│   ├── layout.tsx               # Root layout with fonts
│   ├── page.tsx                 # Main page (3 tabs)
│   └── globals.css              # Tailwind + custom animations
│
├── components/
│   ├── PredictionForm.tsx       # Property input form
│   ├── PredictionResult.tsx     # Price display + breakdown bars
│   ├── ComparisonView.tsx       # Side-by-side corridor comparison
│   └── CorridorStats.tsx        # Corridor explorer with charts
│
├── lib/
│   └── api.ts                   # API client + TypeScript types
│
├── package.json
├── next.config.ts
├── tsconfig.json
└── postcss.config.mjs
```

---

## Setup & Installation

### Prerequisites
- Node.js 18+
- Python 3.9+ (for ML training only)
- PostgreSQL database (or use Neon free tier)

### Backend Setup

```bash
cd property-predictor-api

# Install dependencies
npm install

# Set up environment
cp .env.example .env
# Edit .env with your DATABASE_URL

# Push schema to database
npx prisma db push

# Start server
npm run dev
# → Running on http://localhost:5000
```

### Frontend Setup

```bash
cd my-app

# Install dependencies
npm install

# Start dev server
npm run dev
# → Running on http://localhost:3000
```

### Retrain the ML Model (optional)

```bash
cd property-predictor-api/ml

# Install Python dependencies
pip install -r requirements.txt

# Train and export
python train_model.py
# → Generates xgboost_model.json, model_weights.json, calibration_data.json
```

---

## Deployment

### Backend → Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo, set root directory to `property-predictor-api`
4. Build command: `npm install`
5. Start command: `npm start`
6. Add environment variable: `DATABASE_URL` = your PostgreSQL connection string

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → Import Project
2. Connect your GitHub repo, set root directory to `my-app`
3. Add environment variable: `NEXT_PUBLIC_API_URL` = your Render backend URL
4. Deploy

---

## Summary

This project demonstrates a complete AI-powered property prediction system:

- **ML Pipeline**: Python + XGBoost + scikit-learn trains on 10,000 Pune property samples (R² = 93.2%)
- **Smart Inference**: Trained model weights are exported and used in Node.js via calibration interpolation — no Python needed at runtime
- **Full-Stack App**: React frontend with interactive visualizations, Express API with PostgreSQL persistence
- **Two Corridors**: Covers 18 key Pune locations across Dehu-Solapur and Kolhapur-Nashik corridors
- **Rich Features**: Price prediction, corridor comparison, location explorer, factor breakdowns
