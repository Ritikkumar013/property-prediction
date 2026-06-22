const path = require("path");
const fs = require("fs");

// ── Load XGBoost-trained calibration data and model weights ──────────────────
const mlDir = path.join(__dirname, "..", "ml");
const calibrationData = JSON.parse(
  fs.readFileSync(path.join(mlDir, "calibration_data.json"), "utf-8")
);
const modelWeights = JSON.parse(
  fs.readFileSync(path.join(mlDir, "model_weights.json"), "utf-8")
);

// Build a lookup index from calibration data for fast interpolation
const calibrationIndex = {};
for (const entry of calibrationData) {
  const key = `${entry.corridor}|${entry.bhk}|${entry.furnishing}`;
  if (!calibrationIndex[key]) calibrationIndex[key] = [];
  calibrationIndex[key].push({ sqft: entry.sqft, price: entry.predicted_price });
}
for (const key of Object.keys(calibrationIndex)) {
  calibrationIndex[key].sort((a, b) => a.sqft - b.sqft);
}

const CORRIDOR_CONFIG = {
  "dehu-solapur": {
    name: "Dehu Road to Solapur Road",
    basePricePerSqft: 5800,
    locations: [
      { name: "Dehu Road", multiplier: 0.85 },
      { name: "Pimpri-Chinchwad", multiplier: 0.95 },
      { name: "Hinjewadi", multiplier: 1.15 },
      { name: "Baner", multiplier: 1.25 },
      { name: "Kothrud", multiplier: 1.20 },
      { name: "Shivajinagar", multiplier: 1.10 },
      { name: "Koregaon Park", multiplier: 1.30 },
      { name: "Hadapsar", multiplier: 0.90 },
      { name: "Solapur Road", multiplier: 0.80 },
    ],
    avgPriceRange: { min: 25, max: 180 },
    demandFactor: 1.05,
  },
  "kolhapur-nashik": {
    name: "Kolhapur Road to Nashik Road",
    basePricePerSqft: 4200,
    locations: [
      { name: "Katraj", multiplier: 0.88 },
      { name: "Dhankawadi", multiplier: 0.82 },
      { name: "Sinhagad Road", multiplier: 0.95 },
      { name: "Warje", multiplier: 1.05 },
      { name: "Khadki", multiplier: 0.92 },
      { name: "Aundh", multiplier: 1.15 },
      { name: "Vishrantwadi", multiplier: 0.90 },
      { name: "Moshi", multiplier: 0.78 },
      { name: "Nashik Road", multiplier: 0.75 },
    ],
    avgPriceRange: { min: 18, max: 130 },
    demandFactor: 0.95,
  },
};

const AMENITY_WEIGHTS = {
  gym: 0.02,
  swimming_pool: 0.035,
  clubhouse: 0.025,
  garden: 0.015,
  playground: 0.01,
  security: 0.02,
  power_backup: 0.015,
  lift: 0.02,
  parking: 0.015,
  cctv: 0.01,
};

// ── XGBoost-calibrated prediction via interpolation ──────────────────────────

function interpolate(points, targetSqft) {
  if (targetSqft <= points[0].sqft) {
    return (points[0].price * targetSqft) / points[0].sqft;
  }
  if (targetSqft >= points[points.length - 1].sqft) {
    const last = points[points.length - 1];
    const prev = points[points.length - 2];
    const slope = (last.price - prev.price) / (last.sqft - prev.sqft);
    return last.price + slope * (targetSqft - last.sqft);
  }
  for (let i = 0; i < points.length - 1; i++) {
    if (targetSqft >= points[i].sqft && targetSqft <= points[i + 1].sqft) {
      const ratio =
        (targetSqft - points[i].sqft) /
        (points[i + 1].sqft - points[i].sqft);
      return points[i].price + ratio * (points[i + 1].price - points[i].price);
    }
  }
  return points[0].price;
}

function getXGBoostBasePrice(corridor, bhk, sqft, furnishing) {
  const key = `${corridor}|${bhk}|${furnishing}`;
  const points = calibrationIndex[key];
  if (!points) return null;
  return interpolate(points, sqft);
}

function calculateFloorAdjustment(floor, totalFloors) {
  const ratio = floor / Math.max(totalFloors, 1);
  if (floor <= 0) return 0.95;
  if (ratio <= 0.3) return 1.0;
  if (ratio <= 0.6) return 1.03;
  if (ratio <= 0.85) return 1.06;
  return 1.04;
}

function calculateAgeAdjustment(age) {
  if (age <= 0) return 1.1;
  if (age <= 2) return 1.05;
  if (age <= 5) return 1.0;
  if (age <= 10) return 0.92;
  if (age <= 20) return 0.82;
  return 0.7;
}

function calculateBathroomAdjustment(bathrooms, bhk) {
  if (bathrooms >= bhk) return 1.04;
  if (bathrooms >= Math.max(1, bhk - 1)) return 1.0;
  return 0.96;
}

function predictPrice(input) {
  const {
    corridor,
    bhk,
    sqft,
    bathrooms,
    floor,
    totalFloors = 10,
    propertyAge = 0,
    furnishing = "unfurnished",
    parking = 0,
    amenities = [],
  } = input;

  const corridorData = CORRIDOR_CONFIG[corridor];
  if (!corridorData) {
    throw new Error(
      `Invalid corridor. Use: ${Object.keys(CORRIDOR_CONFIG).join(", ")}`
    );
  }

  // Get XGBoost-trained base prediction via calibration interpolation
  const clampedBhk = Math.min(Math.max(bhk, 1), 5);
  const xgbBase = getXGBoostBasePrice(corridor, clampedBhk, sqft, furnishing);

  let priceLakhs;
  const floorAdj = calculateFloorAdjustment(floor, totalFloors);
  const ageAdj = calculateAgeAdjustment(propertyAge);
  const bathroomAdj = calculateBathroomAdjustment(bathrooms, bhk);
  const parkingAdj = 1 + parking * 0.02;

  let amenityBonus = 1.0;
  const amenityList = Array.isArray(amenities) ? amenities : [];
  for (const amenity of amenityList) {
    amenityBonus += AMENITY_WEIGHTS[amenity] || 0;
  }

  if (xgbBase !== null) {
    // XGBoost calibration already accounts for corridor, bhk, sqft, furnishing
    // Apply remaining adjustments that weren't in the calibration baseline
    const calibrationBaseline = {
      floor: 1.03, // calibration used floor=5 of 10 (mid-high)
      age: 1.05, // calibration used age=2 (new construction)
      bathroom: 1.0, // calibration used bathrooms=bhk-1
      parking: 1.02, // calibration used parking=1
      amenity: 1.0 + 3 * 0.018, // calibration used 3 amenities (avg weight)
    };

    const adjustments =
      (floorAdj / calibrationBaseline.floor) *
      (ageAdj / calibrationBaseline.age) *
      (bathroomAdj / calibrationBaseline.bathroom) *
      (parkingAdj / calibrationBaseline.parking) *
      (amenityBonus / calibrationBaseline.amenity);

    priceLakhs = Math.round(xgbBase * adjustments * 100) / 100;
  } else {
    // Fallback: heuristic prediction if calibration data missing
    const basePrice = corridorData.basePricePerSqft * sqft;
    const bhkMult = { 1: 0.75, 2: 1.0, 3: 1.25, 4: 1.55, 5: 1.85 }[bhk] || 1.0;
    const furnMult = { unfurnished: 1.0, "semi-furnished": 1.08, furnished: 1.18 }[furnishing] || 1.0;
    const rawPrice =
      basePrice * bhkMult * floorAdj * ageAdj * bathroomAdj *
      furnMult * parkingAdj * amenityBonus * corridorData.demandFactor;
    priceLakhs = Math.round((rawPrice / 100000) * 100) / 100;
  }

  const variance = 0.05;
  const minPrice = Math.round(priceLakhs * (1 - variance) * 100) / 100;
  const maxPrice = Math.round(priceLakhs * (1 + variance) * 100) / 100;

  return {
    predictedPrice: priceLakhs,
    priceRange: { min: minPrice, max: maxPrice },
    currency: "INR",
    unit: "Lakhs",
    corridor: corridorData.name,
    model: {
      algorithm: modelWeights.model_info.algorithm,
      r2Score: modelWeights.model_info.r2_score,
      mae: modelWeights.model_info.mae,
      trainingSamples: modelWeights.model_info.training_samples,
    },
    breakdown: {
      basePricePerSqft: corridorData.basePricePerSqft,
      sqft,
      xgboostBasePrice: xgbBase ? Math.round(xgbBase * 100) / 100 : null,
      floorAdjustment: floorAdj,
      ageAdjustment: ageAdj,
      bathroomAdjustment: bathroomAdj,
      furnishing,
      parkingAdjustment: parkingAdj,
      amenityBonus: Math.round(amenityBonus * 1000) / 1000,
      demandFactor: corridorData.demandFactor,
    },
  };
}

function compareCorridor(input) {
  const corridors = Object.keys(CORRIDOR_CONFIG);
  const results = {};

  for (const corridor of corridors) {
    results[corridor] = predictPrice({ ...input, corridor });
  }

  const keys = Object.keys(results);
  const diff =
    Math.round(
      Math.abs(results[keys[0]].predictedPrice - results[keys[1]].predictedPrice) * 100
    ) / 100;

  const cheaper =
    results[keys[0]].predictedPrice < results[keys[1]].predictedPrice
      ? keys[0]
      : keys[1];

  return {
    corridors: results,
    comparison: {
      priceDifference: diff,
      priceDifferenceUnit: "Lakhs",
      cheaperCorridor: cheaper,
      cheaperCorridorName: CORRIDOR_CONFIG[cheaper].name,
      percentageDifference:
        Math.round(
          (diff /
            Math.max(
              results[keys[0]].predictedPrice,
              results[keys[1]].predictedPrice
            )) *
            100 *
            100
        ) / 100,
    },
  };
}

function getCorridorStats(corridor) {
  const corridorData = CORRIDOR_CONFIG[corridor];
  if (!corridorData) {
    throw new Error(
      `Invalid corridor. Use: ${Object.keys(CORRIDOR_CONFIG).join(", ")}`
    );
  }

  const sampleConfigs = [
    { bhk: 1, sqft: 450 },
    { bhk: 2, sqft: 850 },
    { bhk: 3, sqft: 1200 },
    { bhk: 4, sqft: 1800 },
  ];

  const priceEstimates = sampleConfigs.map((config) => {
    const result = predictPrice({
      corridor,
      ...config,
      bathrooms: config.bhk,
      floor: 5,
    });
    return {
      type: `${config.bhk} BHK (${config.sqft} sq.ft)`,
      estimatedPrice: result.predictedPrice,
      unit: "Lakhs",
    };
  });

  return {
    corridor: corridorData.name,
    corridorKey: corridor,
    basePricePerSqft: corridorData.basePricePerSqft,
    locations: corridorData.locations,
    avgPriceRange: corridorData.avgPriceRange,
    demandFactor: corridorData.demandFactor,
    mlModel: modelWeights.model_info,
    featureImportance: modelWeights.feature_importance,
    sampleEstimates: priceEstimates,
  };
}

function getAllCorridors() {
  return Object.entries(CORRIDOR_CONFIG).map(([key, data]) => ({
    key,
    name: data.name,
    basePricePerSqft: data.basePricePerSqft,
    avgPriceRange: data.avgPriceRange,
    locationCount: data.locations.length,
  }));
}

module.exports = {
  predictPrice,
  compareCorridor,
  getCorridorStats,
  getAllCorridors,
  CORRIDOR_CONFIG,
};
