const prisma = require("../prisma/prisma");
const {
  predictPrice,
  compareCorridor,
  getCorridorStats,
  getAllCorridors,
} = require("../services/priceModel");

const createProperty = async (req, res) => {
  try {
    const property = await prisma.property.create({
      data: req.body,
    });

    res.status(201).json({
      success: true,
      data: property,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const getProperties = async (req, res) => {
  try {
    const { corridor } = req.query;
    const where = corridor ? { corridor } : {};
    const properties = await prisma.property.findMany({
      where,
      orderBy: { createdAt: "desc" },
    });

    res.status(200).json({
      success: true,
      count: properties.length,
      data: properties,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const predict = async (req, res) => {
  try {
    const {
      corridor,
      bhk,
      sqft,
      bathrooms,
      floor,
      totalFloors,
      propertyAge,
      furnishing,
      parking,
      amenities,
    } = req.body;

    if (!corridor || !bhk || !sqft || !bathrooms || floor === undefined) {
      return res.status(400).json({
        success: false,
        message: "Required fields: corridor, bhk, sqft, bathrooms, floor",
      });
    }

    const prediction = predictPrice({
      corridor,
      bhk: Number(bhk),
      sqft: Number(sqft),
      bathrooms: Number(bathrooms),
      floor: Number(floor),
      totalFloors: totalFloors ? Number(totalFloors) : undefined,
      propertyAge: propertyAge ? Number(propertyAge) : undefined,
      furnishing,
      parking: parking ? Number(parking) : undefined,
      amenities,
    });

    const saved = await prisma.property.create({
      data: {
        corridor,
        bhk: Number(bhk),
        sqft: Number(sqft),
        bathrooms: Number(bathrooms),
        floor: Number(floor),
        totalFloors: totalFloors ? Number(totalFloors) : 10,
        propertyAge: propertyAge ? Number(propertyAge) : 0,
        furnishing: furnishing || "unfurnished",
        parking: parking ? Number(parking) : 0,
        amenities: amenities || [],
        predictedPrice: prediction.predictedPrice,
      },
    });

    res.status(200).json({
      success: true,
      data: {
        id: saved.id,
        ...prediction,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const compare = async (req, res) => {
  try {
    const {
      bhk,
      sqft,
      bathrooms,
      floor,
      totalFloors,
      propertyAge,
      furnishing,
      parking,
      amenities,
    } = req.body;

    if (!bhk || !sqft || !bathrooms || floor === undefined) {
      return res.status(400).json({
        success: false,
        message: "Required fields: bhk, sqft, bathrooms, floor",
      });
    }

    const result = compareCorridor({
      bhk: Number(bhk),
      sqft: Number(sqft),
      bathrooms: Number(bathrooms),
      floor: Number(floor),
      totalFloors: totalFloors ? Number(totalFloors) : undefined,
      propertyAge: propertyAge ? Number(propertyAge) : undefined,
      furnishing,
      parking: parking ? Number(parking) : undefined,
      amenities,
    });

    res.status(200).json({
      success: true,
      data: result,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const corridorStats = async (req, res) => {
  try {
    const { corridor } = req.params;

    if (!corridor) {
      return res.status(400).json({
        success: false,
        message: "Corridor parameter is required",
      });
    }

    const stats = getCorridorStats(corridor);

    const dbStats = await prisma.property.aggregate({
      where: { corridor },
      _avg: { predictedPrice: true, sqft: true },
      _count: true,
      _min: { predictedPrice: true },
      _max: { predictedPrice: true },
    });

    res.status(200).json({
      success: true,
      data: {
        ...stats,
        historicalData: {
          totalPredictions: dbStats._count,
          avgPredictedPrice: dbStats._avg.predictedPrice
            ? Math.round(dbStats._avg.predictedPrice * 100) / 100
            : null,
          avgSqft: dbStats._avg.sqft
            ? Math.round(dbStats._avg.sqft * 100) / 100
            : null,
          minPrice: dbStats._min.predictedPrice,
          maxPrice: dbStats._max.predictedPrice,
        },
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const corridors = async (req, res) => {
  try {
    const data = getAllCorridors();
    res.status(200).json({
      success: true,
      data,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

module.exports = {
  createProperty,
  getProperties,
  predict,
  compare,
  corridorStats,
  corridors,
};
